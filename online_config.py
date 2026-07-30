#!/usr/bin/env python3
"""
Shared configuration + base-model builder for per-drone ONLINE learning.

Ties the online stack (adapter_manager / online_learner / streaming / predictor)
to the validated 40-frame base and the correct upstream-only LoRA config (NO
delta_head, per-target ranks).

Deployment shape:
    40-frame base  ->  per-drone LoRA online

Global LoRA (dir_lora_40) is DISABLED by default — it was trained on the full
UAV-Flow window set (data leakage) and its cross-flight generalization is modest
(+7.3% FDE). It remains available as opt-in via with_global=True / use_global=True
for in-distribution use, but the recommended deployment path is base + per-drone
online LoRA only.
"""

import torch
from pathlib import Path

WEIGHT_DIR = Path(__file__).parent / 'weights'

# 40-frame model geometry
HIST_LEN = 40
PRED_LEN = 20
SCALE_POS = 100.0
SCALE_VEL = 10.0

# Validated per-drone LoRA targets (upstream only, NO delta_head -> no zigzag).
# (path, rank); alpha defaults to 2*rank inside LoRAAdapter.
ONLINE_LORA_TARGETS = [
    ('emam_se.mamba_blocks.0.ssm.in_proj', 24),
    ('emam_se.mamba_blocks.0.ssm.out_proj', 24),
    ('emam_se.mamba_blocks.1.ssm.in_proj', 24),
    ('emam_se.mamba_blocks.1.ssm.out_proj', 24),
    ('ua_pgd.feat_compress', 96),
    ('ua_pgd.neural_decoder.proj.0', 64),
]
ONLINE_HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

# Global LoRA to merge into the base before per-drone adaptation.
# dir_lora_40 is the current best global LoRA (+19.4% FDE, dir 13.8->12.1deg).
GLOBAL_LORA_FILE = 'dir_lora_40.pth'
BASE_40_FILE = 'low_speed_6class_40frame.pth'
MULTIHEAD_40_FILE = 'low_multihead_K5_40frame.pth'  # K=5 WTA decoder

# HIGH model (SimCruise, 1Hz, 4-class) — online learning uses base + per-drone
# LoRA only, NO global layer (there is no HIGH global LoRA; 40-frame expansion
# was rejected in session 3, so HIGH stays 20-frame).
HIGH_BASE_FILE = 'high_speed_4class.pth'
HIGH_HIST_LEN = 20
HIGH_N_CLASSES = 4
HIGH_DT = 1.0

# ── Deployment gating ───────────────────────────────────────────────────────
# Online per-drone learning only kicks in on long enough flights; short ones use
# the plain base. Below this threshold, no per-drone LoRA is created or trained.
ONLINE_MIN_FRAMES = 60      # rolling-buffer frames before per-drone learning turns on

# Global LoRA speed gate (only relevant when opt-in use_global=True).
# dir_lora_40 was trained on 0-3 m/s real DJI flight; faster motion is out of its domain.
GLOBAL_MAX_SPEED = 4.0      # m/s; above this the low-speed global LoRA is switched off


def _resolve(model, path):
    obj = model
    for p in path.split('.'):
        obj = getattr(obj, p)
    return obj


def _set(model, path, mod):
    parts = path.split('.'); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)


def build_base_model(num_intent_classes=6, device='cuda',
                     weight_file=BASE_40_FILE, hist_len=HIST_LEN):
    """Plain base model (no global LoRA). Defaults to the 40-frame LOW base."""
    from emam_model import TrajectoryPredictor
    m = TrajectoryPredictor(
        input_dim=6, history_len=hist_len, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=num_intent_classes,
        use_trigger=True, trigger_mode='simple',
    ).to(device).eval()
    ckpt = torch.load(WEIGHT_DIR / weight_file, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])
    return m


def build_high_base(device='cuda'):
    """Plain HIGH base (20-frame, 4-class SimCruise). NO global layer — HIGH
    online learning is base + per-drone LoRA only."""
    return build_base_model(num_intent_classes=HIGH_N_CLASSES, device=device,
                            weight_file=HIGH_BASE_FILE, hist_len=HIGH_HIST_LEN)


def merge_global_lora(model, global_file=GLOBAL_LORA_FILE, device='cuda'):
    """Fold a saved global LoRA into the base weights in place (merge-then-train).

    Uses the validated per-target LoRA structure. After this, the base Linears
    carry the global adaptation and per-drone LoRA stacks cleanly on top.
    Returns True on success, False if the global file is missing.
    """
    from lora import LoRALinear
    path = WEIGHT_DIR / global_file
    if not path.exists():
        return False
    g = torch.load(path, map_location=device)
    lls = {}
    for p, rank in ONLINE_LORA_TARGETS:
        orig = _resolve(model, p)
        lora = LoRALinear(orig, r=rank, alpha=rank * 2.0)
        _set(model, p, lora); lls[p] = lora
    for p, mats in g['lora_state'].items():
        if p in lls:
            lls[p].lora_A.data.copy_(mats['A'].to(device))
            lls[p].lora_B.data.copy_(mats['B'].to(device))
    for key, t in g.get('head_state', {}).items():
        pth, attr = key.rsplit('.', 1)
        lay = _resolve(model, pth)
        if attr == 'weight':
            lay.weight.data.copy_(t.to(device))
    for p, lora in lls.items():
        lora.merge()
        _set(model, p, lora.base_layer)
    return True


def build_online_base(num_intent_classes=6, device='cuda', with_global=False):
    """Build the deployment base: 40-frame model.

    with_global=False (default): plain 40-frame base — recommended.
    with_global=True (opt-in): merge global LoRA (dir_lora_40) for in-distribution use.
    """
    m = build_base_model(num_intent_classes=num_intent_classes, device=device)
    if with_global:
        merge_global_lora(m, device=device)
    return m


def build_multihead_base(num_intent_classes=6, device='cuda', K=5):
    """Build 40-frame base with K-hypothesis WTA decoder.

    Loads the single-head base weights first, then replaces the decoder with
    MultiHeadNeuralDecoder and loads the pre-trained K=5 decoder weights.
    Per-drone LoRA stacks cleanly on top (targets are in the encoder + shared
    projection, not in the K delta heads).
    """
    from emam_model import TrajectoryPredictor
    m = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=num_intent_classes,
        use_trigger=True, trigger_mode='simple',
    ).to(device).eval()
    ckpt = torch.load(WEIGHT_DIR / BASE_40_FILE, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])

    # Replace decoder with multi-head variant
    m.ua_pgd.replace_with_multi_head(K=K, noise_std=0.0)
    mh_ckpt = torch.load(WEIGHT_DIR / MULTIHEAD_40_FILE, map_location=device)
    m.ua_pgd.neural_decoder.load_state_dict(mh_ckpt['multi_decoder_state'])
    # replace_with_multi_head creates the new decoder on CPU — move it to device
    m.ua_pgd.neural_decoder = m.ua_pgd.neural_decoder.to(device)
    return m
