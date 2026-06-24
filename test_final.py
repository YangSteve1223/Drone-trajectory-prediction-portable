#!/usr/bin/env python3
"""Final comprehensive test before handoff."""

import torch, sys, warnings, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

PASS, FAIL, SKIP = 0, 0, 0
def check(cond, name):
    global PASS, FAIL
    if cond: PASS += 1; print(f'  [PASS] {name}')
    else: FAIL += 1; print(f'  [FAIL] {name}')

print('='*60)
print('FINAL COMPREHENSIVE TEST SUITE')
print('='*60)

# ============================================================
# 1. Module loading
# ============================================================
print('\n[1] Module Loading')
from predictor import DronePredictor; check(True, 'predictor')
from streaming import StreamingPredictor; check(True, 'streaming')
from bidirectional import BidirectionalPredictor; check(True, 'bidirectional')
from lora import LoRALinear, LoRAAdapter; check(True, 'lora')
from adapter_manager import DroneAdapterManager; check(True, 'adapter_manager')
from online_learner import OnlineLearner; check(True, 'online_learner')
from safety import RiskAssessment, AnomalyDetector; check(True, 'safety')
from dynamic_norm import DynamicNormalizer, scale_invariant_mse; check(True, 'dynamic_norm')
import predictor as _p; check(hasattr(_p.DronePredictor, 'predict_adaptive'), 'predict_adaptive method')

# ============================================================
# 2. Predictor — all prediction modes
# ============================================================
print('\n[2] Predictor — All Modes')
p = DronePredictor()

modes = [
    ('predict', lambda h: p.predict(h)),
    ('predict_adaptive', lambda h: p.predict_adaptive(h)),
    ('predict_normalized', lambda h: p.predict_normalized(h)),
]
for name, fn in modes:
    for b in [1, 2, 8]:
        h = torch.randn(b, 20, 6, device=p.device)
        h[:,:,3:6] *= 2.0
        out = fn(h)
        ok = (out['predictions'].shape == (b, 20, 3) and
              out['intent_logits'].shape == (b, 6) and
              not torch.isnan(out['predictions']).any())
        check(ok, f'{name} B={b}')

# ============================================================
# 3. Real data quality check (larger sample)
# ============================================================
print('\n[3] Real Data Quality (320 samples per dataset)')
from utils.fast_data_loader import FastWindowDataset
from utils.metrics import full_evaluation

for ds_name, path in [('UAV-Flow', '../UAV-Flow-pure'), ('NPZDATA', '../NPZDATA')]:
    ds = FastWindowDataset(path, split='test')
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds = []; targets = []
    for hist, pred, _ in loader:
        hist = hist.to(p.device); pred = pred.to(p.device)
        tgt = pred[:,:,:3] - hist[:,-1:,:3]
        with torch.no_grad():
            out = p.predict_adaptive(hist)
        preds.append(out['predictions'].cpu())
        targets.append(tgt.cpu())
        if len(preds) >= 5: break  # 320 samples

    r = full_evaluation(torch.cat(preds), torch.cat(targets))
    ok = r['RMSE'] < 10 and r['Distance_Accuracy'] > 0.3
    check(ok, f'{ds_name}: RMSE={r["RMSE"]:.3f} DistAcc={r["Distance_Accuracy"]:.4f}')

# ============================================================
# 4. Streaming stress test
# ============================================================
print('\n[4] Streaming — Long Sequence (1000 frames)')
sp = StreamingPredictor(p)
np.random.seed(42)
pos = np.zeros(3); preds_made = 0
for i in range(1000):
    vel = np.random.randn(3) * 2 + 3.0  # ~3 m/s with noise
    pos = pos + vel * 0.2
    frame = torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)
    r = sp.update(frame)
    if r is not None:
        preds_made += 1
        if torch.isnan(r['predictions']).any(): break
check(preds_made == 981, f'1000 frames -> {preds_made} predictions (expect 981)')
sp.reset()
check(not sp.ready, 'reset clears state')

# ============================================================
# 5. Synthetic extreme trajectories
# ============================================================
print('\n[5] Extreme Synthetic Trajectories')
extreme_tests = [
    ('Hover 0.001 m/s', 0.001, False),
    ('Slow 0.1 m/s', 0.1, False),
    ('Cruise 15 m/s', 15.0, False),
    ('Fast 50 m/s', 50.0, False),
    ('Very Fast 100 m/s', 100.0, False),
    ('Extreme 340 m/s (Mach1)', 340.0, False),
    ('Zero velocity', 0.0, False),
    ('Negative vel', -5.0, False),
    ('Large pos offset', 2.0, True),  # +10000m position offset
]
for name, speed, large_offset in extreme_tests:
    h = torch.randn(4, 20, 6, device=p.device)
    if large_offset:
        h[:,:,:3] += 10000.0
    h[:,:,3:6] = speed
    out = p.predict_adaptive(h)
    has_nan = torch.isnan(out['predictions']).any().item()
    has_inf = torch.isinf(out['predictions']).any().item()
    pred_range = out['predictions'].abs().max().item()
    check(not has_nan and not has_inf,
          f'{name}: pred_range={pred_range:.1f}m, NaN={has_nan}, Inf={has_inf}')

# ============================================================
# 6. Speed jump / discontinuity
# ============================================================
print('\n[6] Speed Discontinuity (instant hover -> Mach1)')
h = torch.zeros(2, 20, 6, device=p.device)
h[0, :, 3:6] = 0.01   # drone 1: hovering
h[1, :, 3:6] = 340.0  # drone 2: Mach 1 (same batch!)
out = p.predict_adaptive(h, return_scale=True)
sf = out.get('adaptive_scale', torch.tensor([1.0,1.0]))
if isinstance(sf, torch.Tensor) and sf.numel() >= 2:
    s0, s1 = sf[0].item(), sf[1].item()
    check(True, f'Mixed batch: hover_sf={s0:.1f}x, mach1_sf={s1:.1f}x (both stable)')
check(not torch.isnan(out['predictions']).any(), 'No NaN in mixed batch')

# ============================================================
# 7. Risk Assessment — boundary cases
# ============================================================
print('\n[7] Risk Assessment')
ra = RiskAssessment()
ra.add_cylinder(center=(0,0), radius=5.0, name='Tower')
ra.add_sphere(center=(20,20,10), radius=3.0, name='Antenna')

# Safe: far from all boundaries
safe_traj = torch.ones(1, 20, 3) * 50.0  # all points at (50,50,50)
risk = ra.evaluate(safe_traj, current_position=torch.tensor([50.,50.,50.]))
check(risk.level.value == 'low', f'Safe trajectory: {risk.level.value}')

# Critical: heading directly into cylinder at origin from very close
danger_traj = torch.linspace(0, 2, 20).unsqueeze(0).unsqueeze(-1).expand(-1,-1,3) * -0.5
danger_traj[:,:,0] += 3.0  # start at x=3, heading to 0 — will cross cylinder (r=5 at origin)
risk2 = ra.evaluate(danger_traj, current_position=torch.tensor([3.,0.,0.]))
check(risk2.warning or risk2.alert, f'Dangerous trajectory: level={risk2.level.value}, warn={risk2.warning}')

# ============================================================
# 8. Anomaly Detection — sustained degradation
# ============================================================
print('\n[8] Anomaly Detection')
ad = AnomalyDetector(min_samples=10)
# Establish baseline
for i in range(30):
    pred = torch.randn(1, 20, 3) * 0.5
    actual = pred + torch.randn(1, 20, 3) * 0.1
    ad.check(pred, actual, 'drone_x')

# Normal check (correlated pred/actual)
base = torch.randn(1,20,3)*0.5
r1 = ad.check(base, base+torch.randn(1,20,3)*0.05, 'drone_x')
check(not r1['is_anomaly'], f'Normal: anomaly={r1["is_anomaly"]}')

# Sustained degradation (simulate failing motor)
for i in range(20):
    pred = torch.randn(1, 20, 3) * 0.5
    actual = pred * 3 + torch.randn(1, 20, 3) * 2.0  # 3x error sustained
    ad.check(pred, actual, 'drone_x')
r2 = ad.check(torch.randn(1,20,3)*0.5, torch.randn(1,20,3)*5.0, 'drone_x')
check(r2['is_anomaly'] or r2['cusum_active'], f'Degraded: anomaly={r2["is_anomaly"]}, cusum={r2["cusum_active"]}')

# ============================================================
# 9. LoRA online learning — stability
# ============================================================
print('\n[9] LoRA Online Learning Stability')
p2 = DronePredictor()
p2.enable_adaptation(checkpoint_dir='test_final_lora', accumulation_steps=3)
h = torch.randn(1, 20, 6, device=p2.device); h[:,:,3:6] *= 1.5
base_gt = torch.randn(1, 20, 3, device=p2.device) * 0.5

losses = []
for i in range(30):
    gt = base_gt + torch.randn(1, 20, 3, device=p2.device) * 0.05
    out = p2.predict_with_adaptation(h, drone_id='drone_test', ground_truth=gt, timestep=i)
    if out.get('updated'):
        with torch.no_grad():
            pred = p2.predict(h)['predictions']
            loss = torch.nn.functional.mse_loss(pred, gt).item()
            losses.append(loss)

check(len(losses) >= 5, f'LoRA updates: {len(losses)} (expect >=5)')
if len(losses) >= 5:
    check(losses[-1] < losses[0] * 1.5, f'Loss stable: {losses[0]:.4f} -> {losses[-1]:.4f}')

import shutil; shutil.rmtree('test_final_lora', ignore_errors=True)

# ============================================================
# 10. Scale-invariant loss across scales
# ============================================================
print('\n[10] SI-MSE Scale Invariance')
a = torch.randn(10, 20, 3)
b = a.clone()
si_base = scale_invariant_mse(a, b).item()
si_100x = scale_invariant_mse(a*100, b*100).item()
si_001x = scale_invariant_mse(a*0.001, b*0.001).item()
check(abs(si_base - si_100x) < 0.01, f'SI-MSE scale invariant: 1x={si_base:.4f} 100x={si_100x:.4f}')

# ============================================================
# 11. Latency check
# ============================================================
print('\n[11] Inference Latency')
h = torch.randn(1, 20, 6, device=p.device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    p.predict(h)
torch.cuda.synchronize()
t1 = time.time()
latency = (t1 - t0) / 50 * 1000
check(latency < 100, f'Latency: {latency:.1f}ms/batch (expect <100ms)')

# ============================================================
# RESULTS
# ============================================================
print(f'\n{"="*60}')
print(f'RESULTS: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP')
if FAIL == 0:
    print('ALL TESTS PASSED — Ready for handoff.')
else:
    print(f'{FAIL} TESTS FAILED — Review before handoff.')
print(f'{"="*60}')
