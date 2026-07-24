#!/usr/bin/env python3
"""
Inference-latency benchmark for the two deployment entry points.

Measures the real public call paths on the current hardware:
  1. DronePredictor.predict          — general short-range inference (20-frame)
  2. DeployedLowPredictor.predict    — long-trajectory, inference-only (no GT)
  3. DeployedLowPredictor.predict    — with online learning (GT every call)

Reports mean / p50 / p95 / p99 per-call latency (ms) and throughput (frames/s).
Single-sample (batch=1) — the real per-drone streaming case. Run from repo root:
    python tests/benchmark_latency.py [--iters 300]
"""

import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import numpy as np
import torch


def _stats(times_ms):
    a = np.array(times_ms)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99)),
            'fps': float(1000.0 / a.mean())}


def _sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


def _bench(fn, iters, warmup, dev):
    for _ in range(warmup):
        fn()
    _sync(dev)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


def _row(name, s):
    print(f'  {name:<42}{s["mean"]:>8.2f}{s["p50"]:>8.2f}'
          f'{s["p95"]:>8.2f}{s["p99"]:>8.2f}{s["fps"]:>10.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=30)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_name = torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'
    print('=' * 90)
    print(f'Inference latency benchmark  |  device={dev} ({dev_name})  '
          f'|  iters={args.iters} warmup={args.warmup}')
    print('=' * 90)

    from predictor import DronePredictor
    from deploy import DeployedLowPredictor

    torch.manual_seed(0)
    hist20 = torch.randn(1, 20, 6)
    hist40 = torch.randn(1, 40, 6)
    gt = torch.randn(1, 20, 3) * 0.1

    results = {}

    # 1) DronePredictor
    dp = DronePredictor()
    results['DronePredictor.predict (20f)'] = _bench(
        lambda: dp.predict(hist20), args.iters, args.warmup, dev)

    # 2) DeployedLowPredictor — inference only (no learning)
    dl = DeployedLowPredictor(use_global=True)
    results['DeployedLowPredictor.predict (40f, infer)'] = _bench(
        lambda: dl.predict(hist40, frames_seen=10), args.iters, args.warmup, dev)

    # 3) DeployedLowPredictor — with online learning (GT every call)
    #    amortized: most calls just buffer; every accumulation_steps triggers an update.
    step = {'t': 0}
    def learn_call():
        step['t'] += 1
        dl.predict(hist40, drone_id='bench_drone', ground_truth=gt,
                   frames_seen=200, timestep=step['t'])
    results['DeployedLowPredictor.predict (40f, online)'] = _bench(
        learn_call, args.iters, args.warmup, dev)

    print(f'\n  {"path":<42}{"mean":>8}{"p50":>8}{"p95":>8}{"p99":>8}{"fps":>10}')
    print(f'  {"":<42}{"(ms)":>8}{"(ms)":>8}{"(ms)":>8}{"(ms)":>8}{"":>10}')
    print(f'  {"-"*84}')
    for name, s in results.items():
        _row(name, s)
    print('=' * 90)

    # real-time headroom vs the two stream rates
    infer = results['DeployedLowPredictor.predict (40f, infer)']['mean']
    print(f'\n  Real-time check (single drone, batch=1):')
    print(f'    LOW  5 Hz  budget 200 ms/frame  ->  infer {infer:.1f} ms  '
          f'({200/infer:.0f}x headroom)')
    print(f'    HIGH 1 Hz  budget 1000 ms/frame ->  infer {infer:.1f} ms  '
          f'({1000/infer:.0f}x headroom)')

    import json
    out = Path(__file__).resolve().parents[1] / 'pic-results' / 'latency_benchmark.json'
    json.dump({'device': dev, 'device_name': dev_name, 'iters': args.iters,
               'results_ms': results}, open(out, 'w'), indent=2)
    print(f'\n  Saved: pic-results/latency_benchmark.json')


if __name__ == '__main__':
    main()
