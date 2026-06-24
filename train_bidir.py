#!/usr/bin/env python3
"""Train bidirectional enhancer on UAV-Flow pure low-speed data."""
import sys, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from bidirectional import BidirectionalPredictor
from utils.fast_data_loader import FastWindowDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../UAV-Flow-pure')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save', default='checkpoints/bidir_enhancer.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Data: {args.data_dir}')

    # Load predictor + wrap
    print('Loading DronePredictor...')
    p = DronePredictor(device=device)
    bp = BidirectionalPredictor(p)

    # Load UAV-Flow train data
    print(f'Loading UAV-Flow train data from {args.data_dir}...')
    ds = FastWindowDataset(args.data_dir, split='train')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0, drop_last=True)
    print(f'Train samples: {len(ds):,}, batches: {len(loader)}')

    # Train using the BidirectionalPredictor's built-in trainer
    # (uses feature injection hook for proper gradient flow, FP32, no AMP)
    bp.train_enhancer(loader, epochs=args.epochs, lr=args.lr, amp=False)

    print(f'\nDone. Weights: checkpoints/bidir_enhancer.pth')

if __name__ == '__main__':
    main()
