#!/usr/bin/env python3
"""Fast DataLoader v2: preloads all chunks into memory to remove disk I/O bottleneck."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Literal


class FastWindowDataset(Dataset):
    """Preloaded dataset: loads all chunks into RAM at init, __getitem__ indexes memory only."""

    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test"] = "train",
        label_remap: dict = None,
    ):
        """
        Args:
            data_root: SimCruise directory path
            split: train/val/test
            label_remap: label remap dict, e.g. {4:3} maps DESCEND->3 (ASCEND absent)
        """
        self.data_root = Path(data_root)
        self.split = split

        # Collect chunk files
        chunk_files = sorted(self.data_root.rglob(f"windows_{split}_chunk*.npz"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No windows_{split}_chunk*.npz found in {self.data_root}"
            )

        # First pass: count samples, build index ranges
        chunk_offsets = [0]
        n_per_chunk = []
        valid_files = []

        for f in chunk_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    n = len(data["hist"])
                n_per_chunk.append(n)
                chunk_offsets.append(chunk_offsets[-1] + n)
                valid_files.append(f)
            except Exception as e:
                print(f"[FastWindowDataset] Skipping corrupt {f.name}: {e}")

        if not valid_files:
            raise FileNotFoundError(
                f"No valid chunks for {split} in {self.data_root}"
            )

        self.chunk_offsets = chunk_offsets
        self.n_per_chunk = n_per_chunk
        self.n_samples = chunk_offsets[-1]
        self.label_remap = label_remap or {}

        # Second pass: preload all chunks into RAM
        print(
            f"[FastWindowDataset] Loading {len(valid_files)} chunks "
            f"({self.n_samples:,} {split} samples) into RAM..."
        )
        self._hist = []
        self._pred = []
        self._intent = []
        for f in valid_files:
            data = np.load(f)
            self._hist.append(data["hist"])
            self._pred.append(data["pred"])
            self._intent.append(data["intent"])
            del data
        print(f"[FastWindowDataset] Done. ~{self.n_samples * 724 / 1e9:.1f} GB loaded.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Index directly from preloaded memory arrays, zero I/O."""
        for ci, (start, n) in enumerate(
            zip(self.chunk_offsets[:-1], self.n_per_chunk)
        ):
            if idx < start + n:
                local_idx = idx - start
                break
        else:
            local_idx = 0

        hist = torch.from_numpy(self._hist[ci][local_idx])
        pred = torch.from_numpy(self._pred[ci][local_idx])
        intent = int(self._intent[ci][local_idx])
        intent = self.label_remap.get(intent, intent)
        intent = torch.tensor(intent, dtype=torch.long)
        return hist, pred, intent


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle: bool = None,
    label_remap: dict = None,
):
    """Factory function. Needs if __name__ == '__main__' guard on Windows."""
    import platform
    if shuffle is None:
        shuffle = split == "train"

    # Windows spawn multiprocessing needs main-module guard; fall back to single process if unprotected
    effective_workers = num_workers
    if platform.system() == 'Windows' and num_workers > 0:
        try:
            import multiprocessing
            multiprocessing.get_context('spawn')
        except Exception:
            effective_workers = 0

    dataset = FastWindowDataset(data_root=data_root, split=split, label_remap=label_remap)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=effective_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=2 if effective_workers > 0 else None,
        drop_last=split == "train",
    )
    if effective_workers > 0:
        print(f"  DataLoader: {effective_workers} workers, prefetch=2")
    return loader


if __name__ == "__main__":
    import time
    from pathlib import Path

    data_root = str(Path(__file__).resolve().parent.parent.parent / "SimCruise")

    for split in ["train", "val"]:
        t0 = time.time()
        ds = FastWindowDataset(data_root, split=split)
        print(f"  init={time.time() - t0:.1f}s")
        # Test read speed
        t0 = time.perf_counter()
        for i in range(1000):
            _ = ds[i % len(ds)]
        elapsed = time.perf_counter() - t0
        print(f"  1000 reads: {elapsed:.3f}s  ({elapsed/1000*1000:.1f}ms/sample)")

    # Test DataLoader
    ds = FastWindowDataset(data_root, split="train")
    loader = get_dataloader(data_root, split="train", batch_size=32)
    t0 = time.perf_counter()
    for idx, batch in enumerate(loader):
        if idx >= 5:
            break
    elapsed = time.perf_counter() - t0
    print(f"\n5 batches: {elapsed:.1f}s  ({elapsed/5:.2f}s per batch)")
