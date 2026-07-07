#!/usr/bin/env python3
"""
快速 DataLoader v2：预加载所有 chunk 到内存，消除磁盘 I/O 瓶颈。

npz 文件结构：
    hist    : (N, hist_len, 6)  float32  [pos+vel]
    pred    : (N, pred_len, 3)  float32  [pos delta]
    intent  : (N,)             int32    意图标签
    maneuver: (N,)             int32    机动等级
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Literal


class FastWindowDataset(Dataset):
    """
    预加载型数据集：初始化时将所有 chunk 加载到内存（约 1.2 GB），
    __getitem__ 直接从内存索引，不再触碰磁盘。
    """

    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test"] = "train",
        label_remap: dict = None,
    ):
        """
        Args:
            data_root: SimCruise 目录路径
            split: train/val/test
            label_remap: 标签重映射 dict, 如 {4:3} 将 DESCEND→3 (ASCEND空缺)
        """
        self.data_root = Path(data_root)
        self.split = split

        # 收集 chunk 文件
        chunk_files = sorted(self.data_root.rglob(f"windows_{split}_chunk*.npz"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No windows_{split}_chunk*.npz found in {self.data_root}"
            )

        # 第一遍：统计样本数，建立索引范围
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

        # 第二遍：预加载所有 chunk 到内存
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
        """从预加载内存数组直接索引，零 I/O"""
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
    """工厂函数。Windows 下需要 if __name__ == '__main__' 保护。"""
    import platform
    if shuffle is None:
        shuffle = split == "train"

    # Windows 下 multiprocessing spawn 需要主模块保护
    # 如果调用方未正确保护, 自动降级为单进程
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
        # 测试读取速度
        t0 = time.perf_counter()
        for i in range(1000):
            _ = ds[i % len(ds)]
        elapsed = time.perf_counter() - t0
        print(f"  1000 reads: {elapsed:.3f}s  ({elapsed/1000*1000:.1f}ms/sample)")

    # 测试 DataLoader
    ds = FastWindowDataset(data_root, split="train")
    loader = get_dataloader(data_root, split="train", batch_size=32)
    t0 = time.perf_counter()
    for idx, batch in enumerate(loader):
        if idx >= 5:
            break
    elapsed = time.perf_counter() - t0
    print(f"\n5 batches: {elapsed:.1f}s  ({elapsed/5:.2f}s per batch)")
