"""预处理 Argoverse v1.1 真实数据：CSV → 轨迹提取 → 9维特征 → .npz。

用法: python scripts/preprocess_real_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import argparse

from data.preprocessing import process_csv_file, city_to_id
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/game/da_chuang/1")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--val-from-train", action="store_true",
                        help="从训练集抽取验证集（而非用官方val）")
    args = parser.parse_args()

    set_seed(42)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 CSV 文件
    train_csvs = sorted((data_root / "train" / "data").glob("*.csv"))
    val_csvs = sorted((data_root / "val" / "data").glob("*.csv"))

    print(f"训练 CSV: {len(train_csvs):,}")
    print(f"验证 CSV: {len(val_csvs):,}")

    # 处理训练集 CSV
    print("\n处理训练集...")
    all_samples = []
    batch_size = 5000
    batch = []

    for i, csv_path in enumerate(train_csvs):
        samples = process_csv_file(str(csv_path), args.obs_len, args.pred_len, split="train")
        batch.extend(samples)

        if len(batch) >= batch_size:
            all_samples.extend(batch)
            batch = []
            print(f"  训练进度: {i+1}/{len(train_csvs)} ({len(all_samples):,} 样本)")

        if (i + 1) % 5000 == 0:
            sys.stdout.flush()

    all_samples.extend(batch)  # 剩余批次
    train_total = len(all_samples)
    print(f"训练集样本: {train_total:,}")

    # 处理验证集 CSV（如果不用的话也提取，然后用整个数据做7:1:2划分）
    print("\n处理验证集...")
    val_samples = []
    for i, csv_path in enumerate(val_csvs):
        samples = process_csv_file(str(csv_path), args.obs_len, args.pred_len, split="val")
        val_samples.extend(samples)
        if (i + 1) % 5000 == 0:
            print(f"  验证进度: {i+1}/{len(val_csvs)} ({len(val_samples):,} 样本)")

    val_total = len(val_samples)
    print(f"验证集样本: {val_total:,}")
    total = train_total + val_total
    print(f"\n总样本: {total:,}")

    # 合并 + 打乱 + 7:1:2 划分
    all_samples.extend(val_samples)

    # 提取数组
    n = len(all_samples)
    histories = np.array([s["history"] for s in all_samples], dtype=np.float32)
    futures = np.array([s["future"] for s in all_samples], dtype=np.float32)
    categories = np.array([s["category"] for s in all_samples], dtype=np.int32)
    seq_ids = np.array([s["seq_id"] for s in all_samples])
    cities = np.array([city_to_id(s.get("city", "UNK")) for s in all_samples], dtype=np.int32)
    neighbor_histories = np.array([s["neighbor_histories"] for s in all_samples], dtype=np.float32)
    neighbor_futures = np.array([s["neighbor_futures"] for s in all_samples], dtype=np.float32)
    neighbor_mask = np.array([s["neighbor_mask"] for s in all_samples], dtype=np.float32)

    # 确定性打乱
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)

    histories = histories[indices]
    futures = futures[indices]
    categories = categories[indices]
    seq_ids = seq_ids[indices]
    cities = cities[indices]
    neighbor_histories = neighbor_histories[indices]
    neighbor_futures = neighbor_futures[indices]
    neighbor_mask = neighbor_mask[indices]

    # 7:1:2 划分
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }

    for split_name, s in splits.items():
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            histories=histories[s],
            futures=futures[s],
            categories=categories[s],
            seq_ids=seq_ids[s],
            cities=cities[s],
            neighbor_histories=neighbor_histories[s],
            neighbor_futures=neighbor_futures[s],
            neighbor_mask=neighbor_mask[s],
        )
        count = s.stop - s.start
        print(f"  {split_name}: {count:,} 样本 → {output_dir / f'{split_name}.npz'}")

    # 场景统计
    cat_names = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}
    print(f"\n场景分布:")
    for cid, cname in cat_names.items():
        cnt = int(np.sum(categories == cid))
        print(f"  {cname}: {cnt:,} ({cnt/n*100:.1f}%)")

    # 交互邻居统计（第三阶段）
    has_nbr1 = int(np.sum(neighbor_mask[:, 0] == 1))
    has_nbr2 = int(np.sum(neighbor_mask[:, 1] == 1))
    print(f"\n交互邻居统计:")
    print(f"  有最近交互车(neighbor_mask[0]=1): {has_nbr1:,} ({has_nbr1/n*100:.1f}%)")
    print(f"  有第二交互车(neighbor_mask[1]=1): {has_nbr2:,} ({has_nbr2/n*100:.1f}%)")
    print(f"  无 OTHERS 邻居: {n - has_nbr1:,} ({(n - has_nbr1)/n*100:.1f}%)")

    print(f"\n预处理完成！总样本: {n:,}")


if __name__ == "__main__":
    main()
