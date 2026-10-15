#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记录当前 train/val/test.npz 的锚点信息（样本数 + seq_id 集合哈希 + 类别分布）。

重跑预处理后用它再次运行，比对输出是否一致——这是"逐样本不变"锚点验证的第一步
（更严格的 overall minADE 逐位一致由重评估脚本验证）。

用法:
    python scripts/record_data_anchor.py
"""

import sys
import os
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    for split in ["train", "val", "test"]:
        p = f"data/processed/{split}.npz"
        d = np.load(p, allow_pickle=False)
        seqs = d["seq_ids"]
        n = len(seqs)

        # seq_id 集合哈希（排序后逐条拼接，与顺序无关）
        h = hashlib.sha256()
        for s in sorted(seqs.tolist()):
            h.update(s.encode("utf-8"))
            h.update(b"\x00")

        cats = d["categories"]
        cat_counts = {int(c): int((cats == c).sum()) for c in np.unique(cats)}
        print(f"{split}: n={n:,}  seq_id_hash={h.hexdigest()[:16]}  cats={cat_counts}")


if __name__ == "__main__":
    main()
