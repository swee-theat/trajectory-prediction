"""快速下载脚本：使用 Python requests 流式下载 + 大缓冲区。"""

import requests
import sys
from pathlib import Path

URLS = {
    "train": "https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_train_v1.1.tar.gz",
    "val": "https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_val_v1.1.tar.gz",
    "test": "https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_test_v1.1.tar.gz",
}

def download(name, url, output_dir):
    output_path = Path(output_dir) / f"forecasting_{name}_v1.1.tar.gz"

    # 检查已下载大小（断点续传）
    existing_size = output_path.stat().st_size if output_path.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"  续传: 已有 {existing_size/1024/1024:.1f} MB")

    resp = requests.get(url, stream=True, headers=headers, timeout=30)

    total = existing_size + int(resp.headers.get("content-length", 0))
    downloaded = existing_size

    mode = "ab" if existing_size > 0 else "wb"
    with open(output_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):  # 1MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total > 0 else 0
            mb = downloaded / 1024 / 1024
            sys.stdout.write(f"\r  {name}: {mb:.0f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    print(f"\n  {name}: 完成! ({downloaded/1024/1024:.0f} MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["train","val","test","all"], default="all")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.name == "all":
        for name in ["train", "val", "test"]:
            print(f"\n下载 {name}...")
            download(name, URLS[name], args.output_dir)
    else:
        download(args.name, URLS[args.name], args.output_dir)
