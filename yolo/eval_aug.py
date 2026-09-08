"""增强集检测评测：复用 eval_pipeline 的指标计算，支持分块跑（CPU 环境）。

用法：
  .conda/python.exe eval_aug.py --arm baseline --start 0 --n 500   # 分块
  .conda/python.exe eval_aug.py --arm baseline --merge             # 合并出指标
输出：results/eval/aug_<arm>_part<start>.json → 合并 results/aug_<arm>_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_pipeline as ep

ep.IMAGES_DIR = ROOT / "mydata_aug" / "images"
ep.LABELS_DIR = ROOT / "mydata_aug" / "labels"

PART_DIR = ROOT / "results" / "eval"


def run_chunk(arm: str, start: int, n: int, stride: int = 1) -> None:
    from integrate import YOLOProcessor
    use_sahi = arm == "sahi"
    yolo = YOLOProcessor(str(ep.WEIGHTS))
    paths = sorted(p for p in ep.IMAGES_DIR.iterdir() if p.suffix.lower() in ep.IMG_EXTS)
    if stride > 1:
        paths = paths[::stride]
    chunk = paths[start:start + n]
    tag = f"_s{stride}" if stride > 1 else ""
    per_image = {}
    t0 = time.time()
    for i, img_path in enumerate(chunk, 1):
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径兼容
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = ep.load_gt_boxes(ep.LABELS_DIR / f"{img_path.stem}.txt", w, h)
        detected = yolo.detect_objects(image, conf_threshold=ep.CONF_THRESHOLD, use_sahi=use_sahi)
        preds = ep.extract_pred_boxes(detected)
        per_image[img_path.stem] = {"gt": gt.tolist(),
                                    "pred": [[*p["xyxy"], p["conf"]] for p in preds]}
        if i % 50 == 0:
            print(f"  {i}/{len(chunk)}  {time.time() - t0:.0f}s")
    PART_DIR.mkdir(parents=True, exist_ok=True)
    out = PART_DIR / f"aug_{arm}{tag}_part{start}.json"
    out.write_text(json.dumps(per_image, ensure_ascii=False), encoding="utf-8")
    print(f"chunk {start}-{start + len(chunk)} 完成 {time.time() - t0:.0f}s → {out.name}")


def merge(arm: str) -> None:
    per_image = {}
    for f in sorted(PART_DIR.glob(f"aug_{arm}_part*.json")):
        per_image.update(json.loads(f.read_text(encoding="utf-8")))
    metrics = ep.compute_metrics(per_image)
    metrics["arm"] = arm
    metrics["images"] = len(per_image)
    metrics["note"] = "光度增强集（21 原图 × 100 变体，GT 框复用），CPU 推理"
    out = ROOT / "results" / f"aug_{arm}_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "sahi"], required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="每隔 K 张取 1 张（SAHI 慢速臂抽稀用）")
    args = ap.parse_args()
    if args.merge:
        merge(args.arm)
    else:
        run_chunk(args.arm, args.start, args.n, args.stride)
