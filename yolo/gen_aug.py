"""评测集光度增强生成器：21 张原图 × N 变体，几何不变（标注框直接复用）。

增强类型（全部不改变框坐标，GT 标签逐文件复制）：
  亮度 ±30% / 对比度 0.7-1.3 / 高斯噪声 / 高斯模糊 / JPEG 压缩伪影 / 组合
口径声明：增强集用于压力与鲁棒性评测，原图 21 张的人工口径指标单独报告，不混算。

运行：.conda/python.exe gen_aug.py [--n 100] [--seed 42]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_IMG = ROOT / "mydata" / "images" / "train"
SRC_LBL = ROOT / "mydata" / "labels" / "train"
DST_IMG = ROOT / "mydata_aug" / "images"
DST_LBL = ROOT / "mydata_aug" / "labels"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    # 亮度 + 对比度
    alpha = rng.uniform(0.7, 1.3)          # 对比度
    beta = rng.uniform(-0.15, 0.15) * 255  # 亮度
    out = out * alpha + beta
    # 高斯噪声
    if rng.random() < 0.6:
        out = out + rng.normal(0, rng.uniform(2, 10), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 高斯模糊
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    # JPEG 压缩伪影
    if rng.random() < 0.5:
        q = int(rng.integers(40, 90))
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="每张原图的变体数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    DST_IMG.mkdir(parents=True, exist_ok=True)
    DST_LBL.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in SRC_IMG.iterdir() if p.suffix.lower() in IMG_EXTS)
    total = 0
    for src in sources:
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径兼容
        lbl = SRC_LBL / f"{src.stem}.txt"
        lbl_text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        for i in range(args.n):
            aug = augment(img, rng)
            name = f"{src.stem}__aug{i:03d}"
            ok, enc = cv2.imencode(".jpg", aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            enc.tofile(str(DST_IMG / f"{name}.jpg"))  # 中文路径兼容
            (DST_LBL / f"{name}.txt").write_text(lbl_text, encoding="utf-8")
            total += 1
    print(f"生成 {total} 张增强图（{len(sources)} 原图 × {args.n}），标签逐文件复用 → mydata_aug/")


if __name__ == "__main__":
    main()
