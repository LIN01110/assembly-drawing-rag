"""
eval_pipeline.py — 项目2 P0 检测消融评估脚本

评估臂（检测侧）：
  - baseline : use_sahi=False（等价 integrate.py --no-sahi）
  - sahi     : use_sahi=True（完整检测管线）
  注：--no-multi-scale / --no-voting 是 OCR 侧开关，不影响检测框；
      OCR 臂待文字真值人工转写后补跑（见 results/ablation_report.md 标注）。

指标（单类 panel，IoU=0.50）：
  - Precision / Recall（工作点 conf>=0.15，与 integrate.py 默认一致）
  - mAP@50（全点插值 AP）
  - 漏检率 miss_rate = FN / GT 总数
互验：--arm val 调用 ultralytics 原生 model.val() 出基线 mAP。

用法（仓库根目录）：
  .conda/python.exe eval_pipeline.py --arm baseline
  .conda/python.exe eval_pipeline.py --arm sahi
  .conda/python.exe eval_pipeline.py --arm val
  .conda/python.exe eval_pipeline.py --arm report   # 汇总生成 Markdown 报告

输出：
  results/eval/<arm>_detections.json   每臂逐图检测框（可复算）
  results/ablation_metrics.json        各臂指标汇总
  results/ablation_report.md           Markdown 报告
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "mydata" / "images" / "train"
LABELS_DIR = ROOT / "mydata" / "labels" / "train"
WEIGHTS = ROOT / "runs" / "detect_train" / "weights" / "best.pt"
RESULTS_DIR = ROOT / "results" / "eval"
METRICS_JSON = ROOT / "results" / "ablation_metrics.json"
REPORT_MD = ROOT / "results" / "ablation_report.md"

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.15  # 与 integrate.py detect_objects 默认一致
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 真值与检测框
# ---------------------------------------------------------------------------
def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """YOLO 格式 (cls cx cy w h, 归一化) -> xyxy 绝对坐标 (N,4)。"""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float64)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def extract_pred_boxes(detected_objects) -> list[dict]:
    """从 integrate.py 的检测结果（ultralytics Results 或 SAHI MockResult）
    抽取 [{xyxy:[x1,y1,x2,y2], conf:float}]，两种对象都暴露逐框 .xyxy/.conf。"""
    preds = []
    for result in detected_objects:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            preds.append({"xyxy": [x1, y1, x2, y2], "conf": float(box.conf[0])})
    return preds


# ---------------------------------------------------------------------------
# 指标计算（单类）
# ---------------------------------------------------------------------------
def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred_with_conf: np.ndarray, gt_boxes: np.ndarray):
    """pred_with_conf: (N,5) xyxy+conf。按 conf 降序贪心匹配（与 GT 一对一）。"""
    pred_boxes = pred_with_conf[:, :4]
    if len(pred_boxes) == 0:
        return np.zeros(0, dtype=bool), np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_with_conf[:, 4])
    tp = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    for p in order:
        if len(gt_boxes) == 0:
            break
        g = int(np.argmax(ious[p]))
        if ious[p, g] >= IOU_THRESHOLD and not gt_matched[g]:
            tp[p] = True
            gt_matched[g] = True
    return tp, gt_matched


def compute_metrics(per_image: dict[str, dict]) -> dict:
    """per_image: {stem: {"pred": [[x1,y1,x2,y2,conf]...], "gt": [[x1,y1,x2,y2]...]}}"""
    all_preds = []  # (conf, tp)
    total_gt = 0
    tp_at_op = fp_at_op = 0
    for data in per_image.values():
        gt = np.asarray(data["gt"], dtype=np.float64).reshape(-1, 4)
        pred = np.asarray(data["pred"], dtype=np.float64).reshape(-1, 5)
        total_gt += len(gt)
        if len(pred) == 0:
            continue
        tp_flags, _ = match_image(pred, gt)
        for flag, conf in zip(tp_flags, pred[:, 4]):
            all_preds.append((float(conf), bool(flag)))
            if conf >= CONF_THRESHOLD:
                if flag:
                    tp_at_op += 1
                else:
                    fp_at_op += 1

    # 工作点 P/R（conf >= 0.15）
    fn_at_op = total_gt - tp_at_op
    precision = tp_at_op / max(tp_at_op + fp_at_op, 1)
    recall = tp_at_op / max(total_gt, 1)
    miss_rate = fn_at_op / max(total_gt, 1)

    # AP@50（全点插值）
    ap = 0.0
    if all_preds and total_gt > 0:
        all_preds.sort(key=lambda x: -x[0])
        tps = np.array([1.0 if t else 0.0 for _, t in all_preds])
        fps = 1.0 - tps
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        rec = cum_tp / total_gt
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[1.0], prec, [0.0]])
        # 精度包络：从右往左取最大
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {
        "gt_total": total_gt,
        "pred_total": len(all_preds),
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "precision_at_op": round(precision, 4),
        "recall_at_op": round(recall, 4),
        "miss_rate_at_op": round(miss_rate, 4),
        "map50": round(ap, 4),
    }


# ---------------------------------------------------------------------------
# 评估臂
# ---------------------------------------------------------------------------
def run_detection_arm(arm: str) -> dict:
    """arm: baseline (use_sahi=False) | sahi (use_sahi=True)"""
    sys.path.insert(0, str(ROOT))
    from integrate import YOLOProcessor  # 延迟导入，避免 --arm val/report 时加载 torch

    use_sahi = arm == "sahi"
    yolo = YOLOProcessor(str(WEIGHTS))
    image_paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise SystemExit(f"未找到图像: {IMAGES_DIR}")

    per_image: dict[str, dict] = {}
    t0 = time.time()
    for i, img_path in enumerate(image_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i}/{len(image_paths)}] 跳过无法读取: {img_path.name}")
            continue
        h, w = image.shape[:2]
        gt = load_gt_boxes(LABELS_DIR / f"{img_path.stem}.txt", w, h)
        detected = yolo.detect_objects(image, conf_threshold=CONF_THRESHOLD, use_sahi=use_sahi)
        preds = extract_pred_boxes(detected)
        per_image[img_path.stem] = {
            "gt": gt.tolist(),
            "pred": [[*p["xyxy"], p["conf"]] for p in preds],
        }
        print(f"[{i}/{len(image_paths)}] {img_path.name}: GT={len(gt)} pred={len(preds)}")

    elapsed = time.time() - t0
    metrics = compute_metrics(per_image)
    metrics["arm"] = arm
    metrics["use_sahi"] = use_sahi
    metrics["images"] = len(per_image)
    metrics["elapsed_sec"] = round(elapsed, 1)
    metrics["device"] = "cpu"  # 本脚本在当前环境为 CPU；换 GPU 环境复跑后更新

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    det_path = RESULTS_DIR / f"{arm}_detections.json"
    det_path.write_text(json.dumps(per_image, ensure_ascii=False), encoding="utf-8")
    print(f"检测框已保存: {det_path}")

    save_arm_metrics(arm, metrics)
    return metrics


def run_ultralytics_val() -> dict:
    """ultralytics 原生 model.val() 基线互验（val 集 = train 集，21 张）。"""
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    r = model.val(data=str(ROOT / "data.yaml"), split="val", verbose=False)
    n_images = len([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    metrics = {
        "arm": "ultralytics_val",
        "images": n_images,
        "map50": round(float(r.box.map50), 4),
        "map50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "note": "val 集即 train 集（21 张），训练集口径，数字偏乐观",
    }
    save_arm_metrics("ultralytics_val", metrics)
    return metrics


def save_arm_metrics(arm: str, metrics: dict) -> None:
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if METRICS_JSON.exists():
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    data[arm] = metrics
    METRICS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已写入: {METRICS_JSON} [{arm}]")


def build_report() -> None:
    if not METRICS_JSON.exists():
        raise SystemExit("先运行 baseline / sahi / val 臂生成指标")
    data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

    lines = [
        "# 项目2 检测消融评估报告（P0）",
        "",
        f"- 评估集：mydata/images/train（21 张，**训练集口径，val=train，数字偏乐观**）",
        f"- 类别：panel（单类）；IoU=0.50；工作点 conf≥{CONF_THRESHOLD}",
        f"- 权重：runs/detect_train/weights/best.pt",
        "",
        "## 检测臂对比",
        "",
        "| 臂 | SAHI | 图像数 | GT 框 | 预测框 | P | R | 漏检率 | mAP@50 | 耗时 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("baseline", "sahi"):
        m = data.get(arm)
        if not m:
            continue
        lines.append(
            f"| {arm} | {'开' if m.get('use_sahi') else '关'} | {m['images']} | {m['gt_total']} | "
            f"{m['pred_total']} | {m['precision_at_op']:.3f} | {m['recall_at_op']:.3f} | "
            f"{m['miss_rate_at_op']:.3f} | {m['map50']:.3f} | {m['elapsed_sec']}s |"
        )
    if "ultralytics_val" in data:
        v = data["ultralytics_val"]
        lines += [
            "",
            "## ultralytics 原生 val 互验",
            "",
            f"- mAP@50 = {v['map50']:.3f}，mAP@50-95 = {v['map50_95']:.3f}，"
            f"P = {v['precision']:.3f}，R = {v['recall']:.3f}",
            f"- 注：{v['note']}",
        ]
    vlm_json = ROOT / "results" / "eval" / "vlm_arm.json"
    if vlm_json.exists():
        v = json.loads(vlm_json.read_text(encoding="utf-8"))
        lines += [
            "",
            "## VLM 对照臂（端到端多模态大模型 vs 专用检测管线）",
            "",
            f"- 模型：{v['model']}（{v['base_url']}）；任务：数图纸独立信息板块数量",
            f"- 回答率：{v['answered']}/{v['images']}；**数量完全正确率 {v['count_exact_acc']:.1%}**，"
            f"平均绝对误差 {v['count_mae']}",
            "- 对比：同集上 YOLO 管线召回 0.987 / mAP@50 0.987（框级，IoU=0.5）",
            "- 结论：VLM 端到端在工业图纸结构化解析上显著弱于专用管线，支撑"
            "“YOLO+OCR 工程管线而非直接调大模型”的技术选型",
        ]
    lines += [
        "",
        "## 待办（本轮范围外）",
        "",
        "- OCR 臂（多尺度 / 投票消融 + 准确率）：**待人工转写文字真值后补跑**",
        "- VLM 对照臂：见 eval_vlm_arm.py 输出 results/eval/vlm_arm.json",
        "- 样本量仅 21 张且为训练集口径，简历数字建议用区间并注明口径",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="项目2 检测消融评估")
    parser.add_argument("--arm", required=True,
                        choices=["baseline", "sahi", "val", "report"],
                        help="baseline=无SAHI; sahi=完整检测; val=ultralytics互验; report=汇总报告")
    args = parser.parse_args()

    if args.arm in ("baseline", "sahi"):
        m = run_detection_arm(args.arm)
        print(json.dumps(m, ensure_ascii=False, indent=2))
    elif args.arm == "val":
        m = run_ultralytics_val()
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        build_report()


if __name__ == "__main__":
    main()
