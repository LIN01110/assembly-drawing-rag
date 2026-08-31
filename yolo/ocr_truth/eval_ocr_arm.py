"""
eval_ocr_arm.py — 项目2 OCR 臂评估（待真值后运行）

评估臂（OCR 侧开关组合，对应 integrate.py 的 --no-multi-scale / --no-voting）：
  full    : 多尺度开 + 投票开（完整管线）
  no_ms   : 关多尺度
  no_vt   : 关投票
  base    : 多尺度关 + 投票关（最简管线）

指标：
  - 字段准确率：标题栏 12 字段逐一 exact match（空白规整后），
    分母 = 真值中非空字段数（图纸里有的字段）；另报"误提取率"（图纸没有却提了）。
  - 技术要求 CER：pipeline 技术要求文本 vs 真值 tech_requirements 的字符错误率
    （Levenshtein 距离 / 真值字符数；越小越好）。

用法（仓库根目录，.conda 环境）：
  # 1. 跑臂（每个臂约 3-6 分钟 CPU）
  .conda/python.exe ocr_truth/eval_ocr_arm.py --arm full
  .conda/python.exe ocr_truth/eval_ocr_arm.py --arm no_ms
  .conda/python.exe ocr_truth/eval_ocr_arm.py --arm no_vt
  .conda/python.exe ocr_truth/eval_ocr_arm.py --arm base
  # 2. 全部跑完后出指标（需要先有人工校正后的 ocr_truth/ground_truth.json）
  .conda/python.exe ocr_truth/eval_ocr_arm.py --metrics

输出：
  results/ocr_arms/<arm>/*_result.json   各臂管线输出
  results/ocr_metrics.json               指标汇总
  results/ocr_report.md                  Markdown 报告
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_JSON = ROOT / "ocr_truth" / "ground_truth.json"
ARMS_DIR = ROOT / "results" / "ocr_arms"
METRICS_JSON = ROOT / "results" / "ocr_metrics.json"
REPORT_MD = ROOT / "results" / "ocr_report.md"

ARM_FLAGS = {
    "full": [],
    "no_ms": ["--no-multi-scale"],
    "no_vt": ["--no-voting"],
    "base": ["--no-multi-scale", "--no-voting"],
}

FIELDS = ["物品名称", "材料", "图号", "数量", "张数", "重量",
          "比例", "设计", "审核", "工艺", "标准化", "批准"]

FIELD_KEYS = {  # 与 integrate.py 的 field_mapping 一致
    "物品名称": ["物品名称", "物品"],
    "材料": ["材料", "材质", "牌号"],
    "图号": ["图号", "图纸编号"],
    "数量": ["数量", "件数"],
    "张数": ["张数"],
    "重量": ["重量", "质量"],
    "比例": ["比例", "SCALE"],
    "设计": ["设计"],
    "审核": ["审核"],
    "工艺": ["工艺"],
    "标准化": ["标准化"],
    "批准": ["批准"],
}


# ---------------------------------------------------------------------------
# 跑臂
# ---------------------------------------------------------------------------
def run_arm(arm: str) -> None:
    out_dir = ARMS_DIR / arm
    if out_dir.exists() and len(list(out_dir.glob("*_result.json"))) >= 21:
        print(f"[{arm}] 已存在完整结果（21/21），跳过（如需重跑请删除 {out_dir}）")
        return
    cmd = [
        str(ROOT / ".conda" / "python.exe"),
        str(ROOT / "integrate.py"),
        "--mode", "batch",
        "--input", str(ROOT / "mydata" / "images" / "train"),
        "--model", str(ROOT / "runs" / "detect_train" / "weights" / "best.pt"),
        "--output", str(out_dir),
        *ARM_FLAGS[arm],
    ]
    print(f"[{arm}] 运行: {' '.join(ARM_FLAGS[arm]) or 'full'} → {out_dir}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    n = len(list(out_dir.glob("*_result.json")))
    print(f"[{arm}] 完成 {n}/21 张，耗时 {time.time() - t0:.0f}s，返回码 {proc.returncode}")
    if n < 21:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    return "".join(str(s).split())  # 去所有空白


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_pipeline_fields(result_path: Path) -> tuple[dict, str]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    si = data.get("structured_info", {})
    table_info = si.get("表格信息", {})
    fields = {}
    for f in FIELDS:
        for k in FIELD_KEYS[f]:
            if k in table_info and str(table_info[k]).strip():
                fields[f] = str(table_info[k]).strip()
                break
    tech = "\n".join(si.get("技术要求", []))
    return fields, tech


def compute_arm_metrics(arm: str, gt: dict) -> dict:
    arm_dir = ARMS_DIR / arm
    per_field = {f: {"hit": 0, "total": 0} for f in FIELDS}
    spurious = 0       # 图纸没有却提取出来的字段值
    spurious_total = 0
    cer_sum = 0.0
    cer_n = 0
    images_done = 0

    for stem, truth in gt.items():
        result_path = arm_dir / f"{stem}_result.json"
        if not result_path.exists():
            continue
        images_done += 1
        pred_fields, pred_tech = load_pipeline_fields(result_path)

        for f in FIELDS:
            gt_val = norm(truth["fields"].get(f, ""))
            pred_val = norm(pred_fields.get(f, ""))
            if gt_val:
                per_field[f]["total"] += 1
                if pred_val == gt_val:
                    per_field[f]["hit"] += 1
            else:
                spurious_total += 1
                if pred_val:
                    spurious += 1

        gt_tech = norm(truth.get("tech_requirements", ""))
        if gt_tech:
            dist = levenshtein(norm(pred_tech), gt_tech)
            cer_sum += dist / max(len(gt_tech), 1)
            cer_n += 1

    total_hit = sum(v["hit"] for v in per_field.values())
    total_all = sum(v["total"] for v in per_field.values())
    return {
        "arm": arm,
        "images": images_done,
        "field_accuracy": round(total_hit / max(total_all, 1), 4),
        "field_hit": total_hit,
        "field_total": total_all,
        "spurious_rate": round(spurious / max(spurious_total, 1), 4),
        "tech_cer": round(cer_sum / max(cer_n, 1), 4) if cer_n else None,
        "per_field": {f: {"acc": round(v["hit"] / max(v["total"], 1), 4) if v["total"] else None,
                          "hit": v["hit"], "total": v["total"]}
                      for f, v in per_field.items()},
    }


def run_metrics() -> None:
    if not GT_JSON.exists():
        raise SystemExit(f"真值不存在: {GT_JSON}（先校正 ocr_truth_draft.xlsx 并运行 build_gt.py）")
    gt = json.loads(GT_JSON.read_text(encoding="utf-8"))

    all_metrics = {}
    for arm in ARM_FLAGS:
        if (ARMS_DIR / arm).exists() and list((ARMS_DIR / arm).glob("*_result.json")):
            m = compute_arm_metrics(arm, gt)
            all_metrics[arm] = m
            print(json.dumps(m, ensure_ascii=False, indent=2))
        else:
            print(f"[{arm}] 无结果，先运行 --arm {arm}")

    if not all_metrics:
        raise SystemExit("没有任何臂的结果")

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 项目2 OCR 臂评估报告",
        "",
        f"- 真值：ocr_truth/ground_truth.json（{len(gt)} 张图，人工校正）",
        "- 字段准确率 = exact match（去空白）/ 真值非空字段数；误提取率 = 图纸无该字段却提取出值的比例",
        "- 技术要求 CER = 字符编辑距离 / 真值字符数（越小越好）",
        "",
        "| 臂 | 多尺度 | 投票 | 字段准确率 | 误提取率 | 技术要求 CER |",
        "|---|---|---|---|---|---|",
    ]
    for arm, m in all_metrics.items():
        lines.append(
            f"| {arm} | {'关' if 'no_ms' in arm or arm == 'base' else '开'} "
            f"| {'关' if 'no_vt' in arm or arm == 'base' else '开'} "
            f"| {m['field_accuracy']:.1%}（{m['field_hit']}/{m['field_total']}） "
            f"| {m['spurious_rate']:.1%} | {m['tech_cer'] if m['tech_cer'] is not None else '—'} |"
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n指标: {METRICS_JSON}\n报告: {REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 臂评估")
    parser.add_argument("--arm", choices=list(ARM_FLAGS), help="运行指定消融臂")
    parser.add_argument("--metrics", action="store_true", help="对照真值计算指标")
    args = parser.parse_args()

    if args.arm:
        run_arm(args.arm)
    elif args.metrics:
        run_metrics()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
