"""合规校验节点：国标规则 + 数值范围硬校验（确定性，非 LLM）。

规则集：
  R1 IT<=6 或 Ra<=0.8 的零件，工艺步骤必须含"磨"（GB/T 1031 / GB/T 1800.1）
  R2 调质类材料(45/40Cr)且 IT<=7，步骤必须含热处理（调质）
  R3 铸铁件(HT*)必须先时效处理再机加工（GB/T 9439）
  R4 淬硬材料(Cr12MoV/65Mn)热处理后不得安排车/铣，只准磨（GB/T 1299/1222）
  R5 含螺纹特征须有螺纹加工/检验步骤（GB/T 197）
  R6 派工机床必须覆盖全部推导工序且在可用状态（资源约束）
  R7 审批通过前必须至少一条依据引用（可追溯性）
  R8 引用真实性（外键校验）：basis 中的 GB 编号/历史卡号必须在库中存在（防幻觉引用）
"""
import json
import re
from pathlib import Path

HARDENED = {"Cr12MoV", "65Mn"}
QT_MATERIALS = {"45", "40Cr"}  # 调质类
# R2 适用范围：受力结构件关键词（垫圈/钻模板/座体等并非必须调质，避免误报）
LOAD_BEARING = ("主轴", "丝杠", "齿轮", "花键", "凸轮", "摇臂", "手柄", "传动轴", "齿轮轴", "精密小轴")

_GB_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "gb_standards.json"
_CARDS_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "process_cards.json"


def _valid_refs() -> tuple[set[str], set[str]]:
    """库中真实存在的 GB 标准号（数字段）与工艺卡号。"""
    clauses = json.loads(_GB_PATH.read_text(encoding="utf-8"))
    gb_numbers = {re.search(r"(\d+(?:\.\d+)?)", c["code"]).group(1) for c in clauses}
    cards = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
    card_ids = {c["id"] for c in cards}
    return gb_numbers, card_ids


def validate_references(plan: dict) -> list[dict]:
    """R8 外键校验：basis 引用的 GB 编号 / PC 卡号必须在库，否则为幻觉引用。"""
    violations: list[dict] = []
    basis = plan.get("basis", [])
    if not basis:
        return violations
    gb_numbers, card_ids = _valid_refs()
    for ref in basis:
        ref = str(ref)
        for m in re.finditer(r"GB/?T?\s*(\d+(?:\.\d+)?)", ref, re.I):
            if m.group(1) not in gb_numbers:
                violations.append({"rule": "R8", "severity": "error",
                                   "message": f"引用不存在的 GB 标准号: GB/T {m.group(1)}（幻觉引用）"})
        for m in re.finditer(r"PC\d{3}", ref):
            if m.group(0) not in card_ids:
                violations.append({"rule": "R8", "severity": "error",
                                   "message": f"引用不存在的历史工艺卡: {m.group(0)}（幻觉引用）"})
    return violations


def check_compliance(part: dict, plan: dict, assignment: dict) -> list[dict]:
    """返回违规列表 [{rule, message, severity}]；空列表=合规。"""
    violations: list[dict] = []
    steps = plan.get("steps", [])
    steps_text = " ".join(steps)
    it = int(part.get("tolerance_it", 12))
    ra = float(part.get("surface_ra", 6.3))
    material = part.get("material", "")
    features = str(part.get("features", "")) + str(part.get("tech_requirements", ""))

    if (it <= 6 or ra <= 0.8) and "磨" not in steps_text:
        violations.append({"rule": "R1", "severity": "error",
                           "message": f"IT{it}/Ra{ra} 高精度但工艺无磨削工序（GB/T 1031）"})
    if material in QT_MATERIALS and it <= 7 and "调质" not in steps_text and "淬火" not in steps_text:
        ptype = str(part.get("part_type", ""))
        tech = str(part.get("tech_requirements", ""))
        needs_ht = any(kw in ptype for kw in LOAD_BEARING) or "调质" in tech or "淬火" in tech
        if needs_ht:
            violations.append({"rule": "R2", "severity": "error",
                               "message": f"{material} 钢 IT{it} 受力件缺热处理工序"})
    if material.startswith("HT"):
        for i, s in enumerate(steps):
            if re.search(r"车|铣|镗|钻", s) and not any("时效" in x for x in steps[:i]):
                violations.append({"rule": "R3", "severity": "error",
                                   "message": "铸铁件机加工前未安排时效处理（GB/T 9439）"})
                break
    if material in HARDENED:
        for i, s in enumerate(steps):
            if "淬火" in s:
                post = steps[i + 1:]
                if any(re.search(r"车|铣(?!型)", x) for x in post) and not any("磨" in x for x in post):
                    violations.append({"rule": "R4", "severity": "error",
                                       "message": f"{material} 淬硬后安排了车/铣而非磨削"})
                break
    if "螺纹" in features and "螺纹" not in steps_text:
        violations.append({"rule": "R5", "severity": "warning",
                           "message": "含螺纹特征但无螺纹加工步骤（GB/T 197）"})
    from src.tools import infer_processes
    needed = infer_processes(steps)
    covered = set()
    for m in assignment.get("machines", []):
        covered.update(m["processes"].split(","))
    missing = [p for p in needed if p not in covered]
    if missing:
        violations.append({"rule": "R6", "severity": "error",
                           "message": f"派工机床不覆盖工序: {missing}"})
    if not plan.get("basis"):
        violations.append({"rule": "R7", "severity": "warning",
                           "message": "工艺无依据引用，不可追溯"})
    violations.extend(validate_references(plan))  # R8 外键校验
    return violations


def confidence_gate(part: dict, plan: dict, ocr_conf: float | None = None) -> list[str]:
    """低置信字段标红，强制人工确认。返回标红字段列表。"""
    flags = []
    if plan.get("confidence", 0) < 0.7:
        flags.append("plan_confidence")
    if ocr_conf is not None and ocr_conf < 0.85:
        flags.append("ocr_fields")
    it = part.get("tolerance_it")
    if it is None:
        flags.append("tolerance_missing")
    return flags
