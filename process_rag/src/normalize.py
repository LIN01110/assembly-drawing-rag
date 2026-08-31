"""输入类型规范化：把 OCR/人工录入的脏字段归一到标准零件 schema。

能力：
- 材料牌号别名归一（Q45→45、A3→Q235、45#→45…）
- 公差文本 → IT 等级估算（"±0.02"→查偏差-IT 对照表，名义尺寸未知时按 18-80mm 档简化估算）
- 表面粗糙度变体归一（"Ra 1.6"/"ra1.6"/"√1.6"→1.6）
- 返回 (规范化后 part, 变更记录 list) —— 变更留痕，支撑"数据治理"叙事
"""
import re

MATERIAL_ALIASES = {
    "Q45": "45", "45#": "45", "45钢": "45", "S45C": "45",
    "A3": "Q235", "Q235A": "Q235",
    "40CR": "40Cr", "40铬": "40Cr",
    "HT200": "HT200", "灰铁200": "HT200",
    "AL6061": "6061", "6061铝": "6061", "6061-T6": "6061",
    "CR12MOV": "Cr12MoV",
    "304": "1Cr18Ni9Ti",  # 近似映射，演示口径
}

# 偏差(mm) → IT 等级估算表（名义尺寸 18-80mm 档简化；±d 的全偏差为 2d）
def deviation_to_it(d: float) -> int:
    table = [(0.0065, 6), (0.011, 7), (0.018, 8), (0.027, 9),
             (0.043, 10), (0.070, 11), (0.110, 12)]
    for limit, it in table:
        if d <= limit:
            return it
    return 13


def normalize_part(part: dict) -> tuple[dict, list[str]]:
    out = dict(part)
    changes: list[str] = []

    # 1. 材料别名归一
    grade = str(out.get("material", "")).strip()
    if grade in MATERIAL_ALIASES:
        out["material"] = MATERIAL_ALIASES[grade]
        changes.append(f"材料 {grade}→{out['material']}")

    # 2. 公差文本 → tolerance_it（仅在缺失时推断）
    blob = f"{out.get('features','')} {out.get('tech_requirements','')}"
    if out.get("tolerance_it") in (None, ""):
        m = re.search(r"[±＋\-+]\s*0?\.(\d+)", blob)
        if m:
            d = float(f"0.{m.group(1)}")
            out["tolerance_it"] = deviation_to_it(d)
            changes.append(f"公差 ±{d}mm → IT{out['tolerance_it']}（估算）")
        elif re.search(r"未注|一般公差", blob):
            out["tolerance_it"] = 12
            changes.append("未注公差 → IT12（GB/T 1804 m 级兜底）")

    # 3. 表面粗糙度归一
    if out.get("surface_ra") in (None, ""):
        m = re.search(r"(?:Ra|√|▽{1,3})\s*([0-9.]+)", blob, re.I)
        if m:
            out["surface_ra"] = float(m.group(1))
            changes.append(f"Ra 归一 → {out['surface_ra']}")

    return out, changes
