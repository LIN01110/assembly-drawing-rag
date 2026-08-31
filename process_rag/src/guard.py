"""输出/接收层守卫：LLM 输出 schema 校验 + Prompt 注入防护。

- validate_plan_schema：接收层对 LLM 工艺草案做字段级结构校验
- sanitize_text：输出层对送入 LLM 的 OCR/人工字段做注入清洗
- DATA_WRAP：prompt 中数据区包裹标记，指令与数据分离
"""
import re

# 注入特征：指令性措辞（中英文常见注入模式）
INJECTION_PATTERNS = [
    r"忽略[以之]前", r"无视.*指令", r"你现在是", r"system\s*prompt", r"ignore\s+(all\s+)?(previous|above)",
    r"forget\s+(all|your)", r" disregard ", r"新指令", r"覆盖.*规则",
]

DATA_WRAP = ("【数据开始】", "【数据结束】")


def sanitize_text(text: str, max_len: int = 500) -> tuple[str, list[str]]:
    """清洗送入 LLM 的文本字段。返回 (清洗后文本, 命中的注入模式列表)。"""
    hits = []
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))
    for pat in INJECTION_PATTERNS:
        if re.search(pat, cleaned, re.I):
            hits.append(pat)
            cleaned = re.sub(pat, "[已过滤]", cleaned, flags=re.I)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
        hits.append("truncated")
    return cleaned, hits


def sanitize_part(part: dict) -> tuple[dict, list[str]]:
    """对零件 dict 的文本字段逐一清洗。"""
    out, all_hits = {}, []
    for k, v in part.items():
        if isinstance(v, str):
            cleaned, hits = sanitize_text(v)
            out[k] = cleaned
            all_hits.extend(hits)
        else:
            out[k] = v
    return out, all_hits


def validate_plan_schema(plan: dict) -> list[dict]:
    """LLM 工艺草案字段级校验（接收层第一道闸，先于业务规则 R1-R8）。"""
    violations: list[dict] = []

    def v(field, message, severity="error"):
        violations.append({"rule": "SCHEMA", "severity": severity,
                           "message": f"字段[{field}]: {message}"})

    if not isinstance(plan, dict):
        return [{"rule": "SCHEMA", "severity": "error", "message": "输出不是 JSON 对象"}]
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        v("steps", "必须是不少于 2 步的工序列表")
    else:
        if any(not isinstance(s, str) or not s.strip() for s in steps):
            v("steps", "存在空工序或非字符串工序")
        if any(len(s) > 40 for s in steps):
            v("steps", "存在超长工序文本（>40 字），疑似生成失控", "warning")
    basis = plan.get("basis")
    if not isinstance(basis, list):
        v("basis", "必须是引用列表", "warning")
    conf = plan.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
        v("confidence", "必须是 0-1 的数值")
    extra = set(plan) - {"steps", "basis", "confidence", "_llm", "_usage", "_error"}
    if extra:
        v("_extra", f"存在计划外字段: {sorted(extra)}", "warning")
    return violations
