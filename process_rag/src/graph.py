"""LangGraph 工艺规划编排：解析→检索→生成→派工→校验→人工审批（interrupt）→归档。

人工在环：审批节点用 interrupt() 挂起，CLI/调用方以 Command(resume={action, actor, role, comment}) 恢复。
审批通过 → 归档写入历史工艺卡（反哺记忆库）；驳回 → 回 generate 节点重生成（最多 2 次）。
"""
import json
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src import rbac
from src.compliance import check_compliance, confidence_gate
from src.guard import sanitize_part, validate_plan_schema
from src.llm import generate_process_plan
from src.memory import GBStore, MemoryStore
from src.normalize import normalize_part
from src.tools import assign_workshop, infer_processes, query_material, raise_master_data_todo

_memory: MemoryStore | None = None
_gb: GBStore | None = None


def stores() -> tuple[MemoryStore, GBStore]:
    global _memory, _gb
    if _memory is None:
        _memory = MemoryStore()
    if _gb is None:
        _gb = GBStore()
    return _memory, _gb


class PlanState(TypedDict, total=False):
    plan_id: str
    part: dict                 # 上游图纸结构化 JSON（零件类型/材料/公差/Ra/特征/技术要求）
    ocr_conf: float | None
    memory_method: str         # cbr / vector / hybrid
    cards: list                # 记忆库召回
    clauses: list              # GB 条款召回
    plan: dict                 # LLM 工艺草案
    assignment: dict           # 派工结果（确定性工具）
    violations: list
    red_flags: list
    approval_state: str        # draft/pending_review/pending_approve/approved/archived
    audit: list
    regenerate_count: int
    final: dict


def parse_input(state: PlanState) -> dict:
    # 输入层：类型规范化（别名/公差→IT/Ra 归一）→ 注入清洗，全程留痕
    part, norm_changes = normalize_part(dict(state["part"]))
    part, injection_hits = sanitize_part(part)
    part.setdefault("tolerance_it", 12)
    part.setdefault("surface_ra", 6.3)
    audit = [{"action": "normalize", "actor": "system", "role": "系统",
              "comment": "; ".join(norm_changes or ["无需归一"])}]
    red_flags = []
    if injection_hits:
        audit.append({"action": "sanitize", "actor": "system", "role": "系统",
                      "comment": f"注入防护命中: {sorted(set(injection_hits))}"})
        red_flags.append("injection_sanitized")
    # 主数据检查：未知材料牌号 → 人工待办（系统不自动写主数据）
    if part.get("material") and not query_material(part["material"]):
        raise_master_data_todo("material", part["material"],
                               context=f"plan 输入材料未知")
        red_flags.append("material_unknown")
    return {"plan_id": state.get("plan_id") or f"PLAN-{uuid.uuid4().hex[:8]}",
            "part": part, "approval_state": "draft", "audit": audit,
            "red_flags": red_flags, "regenerate_count": 0}


def retrieve(state: PlanState) -> dict:
    memory, gb = stores()
    part = state["part"]
    cards = memory.recall(part, method=state.get("memory_method", "hybrid"), top_k=3)
    query = f"{part.get('part_type','')} {part.get('material','')} {part.get('features','')} {part.get('tech_requirements','')}"
    clauses = gb.search(query, top_k=3)
    return {"cards": [{"card": c, "score": s} for c, s in cards],
            "clauses": [{"clause": c, "score": s} for c, s in clauses]}


def generate(state: PlanState) -> dict:
    part = state["part"]
    material = query_material(part.get("material", ""))
    cards = [(item["card"], item["score"]) for item in state["cards"]]
    clauses = [(item["clause"], item["score"]) for item in state["clauses"]]
    plan = generate_process_plan(part, material, cards, clauses)
    return {"plan": plan, "regenerate_count": state.get("regenerate_count", 0) + 1}


def dispatch(state: PlanState) -> dict:
    """派工：纯代码查库（LLM 不参与），保证机床/车间/负责人可追溯。"""
    part = dict(state["part"])
    part["processes"] = infer_processes(state["plan"].get("steps", []))
    return {"assignment": assign_workshop(part)}


def check(state: PlanState) -> dict:
    # 接收层两道闸：字段级 schema 校验（先于业务规则）→ R1-R8 业务合规
    violations = validate_plan_schema(state["plan"]) + \
        check_compliance(state["part"], state["plan"], state["assignment"])
    red_flags = confidence_gate(state["part"], state["plan"], state.get("ocr_conf"))
    # 合并 parse_input 阶段的红标（注入清洗/未知材料）
    red_flags = sorted(set(red_flags) | set(state.get("red_flags", [])))
    return {"violations": violations, "red_flags": red_flags}


def route_after_check(state: PlanState) -> str:
    has_error = any(v["severity"] == "error" for v in state["violations"])
    if has_error and state.get("regenerate_count", 0) < 2:
        return "generate"  # 违规自动重生成（复用幻觉守卫模式）
    if has_error:
        return "quarantine"  # 重生成超限 → 隔离区，不进人工审批
    return "human_review"


def quarantine(state: PlanState) -> dict:
    """隔离区：校验失败且重生成超限的工艺草案隔离待人工处理，不入库不进下游。"""
    from src import db
    payload = {"plan": state["plan"], "assignment": state["assignment"],
               "violations": state["violations"]}
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO quarantine (plan_id, part_json, payload_json, reason) VALUES (?,?,?,?)",
        (state["plan_id"], json.dumps(state["part"], ensure_ascii=False),
         json.dumps(payload, ensure_ascii=False, default=str),
         "; ".join(v["message"] for v in state["violations"] if v["severity"] == "error")))
    conn.commit()
    final = {"plan_id": state["plan_id"], "part": state["part"],
             "quarantined": True, "violations": state["violations"], "archived": False,
             # 保留派工结果供评估与人工复核（派工在隔离前已完成，可追溯）
             "workshop": state["assignment"].get("workshop"),
             "owner": (state["assignment"].get("owner") or {}).get("name"),
             "machines": [m["id"] for m in state["assignment"].get("machines", [])]}
    return {"approval_state": "quarantined", "final": final}


def human_review(state: PlanState) -> dict:
    """人工审批节点：interrupt 挂起，resume 值 = {action, actor, role, comment}。"""
    summary = {
        "plan_id": state["plan_id"],
        "part": state["part"],
        "plan": state["plan"],
        "assignment": state["assignment"],
        "violations": state["violations"],
        "red_flags": state["red_flags"],
    }
    decision = interrupt(summary)  # ← 人工在环：审核员/批准人在此介入
    action = decision.get("action", "reject")
    actor = decision.get("actor", "unknown")
    role = decision.get("role", "审核员")
    comment = decision.get("comment", "")
    # RBAC 校验 + 留痕
    rbac.check_permission(role, action)
    rbac.log_action(state["plan_id"], action, actor, role, comment)
    audit = state.get("audit", []) + [{"action": action, "actor": actor, "role": role, "comment": comment}]
    if action == "reject":
        return {"approval_state": "draft", "audit": audit, "_rejected": True}
    return {"approval_state": "approved", "audit": audit}


def route_after_review(state: PlanState) -> str:
    if state.get("approval_state") == "approved":
        return "archive"
    if state.get("regenerate_count", 0) < 3:
        return "generate"
    return END


def archive(state: PlanState) -> dict:
    part_json = json.dumps(state["part"], ensure_ascii=False)
    plan_json = json.dumps({"plan": state["plan"], "assignment": state["assignment"]},
                           ensure_ascii=False, default=str)
    rbac.archive(state["plan_id"], part_json, plan_json)
    rbac.log_action(state["plan_id"], "archive", "system", "批准人", "审批通过自动归档")
    final = {
        "plan_id": state["plan_id"],
        "part": state["part"],
        "steps": state["plan"].get("steps", []),
        "basis": state["plan"].get("basis", []),
        "llm_used": state["plan"].get("_llm", False),
        "workshop": state["assignment"]["workshop"],
        "owner": (state["assignment"].get("owner") or {}).get("name"),
        "machines": [m["id"] for m in state["assignment"]["machines"]],
        "violations": state["violations"],
        "red_flags": state["red_flags"],
        "audit": state["audit"],
        "archived": True,
    }
    return {"approval_state": "archived", "final": final}


def build_graph():
    g = StateGraph(PlanState)
    g.add_node("parse_input", parse_input)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("dispatch", dispatch)
    g.add_node("check", check)
    g.add_node("quarantine", quarantine)
    g.add_node("human_review", human_review)
    g.add_node("archive", archive)
    g.add_edge(START, "parse_input")
    g.add_edge("parse_input", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "dispatch")
    g.add_edge("dispatch", "check")
    g.add_conditional_edges("check", route_after_check, ["generate", "human_review", "quarantine"])
    g.add_conditional_edges("human_review", route_after_review, ["generate", "archive", END])
    g.add_edge("archive", END)
    g.add_edge("quarantine", END)
    return g.compile(checkpointer=MemorySaver())
