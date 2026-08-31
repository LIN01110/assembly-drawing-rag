"""RBAC 权限与审批留痕。

角色：工艺员(提交) / 审核员(审核驳回或通过) / 批准人(最终批准归档)。
状态机：draft → pending_review → approved → archived；任何环节可 rejected 回到 draft。
"""
from src import db

ROLE_ACTIONS = {
    "工艺员": {"submit"},
    "审核员": {"review_pass", "reject"},
    "批准人": {"approve", "reject"},
}

NEXT_STATE = {
    ("draft", "submit"): "pending_review",
    ("pending_review", "review_pass"): "pending_approve",
    ("pending_review", "reject"): "draft",
    ("pending_approve", "approve"): "approved",
    ("pending_approve", "reject"): "draft",
    ("approved", "archive"): "archived",
}


class PermissionError_(Exception):
    pass


def check_permission(role: str, action: str) -> None:
    if action not in ROLE_ACTIONS.get(role, set()):
        raise PermissionError_(f"角色[{role}]无权执行[{action}]")


def log_action(plan_id: str, action: str, actor: str, role: str,
               comment: str = "", diff: str = "") -> None:
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO approvals (plan_id, action, actor, role, comment, diff) VALUES (?,?,?,?,?,?)",
        (plan_id, action, actor, role, comment, diff))
    conn.commit()


def transition(state: str, action: str, role: str) -> str:
    check_permission(role, action)
    nxt = NEXT_STATE.get((state, action))
    if nxt is None:
        raise ValueError(f"非法状态转移: {state} + {action}")
    return nxt


def archive(plan_id: str, part_json: str, plan_json: str) -> None:
    """仅 approved 状态可归档（调用方保证），归档即写入记忆库语料。"""
    conn = db.get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO process_cards_archive (plan_id, part_json, plan_json) VALUES (?,?,?)",
        (plan_id, part_json, plan_json))
    conn.commit()
