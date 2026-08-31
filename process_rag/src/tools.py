"""只读查询工具：LLM/流程节点只能通过这里访问资源库（防 LLM 污染生产数据）。

所有函数只读；写路径只有 audit.log_action 与 archive.approve_and_archive（审批通过后由代码执行）。
"""
from src import db


def query_machines(process: str | None = None, max_it: int | None = None,
                   workshop: str | None = None) -> list[dict]:
    """查机床：按工序类型 / 最高公差等级 / 车间过滤，仅返回可用机床。"""
    sql = "SELECT * FROM machine_tools WHERE status='可用'"
    params: list = []
    if workshop:
        sql += " AND workshop=?"; params.append(workshop)
    if max_it is not None:
        sql += " AND precision_it<=?"; params.append(max_it)
    rows = db.get_conn().execute(sql, params).fetchall()
    result = [dict(r) for r in rows]
    if process:
        result = [r for r in result if process in r["processes"]]
    return result


def query_material(grade: str) -> dict | None:
    row = db.get_conn().execute("SELECT * FROM materials WHERE grade=?", (grade,)).fetchone()
    return dict(row) if row else None


def query_workshop_owner(workshop: str) -> dict | None:
    row = db.get_conn().execute(
        "SELECT * FROM personnel WHERE role='车间负责人' AND workshop=?", (workshop,)).fetchone()
    return dict(row) if row else None


def assign_workshop(part: dict) -> dict:
    """派工规则（确定性代码，非 LLM）：
    - IT<=6 或 Ra<=0.8 或需磨 → 三精加工车间
    - 含数控/型腔/加工中心需求（铣+镗+钻复合）→ 二数控车间
    - 其余 → 一金工车间
    返回 {workshop, owner, machines, reason}
    """
    it = int(part.get("tolerance_it", 12))
    ra = float(part.get("surface_ra", 6.3))
    processes = part.get("processes", [])  # 由工艺步骤推导
    if it <= 6 or ra <= 0.8 or "磨" in processes:
        workshop, reason = "三精加工车间", f"IT{it}/Ra{ra} 或含磨削 → 精加工车间"
    elif len(set(processes) & {"铣", "镗"}) >= 2:
        workshop, reason = "二数控车间", "铣镗复合工序 → 数控车间加工中心"
    else:
        workshop, reason = "一金工车间", "常规车钻铣 → 一金工车间"
    owner = query_workshop_owner(workshop)
    # 机床按工序全厂匹配（零件跨车间流转：粗加工在普通车间，精加工在精加工车间）
    machines = []
    for p in processes:
        machines.extend(query_machines(process=p, max_it=it))
    # 去重
    seen, uniq = set(), []
    for m in machines:
        if m["id"] not in seen:
            seen.add(m["id"]); uniq.append(m)
    return {"workshop": workshop, "owner": owner, "machines": uniq, "reason": reason}


def raise_master_data_todo(item_type: str, item_key: str, context: str = "") -> None:
    """主数据待办：遇到库中不存在的牌号/机床等，提醒人工录入，系统不自动写主数据。"""
    conn = db.get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO master_data_todos (item_type, item_key, context) VALUES (?,?,?)",
        (item_type, item_key, context))
    conn.commit()


STEPS_TO_PROCESSES = {    "车": "车", "铣": "铣", "磨": "磨", "钻": "钻", "镗": "镗",
    "刨": "铣", "插": "铣", "研": "磨", "刮研": "磨",
}


def infer_processes(steps: list[str]) -> list[str]:
    """从工艺步骤文本推导所需工序类型集合。"""
    out = []
    for step in steps:
        for kw, proc in STEPS_TO_PROCESSES.items():
            if kw in step and proc not in out:
                out.append(proc)
    return out
