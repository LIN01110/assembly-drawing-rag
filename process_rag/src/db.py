"""SQLite 资源库：schema 初始化 + 模拟数据灌库。

三类资源表（机床/材料/人员）+ 审批留痕表 + 工艺卡归档表。
设计原则：LLM 只能通过 tools.py 的只读函数查询；写操作（归档/审批）由后端代码在审批通过后执行。
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "process_rag.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS machine_tools (
    id TEXT PRIMARY KEY,          -- 机床编号
    name TEXT NOT NULL,           -- 名称
    model TEXT NOT NULL,          -- 型号
    workshop TEXT NOT NULL,       -- 所属车间
    processes TEXT NOT NULL,      -- 可加工工序，逗号分隔：车/铣/磨/钻/镗
    precision_it INTEGER NOT NULL,-- 最高可达公差等级 IT（越小越精）
    max_diameter_mm REAL,         -- 最大回转直径
    status TEXT NOT NULL DEFAULT '可用'   -- 可用/检修/停用
);
CREATE TABLE IF NOT EXISTS materials (
    grade TEXT PRIMARY KEY,       -- 材料牌号
    category TEXT NOT NULL,       -- 碳钢/合金钢/铸铁/铝合金
    hardness_hb TEXT,             -- 硬度范围
    heat_treatment TEXT,          -- 典型热处理
    cutting_speed_m_min TEXT,     -- 推荐切削速度范围
    notes TEXT
);
CREATE TABLE IF NOT EXISTS personnel (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,           -- 车间负责人/工艺员/审核员/批准人
    workshop TEXT,                -- 负责车间（负责人用）
    skills TEXT                   -- 资质技能
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    action TEXT NOT NULL,         -- submit/approve/reject/archive
    actor TEXT NOT NULL,
    role TEXT NOT NULL,
    comment TEXT,
    diff TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS process_cards_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT UNIQUE NOT NULL, -- 仅审批通过的工艺卡归档
    part_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,        -- 校验失败隔离：不入库、不进下游，待人工处理
    part_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,   -- 被隔离的工艺草案 + 违规详情
    reason TEXT NOT NULL,
    resolved INTEGER DEFAULT 0,   -- 0=待处理 1=已人工处理
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS master_data_todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT NOT NULL,      -- material / machine / personnel
    item_key TEXT NOT NULL,       -- 未知牌号、未知机床编号等
    context TEXT,                 -- 出现上下文（plan_id / 字段）
    resolved INTEGER DEFAULT 0,   -- 主数据只能人工维护，系统只提醒不写库
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(item_type, item_key)
);
"""

MACHINE_TOOLS = [
    # id, name, model, workshop, processes, precision_it, max_diameter, status
    ("C6132-01", "普通车床", "C6132", "一金工车间", "车,钻", 8, 320, "可用"),
    ("C6132-02", "普通车床", "C6132", "一金工车间", "车,钻", 8, 320, "可用"),
    ("CW6163-01", "卧式车床", "CW6163", "一金工车间", "车,钻,镗", 7, 630, "可用"),
    ("CK6150-01", "数控车床", "CK6150", "二数控车间", "车,钻,镗", 6, 500, "可用"),
    ("CK6150-02", "数控车床", "CK6150", "二数控车间", "车,钻,镗", 6, 500, "检修"),
    ("X5032-01", "立式铣床", "X5032", "一金工车间", "铣,钻", 9, None, "可用"),
    ("XK713-01", "数控铣床", "XK713", "二数控车间", "铣,钻,镗", 7, None, "可用"),
    ("M1432-01", "万能外圆磨床", "M1432", "三精加工车间", "磨", 5, 320, "可用"),
    ("M7130-01", "平面磨床", "M7130", "三精加工车间", "磨", 5, None, "可用"),
    ("Z3050-01", "摇臂钻床", "Z3050", "一金工车间", "钻", 12, None, "可用"),
    ("T68-01", "卧式镗床", "T68", "一金工车间", "镗,铣", 7, None, "可用"),
    ("VMC850-01", "立式加工中心", "VMC850", "二数控车间", "铣,钻,镗", 6, None, "可用"),
]

MATERIALS = [
    ("45", "碳钢", "170-217", "调质 HRC28-32", "80-120", "应用最广的中碳钢，轴类零件常用"),
    ("Q235", "碳钢", "120-160", "不热处理", "100-150", "普通结构件、板件"),
    ("40Cr", "合金钢", "207-241", "调质 HRC28-35", "70-100", "重要轴、齿轮"),
    ("65Mn", "弹簧钢", "210-250", "淬火+中温回火", "60-90", "弹簧、耐磨件"),
    ("HT200", "铸铁", "170-240", "时效处理", "60-100", "箱体、底座，切削性好但脆"),
    ("6061", "铝合金", "95-105", "T6 固溶时效", "300-500", "轻量结构件，高速切削"),
    ("Cr12MoV", "模具钢", "207-255", "淬火 HRC58-62", "40-60", "模具、耐磨零件，磨削为主"),
    ("1Cr18Ni9Ti", "不锈钢", "160-190", "固溶处理", "50-80", "耐蚀件，加工硬化倾向大"),
]

PERSONNEL = [
    ("P001", "王建国", "车间负责人", "一金工车间", "车削/钻削工艺，20年经验"),
    ("P002", "李文静", "车间负责人", "二数控车间", "数控编程，高级技师"),
    ("P003", "张伟", "车间负责人", "三精加工车间", "精密磨削，技师"),
    ("P004", "陈明", "工艺员", None, "机加工艺编制"),
    ("P005", "刘芳", "审核员", None, "工艺审核，熟悉 GB/T 1804/1184"),
    ("P006", "赵强", "批准人", None, "总工艺师"),
]


def init_db(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR REPLACE INTO machine_tools VALUES (?,?,?,?,?,?,?,?)", MACHINE_TOOLS)
    conn.executemany("INSERT OR REPLACE INTO materials VALUES (?,?,?,?,?,?)", MATERIALS)
    conn.executemany("INSERT OR REPLACE INTO personnel VALUES (?,?,?,?,?)", PERSONNEL)
    conn.commit()
    conn.close()
    return db_path


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    print(f"数据库已初始化: {init_db()}")
