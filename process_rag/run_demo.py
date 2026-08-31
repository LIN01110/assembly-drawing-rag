"""CLI 端到端 demo：python run_demo.py [零件JSON文件] [--method hybrid]

演示完整链路：检索 → LLM/模板生成 → 派工 → 校验 → 人工审批(interrupt) → 归档。
无零件文件时使用内置示例。审批通过 CLI 输入模拟人工操作。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.types import Command

from src import db
from src.graph import build_graph

EXAMPLE_PART = {
    "part_type": "精密台阶轴",
    "material": "40Cr",
    "tolerance_it": 6,
    "surface_ra": 0.8,
    "features": "台阶轴 同轴度0.02 键槽",
    "tech_requirements": "调质 HRC28-35；同轴度 0.02；Ra0.8",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part_file", nargs="?", default=None)
    ap.add_argument("--method", default="hybrid", choices=["cbr", "vector", "hybrid"])
    ap.add_argument("--auto-approve", action="store_true", help="跳过交互，自动批准（评测用）")
    args = ap.parse_args()

    db.init_db()
    part = EXAMPLE_PART
    if args.part_file:
        part = json.loads(Path(args.part_file).read_text(encoding="utf-8"))

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}
    result = graph.invoke({"part": part, "memory_method": args.method}, config)

    # interrupt 挂起点：展示给人工
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n===== 待人工审批 =====")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        if args.auto_approve:
            decision = {"action": "approve", "actor": "赵强", "role": "批准人", "comment": "auto"}
        else:
            action = input("操作 [approve/reject]: ").strip() or "approve"
            actor = input("审批人姓名: ").strip() or "赵强"
            role = "批准人" if action == "approve" else "审核员"
            decision = {"action": action, "actor": actor, "role": role,
                        "comment": input("批注: ").strip()}
        result = graph.invoke(Command(resume=decision), config)

    print("\n===== 最终工艺卡 =====")
    print(json.dumps(result.get("final", result), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
