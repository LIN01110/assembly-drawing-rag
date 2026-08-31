"""评测：派工正确率 / 合规率 / 记忆库消融（cbr vs vector vs hybrid）。

指标均为确定性计算（派工与合规由代码工具决定，不依赖 LLM 是否在线）：
  - 派工正确率：分配车间 == 期望车间
  - 合规率：check_compliance 无 error 级违规
  - 记忆库 Recall@3：召回 top3 中包含期望相似工艺卡
运行：python eval/evaluate.py [--no-llm]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.types import Command

from src import db
from src.compliance import check_compliance
from src.graph import build_graph
from src.memory import MemoryStore
from src.tools import assign_workshop, infer_processes

CASES_PATH = Path(__file__).parent / "cases.json"
RESULTS_DIR = ROOT / "results"


def run_case(case: dict, method: str) -> dict:
    part = case["part"]
    # 记忆库召回（消融臂）
    memory = MemoryStore()
    cards = memory.recall(part, method=method, top_k=3)
    recall_hit = any(c["id"] in case.get("similar_cards", []) for c, _ in cards)

    # 生成工艺（默认模板降级即可复现；LLM 在线时走 LLM）
    graph = build_graph()
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{method}"}}
    result = graph.invoke({"part": part, "memory_method": method}, config)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume={
            "action": "approve", "actor": "eval", "role": "批准人", "comment": "eval"}), config)
    final = result.get("final", {})

    dispatch_ok = final.get("workshop") == case["expect_workshop"]
    violations = final.get("violations", [])
    compliance_ok = not any(v["severity"] == "error" for v in violations)
    return {
        "id": case["id"], "method": method,
        "recall_hit": recall_hit,
        "dispatch_ok": dispatch_ok,
        "expect_workshop": case["expect_workshop"], "got_workshop": final.get("workshop"),
        "compliance_ok": compliance_ok,
        "violations": violations,
        "llm_used": final.get("llm_used"),
        "archived": final.get("archived", False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="全程模板降级，75 次调用零 API（可复现）")
    ap.add_argument("--llm-hybrid-only", action="store_true",
                    help="仅 hybrid 臂用 LLM，其余两臂 --no-llm（混合模式）")
    args = ap.parse_args()
    if args.no_llm:
        os.environ["PROCESS_RAG_NO_LLM"] = "1"
    db.init_db()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    print(f"评测用例数: {len(cases)}")

    records: list[dict] = []
    for case in cases:
        for method in ("cbr", "vector", "hybrid"):
            r = run_case(case, method)
            records.append(r)
            print(f"[{case['id']}|{method}] 派工={'✓' if r['dispatch_ok'] else '✗'} "
                  f"合规={'✓' if r['compliance_ok'] else '✗'} 记忆库={'✓' if r['recall_hit'] else '✗'}")

    def rate(rs, key):
        return round(sum(1 for r in rs if r[key]) / len(rs), 4) if rs else 0.0

    summary = {"cases": len(cases)}
    for method in ("cbr", "vector", "hybrid"):
        rs = [r for r in records if r["method"] == method]
        summary[method] = {
            "dispatch_acc": rate(rs, "dispatch_ok"),
            "compliance_rate": rate(rs, "compliance_ok"),
            "memory_recall@3": rate(rs, "recall_hit"),
        }
    summary["llm_used_count"] = sum(1 for r in records if r["llm_used"])
    summary["archived_count"] = sum(1 for r in records if r["archived"])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "process_rag_eval.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
