"""合成评测用例生成器：从 30 张历史工艺卡做参数域内扰动采样。

真值合法性设计（关键）：
- 扰动严格限制在"派工规则区域"内（车间不变），因此 card.workshop 是合法 GT；
- similar_cards=[源卡 id]，特征/材料/零件类型保留，recall 真值成立；
- 生成器与评测脚本同仓库，第三方可复现——合成集不是黑箱数字。

运行：python eval/gen_cases.py [--n 2500] [--seed 42] [--out eval/cases_synth.json]
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = json.loads((ROOT / "knowledge" / "process_cards.json").read_text(encoding="utf-8"))

NOISE_FEATURES = ["键槽", "台阶", "法兰", "螺纹孔", "油孔", "倒角", "退刀槽", "中心孔"]
TECH_REQ = {"45": "调质 HRC28-32", "40Cr": "调质 HRC28-32", "65Mn": "淬火 HRC45-50",
            "Cr12MoV": "淬火 HRC58-62", "Q235": "时效", "HT200": "时效",
            "6061": "阳极氧化", "1Cr18Ni9Ti": "固溶处理"}


def region_params(card: dict, rng: random.Random) -> dict:
    """按源卡车间归属约束扰动范围，保证派工 GT 合法。"""
    ws = card["workshop"]
    if ws == "三精加工车间":  # 精加工域：IT≤6 或 Ra≤0.8
        it = rng.choice([5, 6])
        ra = rng.choice([0.4, 0.8])
    elif ws == "二数控车间":  # 数控域：IT≥7 且 Ra≥1.6（铣镗复合由工艺卡 steps 保证）
        it = rng.choice([7, 8, 9])
        ra = rng.choice([1.6, 3.2])
    else:  # 一金工域：IT≥7 且 Ra≥1.6
        it = rng.choice([7, 8, 9, 11, 12])
        ra = rng.choice([1.6, 3.2, 6.3])
    feats = set(card["features"].split())
    feats |= set(rng.sample(NOISE_FEATURES, rng.randint(0, 2)))
    feats -= {"磨", "磨削"} if ws != "三精加工车间" else set()
    return {"tolerance_it": it, "surface_ra": ra, "features": " ".join(sorted(feats))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "eval" / "cases_synth.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cases = []
    for i in range(args.n):
        card = rng.choice(CARDS)
        p = region_params(card, rng)
        cases.append({
            "id": f"S{i + 1:04d}",
            "part": {"part_type": card["part_type"], "material": card["material"],
                     "tolerance_it": p["tolerance_it"], "surface_ra": p["surface_ra"],
                     "features": p["features"],
                     "tech_requirements": TECH_REQ.get(card["material"], "")},
            "expect_workshop": card["workshop"],
            "similar_cards": [card["id"]],
            "_source_card": card["id"],
        })
    Path(args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    ws_dist = {}
    for c in cases:
        ws_dist[c["expect_workshop"]] = ws_dist.get(c["expect_workshop"], 0) + 1
    print(f"生成 {len(cases)} 例 → {args.out}（seed={args.seed}）车间分布: {ws_dist}")


if __name__ == "__main__":
    main()
