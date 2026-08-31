"""
eval_vlm_arm.py — 项目2 P0 VLM 对照臂

对照设计（检测侧，诚实口径）：
  - 任务：让 VLM 直接数每张图纸中"独立信息板块（标题栏/明细表/技术要求区等）"的数量，
    与检测真值框数量（mydata/labels/train）对比，输出数量准确率/平均绝对误差。
  - 意义：回答"端到端多模态大模型能不能替代 YOLO+OCR 管线的检测环节"。
  - 不做：不让 VLM 画框（VLM 定位坐标不可靠，无法算 IoU，避免编造指标）。
  - OCR 字段提取对照：**待文字真值人工转写后补跑**（本脚本仅占位）。

后端：OpenAI 兼容接口（chat/completions，base64 image_url）。
配置（环境变量优先，其次 项目2/yolo/.env）：
  VLM_API_KEY   （缺省回退 DEEPSEEK_API_KEY）
  VLM_BASE_URL  （缺省回退 DEEPSEEK_BASE_URL，再缺省 https://api.deepseek.com）
  VLM_MODEL     （缺省回退 DEEPSEEK_MODEL，再缺省 deepseek-v4-pro）

用法：
  .conda/python.exe eval_vlm_arm.py --limit 1     # 试跑 1 张
  .conda/python.exe eval_vlm_arm.py               # 全量 21 张
输出：
  results/eval/vlm_arm.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "mydata" / "images" / "train"
LABELS_DIR = ROOT / "mydata" / "labels" / "train"
OUT_JSON = ROOT / "results" / "eval" / "vlm_arm.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PROMPT = (
    "这是一张机械工程图纸。请数出图中有几个独立的信息板块"
    "（例如标题栏、明细表、技术要求区、参数表等被边框围起来的独立区域）。"
    "只回答一个整数，不要任何解释。"
)

MAX_EDGE = 1568  # 控制 base64 体积


def load_config() -> dict:
    env_file = ROOT / ".env"
    file_vars: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                file_vars[k.strip()] = v.strip().strip('"').strip("'")

    def pick(*names: str, default: str = "") -> str:
        for n in names:
            if os.environ.get(n):
                return os.environ[n]
            if file_vars.get(n):
                return file_vars[n]
        return default

    return {
        "api_key": pick("VLM_API_KEY", "DEEPSEEK_API_KEY"),
        "base_url": pick("VLM_BASE_URL", "DEEPSEEK_BASE_URL", default="https://api.deepseek.com"),
        "model": pick("VLM_MODEL", "DEEPSEEK_MODEL", default="deepseek-v4-pro"),
    }


def gt_count(stem: str) -> int:
    label = LABELS_DIR / f"{stem}.txt"
    if not label.exists():
        return 0
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def encode_image(path: Path) -> str:
    import cv2
    import numpy as np

    # cv2.imread 在 Windows 下不支持中文路径，改用 imdecode
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    h, w = img.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def ask_vlm_count(cfg: dict, image_b64: str) -> tuple[int | None, str]:
    """返回 (解析出的整数, 原始回答文本)。失败返回 (None, 错误信息)。"""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 1024,  # 推理模型会先输出 reasoning_content，需要足够预算
        "temperature": 0,
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {cfg['api_key']}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        return None, f"请求失败: {e}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        msg = resp.json()["choices"][0]["message"]
        text = msg.get("content") or ""
        # 推理模型：content 为空时从 reasoning_content 末尾找最后结论数字
        if not text.strip() and msg.get("reasoning_content"):
            nums = re.findall(r"\d+", msg["reasoning_content"])
            return (int(nums[-1]) if nums else None), "[reasoning] " + msg["reasoning_content"][-150:]
    except (KeyError, IndexError, ValueError) as e:
        return None, f"响应解析失败: {e}: {resp.text[:300]}"
    m = re.search(r"\d+", text or "")
    return (int(m.group()) if m else None), (text or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 对照臂：图纸板块计数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 张（配合 --limit 分块跑）")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg["api_key"]:
        raise SystemExit("缺少 API key：设 VLM_API_KEY / DEEPSEEK_API_KEY 环境变量或 项目2/yolo/.env")
    print(f"后端: {cfg['base_url']} 模型: {cfg['model']}")

    image_paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.skip > 0:
        image_paths = image_paths[args.skip:]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    results = []
    for i, p in enumerate(image_paths, 1):
        gt = gt_count(p.stem)
        t0 = time.time()
        try:
            b64 = encode_image(p)
            pred, raw = ask_vlm_count(cfg, b64)
        except Exception as e:  # noqa: BLE001 — 单图失败不阻塞全量
            pred, raw = None, f"处理异常: {e}"
        results.append({
            "image": p.name, "gt_count": gt, "vlm_count": pred,
            "raw": raw[:200], "elapsed_sec": round(time.time() - t0, 1),
        })
        print(f"[{i}/{len(image_paths)}] {p.name}: GT={gt} VLM={pred} ({raw[:60]})")

    # 分块跑时与已有结果按图像名合并
    if OUT_JSON.exists():
        try:
            old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            merged = {d["image"]: d for d in old.get("details", [])}
            for r in results:
                merged[r["image"]] = r
            results = sorted(merged.values(), key=lambda d: d["image"])
        except (ValueError, KeyError):
            pass

    answered = [r for r in results if isinstance(r["vlm_count"], int)]
    exact = sum(1 for r in answered if r["vlm_count"] == r["gt_count"])
    mae = (sum(abs(r["vlm_count"] - r["gt_count"]) for r in answered) / len(answered)) if answered else None
    summary = {
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "images": len(results),
        "answered": len(answered),
        "count_exact_acc": round(exact / len(answered), 4) if answered else None,
        "count_mae": round(mae, 3) if mae is not None else None,
        "note": "计数对照（非 IoU 检测指标）；OCR 字段对照待文字真值人工转写后补跑",
        "details": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
