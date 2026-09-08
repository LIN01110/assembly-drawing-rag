# 工业装配图 RAG 智能输出系统

> 从装配图输入到结构化 JSON、再到工艺规划与派工的完整链路：YOLO 检测 + OCR 结构化管线识别图纸要素，LangGraph 工艺规划 RAG Agent 完成"检索 → 生成 → 派工 → 合规校验 → 人工审批"闭环。

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-状态机-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-检测-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 仓库结构

```text
├── yolo/          # 模块一：图纸检测 + OCR 结构化管线
│   ├── integrate.py        # YOLO 检测 + OCR 整合主流程
│   ├── eval_pipeline.py    # 检测/字段提取评测（21 张原图）
│   ├── gen_aug.py          # 光度增强生成器（×100 变体，GT 框复用）
│   ├── eval_aug.py         # 增强集双臂评测（分块 + 合并）
│   └── config/             # 训练/预测/信息提取配置
└── process_rag/   # 模块二：工艺规划 RAG Agent（LangGraph）
    ├── src/                # 状态机/检索/合规/守卫/RBAC/多输入适配
    ├── eval/               # 三臂消融评测 + LLM-as-judge + 合成用例生成器
    ├── tests/              # 24 项 pytest 断言回归（零 API 可复现）
    └── knowledge/          # 合成演示语料（公开国标要点改写 + 虚构模拟数据）
```

## 模块一：图纸检测 + OCR 结构化（yolo/）

- SAHI 切片检测 + 多尺度多布局投票 OCR + 超分辨率/笔画增强 + 倾斜校正
- 实测：区域检测召回 **98.7%**（mAP@50，21 张评测集）；消融证明多尺度 OCR 为关键组件（关闭后字段完整度 -50%）
- 技术选型实证：端到端多模态大模型直接解析图纸正确率仅 **23.8%**，据此确立专用管线路线
- **增强压力集（21 原图 ×100 光度变体 = 2100 张，2026-09-09）**：baseline mAP@50 **94.2%** / Precision **94.4%**；SAHI mAP@50 89.5% / Precision 68.9%——噪声/模糊/压缩伪影压力下切片路径误报显著增多，推翻"SAHI 恒优"直觉（`gen_aug.py` + `eval_aug.py` 可复现）

## 模块二：工艺规划 RAG Agent（process_rag/）

- **LangGraph 状态机**：多输入适配（OCR JSON/PDF/Word/人工补全）→ 四库检索（GB 标准/材料/机床/历史工艺卡）→ LLM 工艺生成 → 确定性工具派工 → 合规校验（R1-R8）→ 人工审批（interrupt + RBAC 三角色）→ 归档
- **安全设计**：生产数据只读工具查库（LLM 零写权限）、引用真实性外键校验防幻觉、Prompt 注入清洗、字段级 schema 校验、error 级草案隔离区
- **记忆库**：CBR 结构化过滤 + 向量排序混合召回，只收录审批通过的工艺卡

### 实测指标（25 个零件人工用例，三臂消融）

| 指标 | cbr | vector | hybrid |
|---|---|---|---|
| 派工正确率 | 76.0% | 72.0% | 72.0% |
| 工艺合规率 | 96.0% | 100% | 100% |
| 记忆库 Recall@3 | 100% | 100% | 100% |

**大规模合成集**（2500 用例 ×3 臂，工艺卡参数域内扰动生成，`eval/gen_cases.py` seed=42 可复现）：派工正确率 60.6-64.1%、工艺合规率 86.1%、记忆库 Recall@3 100%、质量门隔离率 13.9%——合成随机样本显著难于人工精选集，暴露了派工规则的边界假设。

**LLM-as-judge 三维评测**（真实 LLM，judge 本身做双跑自一致性元评测）：忠实度 **0.918** / 完整性 **0.930** / 相关性 **0.960**；judge 自一致性最大差均值 0.12。

## 快速开始

```bash
# 模块二（零 API 可复现，LLM 未配置时自动模板降级）
cd process_rag
pip install langgraph scikit-learn pypdf python-docx httpx pytest
python -m pytest tests/ -q          # 24 项断言回归
python eval/gen_cases.py --n 2500   # 合成用例生成
python eval/evaluate.py --no-llm --cases-file eval/cases_synth.json   # 大规模评测
python run_demo.py                  # 接 LLM 需配置 DEEPSEEK_API_KEY
```

## 隐私与数据说明

- **评测图纸、训练权重（.pt/.onnx）、数据集、运行时数据库均未上传**
- `process_rag/knowledge/` 与 `eval/cases*.json` 为**合成演示数据**（GB 条款为公开标准要点改写，机床/人员/工艺卡为虚构），不含任何真实生产数据
- API Key 通过环境变量读取，仓库内无任何密钥
