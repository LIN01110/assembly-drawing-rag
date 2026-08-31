# 工艺规划 RAG Agent（process_rag）

工业装配图下游的工艺规划模块：接收上游检测/OCR 的结构化零件 JSON，经知识检索增强生成工艺路线，确定性工具派工（车间/机床/负责人），合规校验 + RBAC 人工审批后归档反哺记忆库。

## 架构

```
零件JSON（上游图纸识别给定）
 → parse_input 归一化
 → retrieve   四库检索：GB标准库(TF-IDF) + 材料/机床/人员(SQL只读) + 历史工艺卡记忆库(CBR+向量混合)
 → generate   LLM 工艺草案（DeepSeek JSON mode；失败/无key降级为历史卡模板）
 → dispatch   派工：纯代码查库（LLM 不参与）——机床/车间/负责人可追溯
 → check      合规校验 R1-R7（国标规则+资源约束硬校验）；error 级自动重生成（≤2次）
 → human_review  LangGraph interrupt 人工审批（RBAC 三角色 + 留痕）
 → archive    审批通过归档，写入历史工艺卡（反哺记忆库）
```

## 设计原则

- **LLM 管表达，数据库管事实**：机床参数、负责人、国标数值全部查库返回；写操作仅审批通过后由代码执行，LLM 对生产数据零写权限。
- **记忆库三方法论消融**：`cbr`（材料/公差带/零件类型结构化过滤，可解释）、`vector`（TF-IDF char-ngram 全库排序）、`hybrid`（CBR 圈候选 → TF-IDF 排序，默认）。

## 运行

```bash
python src/db.py                 # 初始化 SQLite（12机床/8材料/6人员）
python run_demo.py               # 交互式审批 demo（内置示例零件）
python run_demo.py --auto-approve  # 自动批准
python eval/evaluate.py --no-llm   # 25 用例 × 3 记忆库臂，零 API 可复现
```

## 评测结果（2026-08-30 第二轮：R8 外键校验 + 隔离区上线后，25 用例 ×3 臂，--no-llm 可复现）

| 记忆库臂 | 派工正确率 | 合规率 | 记忆库 Recall@3 |
|---|---|---|---|
| cbr | 76.0% | 96.0% | 100% |
| vector | 72.0% | 100.0% | 100% |
| hybrid | 72.0% | 100.0% | 100% |

- 新增 **R8 引用真实性校验**（basis 中的 GB 编号/工艺卡号必须在库，防幻觉引用）与**隔离区**（重生成超限的 error 级草案隔离到 quarantine 表待人工处理，不入库不进下游）。
- 第三轮（2026-08-30）输出/接收层加固：LLM 输出字段级 **schema 校验**（guard.py）；输入**类型规范化**（normalize.py：材料别名、±偏差→IT 估算、Ra 归一，变更留痕）；**Prompt 注入防护**（模式清洗 + 数据区包裹指令隔离，实测拦截）；**主数据待办**（未知牌号 → master_data_todos 提醒人工，系统不写主数据）。
- 第四轮（2026-08-30）**多输入适配层**（ingest.py）：OCR result.json / 文字型 PDF / Word 检验单 / 扫描件标记 needs_ocr 四通道 → 统一零件 JSON；人工补全合并（人工值优先 + 留痕）。测试：`scripts/test_ingest.py` 四通道端到端通过。
- 第五轮（2026-08-30）**pytest 断言式回归固化**：`tests/test_process_rag.py` 24 项用例全绿（归一化/schema 校验/R1-R3 合规/引用真实性/记忆库三臂/GB 检索/RBAC/graph 端到端归档+隔离/四通道 ingest/judge 输出解析），零 API 可复现（`PROCESS_RAG_NO_LLM=1`），2 秒内跑完。
- 第六轮（2026-08-30）**LLM-as-judge 柔性评测**（`eval/llm_judge.py`，25 例真实 LLM 实测）：领域三维 rubric——忠实度 **0.918** / 完整性 **0.930** / 相关性 **0.960**；judge 双跑自一致性最大差均值 0.12（5 例元评测）。两个真实发现：① judge 首轮 11 例误报，根因是 judge 上下文缺生成端可见的材料手册——**judge 上下文必须 ≥ 生成端上下文**，补齐后重评 0.864→0.918；② 抓到真实幻觉 E08（编造"GB/T 1804 m级见PC008备注"），R8 只校验编号在库，**越界引用靠语义 judge 补位**。未用 ragas 包：其指标面向通用 QA，且需另配 LLM+embedding 双客户端。
- 实测隔离案例：E14 齿轮轴 cbr 臂召回不当工艺卡导致模板缺磨削，被 R1 拦截、重生成超限后隔离——质量门闭环真实生效。
- 指标为确定性计算（派工/合规由代码工具决定），LLM 仅生成工艺文本；LLM 路径经 `run_demo.py` 实测连通（DeepSeek 返回含 GB 引用的工序链）。
- 派工失败案例（如阀体 E24）为跨车间规则边界 case，是下一轮规则迭代的输入——符合"评测驱动迭代"的工程叙事。
- 数据口径：GB 条款为公开标准要点改写的演示语料；机床/人员/工艺卡为模拟数据。面试口径：实习接触真实生产数据（保密不可带出），本复现版架构与生产一致。

## 文件

- `src/db.py` schema + 模拟数据；`src/tools.py` 只读查询工具 + 派工规则
- `src/memory.py` 记忆库/GB 库检索；`src/compliance.py` R1-R7 合规规则 + 置信度标红
- `src/rbac.py` 三角色权限 + 审批留痕；`src/graph.py` LangGraph 状态机（interrupt 人工在环）
- `eval/cases.json` 25 评测用例；`results/process_rag_eval.json` 评测输出
