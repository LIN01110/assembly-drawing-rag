# 报告预习文档：基于 YOLO + OCR 的机械图纸信息提取系统

> 本文档供 Agent 生成正式报告使用。代码当前未配环境无法运行，但保留有历史运行结果与模型文件。

---

## 一、项目概述

### 1.1 项目目标
构建一个端到端的机械图纸信息自动提取系统，能够从扫描/拍摄的机械工程图中：
- **检测**出两类关键区域：`table`（标题栏/表格）、`tech_req`（技术要求）
- **识别**区域内的文本内容（使用 PaddleOCR）
- **结构化**输出为：物品名称、材料、数量、图号、比例、重量、设计/审核/批准签名、技术要求条目等

### 1.2 核心流程
```
输入图像 → YOLO目标检测(table/tech_req) → 裁剪ROI → OCR文本识别 → 文本聚类合并 → 结构化提取 → TXT/JSON输出
```

### 1.3 关键文件说明
| 文件 | 说明 |
|------|------|
| `integrate.py` | **核心整合脚本（增强版，1604行）**，集成 SAHI、多尺度OCR、投票机制、笔画增强、倾斜校正、纠错字典等 |
| `src/predict.py` | 原始 YOLO 预测脚本，负责加载模型、执行预测、保存裁剪图 |
| `src/easyocr_utils.py` | 原始 OCR 工具类（类名保留 EasyOCRProcessor，实际底层使用 PaddleOCR） |
| `src/train.py` | YOLO 训练功能封装 |
| `dataset/data.yaml` | 数据集配置，nc=2，names=['table', 'tech_req'] |
| `config/predict.yaml` | 预测配置，指定模型路径、source、置信度阈值等 |

---

## 二、模型训练（YOLO）

### 2.1 数据集
- **规模**：约 30 张机械图纸图像（数量较少，是后续检测漏检的主要原因之一）
- **格式**：YOLO 格式标注（`class_id x_center y_center width height`）
- **类别**：
  - `0`: `table` — 标题栏/信息表格区域
  - `1`: `tech_req` — 技术要求文字区域
- **路径**：`dataset/` 或 `dataset_2class/`

### 2.2 训练配置
- **基础模型**：YOLOv8n / YOLO11n（轻量级，适合CPU部署）
- **训练结果目录**：
  - `runs/detect_train/` — 主要训练任务
  - `runs/detect/fresh_training/` — 另一轮训练
- **关键模型文件**：
  - `runs/detect_train/weights/best.pt` — integrate.py 默认调用的模型
  - `runs/detect/fresh_training/weights/best.pt` — 新训练的最佳模型

### 2.3 训练过程可视化（已有图片）
以下图片位于 `runs/detect_train/` 或 `runs/detect/fresh_training/`，可直接用于报告：

| 图片文件 | 用途 |
|---------|------|
| `results.png` | 训练指标汇总（BoxLoss、ClsLoss、mAP50、mAP50-95 随 epoch 变化） |
| `BoxPR_curve.png` | 检测框的 PR 曲线 |
| `BoxP_curve.png` / `BoxR_curve.png` | 精确率/召回率曲线 |
| `BoxF1_curve.png` | F1 分数曲线 |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | 混淆矩阵（绝对值 & 归一化） |
| `labels.jpg` | 数据集标签分布可视化（中心点、宽高分布） |
| `train_batch0.jpg` / `train_batch1.jpg` / `train_batch2.jpg` | 训练批次样本（带标注框） |
| `train_batch1120.jpg` 等 | 后期训练批次（验证增强效果） |
| `val_batch0_labels.jpg` / `val_batch0_pred.jpg` | 验证集标签 vs 预测结果对比 |
| `val_batch1_labels.jpg` / `val_batch1_pred.jpg` | 同上 |

---

## 三、原始 OCR 模块

### 3.1 引擎选择
- 实际使用 **PaddleOCR**（非 EasyOCR，类名是历史遗留）
- 语言：`ch`（中文）
- 配置：开启角度分类 (`use_angle_cls=True`)、文本检测+识别+方向分类

### 3.2 原始预处理流程
```
图像 → 灰度转换 → 自适应阈值二值化 → 高斯模糊去噪 → CLAHE增强 → GRAY2RGB → PaddleOCR
```

### 3.3 原始后处理
- 正则过滤特殊字符
- 全角转半角
- 基于关键词正则提取表格字段（物品、材质、比例、重量、设计、审核、批准等）

---

## 四、integrate.py 增强改进策略（报告重点）

代码头部明确列出了 **6 大改进策略**，均引用自近年论文，是报告的技术亮点：

### 4.1 SAHI（Slicing Aided Hyper Inference）
- **论文**："Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection" (IEEE, 2022)
- **问题**：机械图纸通常 2000×3000 像素以上，直接输入 YOLO 时小目标（技术要求区域）特征占比极小，漏检严重
- **方案**：大图切分为 640×640 重叠切片（overlap 20%），分别检测后用 NMS 合并回原始坐标
- **触发条件**：图像长边 > 1500px 时自动启用
- **已有输出**：`integrate.py` 运行后若启用 SAHI，日志会记录切片数量与合并后目标数

### 4.2 多布局投票机制（Multi-Layout Adjustment Voting）
- **论文**：MDPI Applied Sciences 2025
- **问题**：OCR 识别准确率受画布 padding 大小显著影响
- **方案**：对同一裁剪区域生成 5 种 padding 比例（0.5x, 1.0x, 1.5x, 2.0x, 3.0x）的版本，分别 OCR 后按文本内容分组投票，取出现次数最多且置信度最高的结果
- **适用范围**：默认仅对 `tech_req` 区域启用

### 4.3 超分辨率预处理 + 笔画增强
- **论文**：MDPI Machines 2025 (eDOCr2)
- **问题**：机械图纸文字笔画极细（<2.5px），OCR 难以识别
- **方案**：
  - **超分辨率**：小图使用 `LANCZOS4` 插值放大 1.5x / 2.0x
  - **笔画增强**：距离变换估计平均笔画宽度，过细时使用形态学膨胀（dilate）增强，保持文字结构

### 4.4 多尺度 OCR 融合（Multi-Scale OCR Fusion）
- **方案**：对同一 ROI 同时生成 4 个预处理版本：
  1. `standard` — 标准预处理
  2. `upscale_1.5x` — 1.5 倍放大（适用于 <500px 小图）
  3. `upscale_2x` — 2 倍放大（适用于 <400px 极小图）
  4. `stroke_enhanced` — 笔画增强（条件触发）
- **决策**：分别 OCR 后，选择平均置信度最高的版本作为最终结果
- **已有输出**：`debug_preprocessed/` 目录下会保存各版本的对比图，命名如 `tech_req_big0_1234_5678_standard.jpg`

### 4.5 基于位置的文本聚类合并
- **论文**：eDOCr2 框架
- **问题**：OCR 通常按字/词检测，导致一个完整技术要求条目被断成多段
- **方案**：
  - 按 y 坐标排序，判断是否在“同一行”（y 差异 < 平均高度 × 0.5）
  - 判断 x 间隔是否合理（< 平均字宽 × 3）
  - 满足条件则合并为完整句子，更新联合边界框

### 4.6 倾斜校正（Deskew）
- **方案**：二值化 → 查找文字轮廓 → 最小外接矩形计算倾斜角 → 若 |角度| > 0.5° 则 CUBIC 插值旋转校正，白色背景填充

### 4.7 扩展纠错字典与领域验证
- **纠错字典**：80+ 词条，覆盖：
  - 常见 OCR 错误（O→0, l→1, S→5 等）
  - 机械术语（阀体、盒体、座体等）
  - 材料牌号（ZL102, 5A06-F, 45#钢, HT200, Q235 等）
  - 工艺关键词（热处理、调质、淬火、渗碳、镀锌、粗糙度 Ra、硬度 HRC 等）
- **验证器**：基于 `TECH_REQ_KEYWORDS` 词库打分，过滤低质量识别结果

### 4.8 各类别差异化预处理
| 区域类型 | 特殊处理 |
|---------|---------|
| `tech_req` | NLMeans 去噪(更强) → CLAHE(clipLimit=4.0) → 锐化核 → 自适应二值化(15,10) → 开运算+闭运算 |
| `table/cell/col/row` | NLMeans 去噪 → CLAHE(3.0) → 自适应二值化(11,5) → 闭运算 |
| 其他 | NLMeans 去噪 → CLAHE(2.0) → OTSU 二值化 |

---

## 五、历史运行结果与可用素材

### 5.1 输入图像样本
位于 `pic_batch/` 和 `pic/`，报告可使用：
- `pic_batch/big0.png`, `big1.png`, `big2.png`, `big3.jpg` — 批量测试用图
- `pic/big_scan0.png` ~ `big_scan27.png` — 扫描版工程图（部分为整页 A3/A2 扫描件）

### 5.2 检测可视化图
位于 `output_results/`：
- `detection_20251226_151426.jpg` — YOLO 检测框可视化（table/tech_req 用不同颜色标注，带置信度）
- `detection_20251226_151738.jpg`
- `detection_20251226_152129.jpg`
- `text_annotated_20251226_151426.jpg` — 在检测框基础上叠加 OCR 文本位置（黄色多边形）

### 5.3 裁剪图与预处理调试图
- `output_results/crops/` — YOLO 检测后的原始 ROI 裁剪（`cell_`, `col_`, `row_` 等命名）
- `debug_preprocessed/`（根目录）— 包含三类对比图：
  - `*_original.jpg` — 裁剪原图
  - `*_preprocessed.jpg` — 预处理后图像
  - `*_comparison.jpg` — 左右拼接对比图
- `output_results/debug_preprocessed/` — 另一批预处理调试输出
- `predict/2025*/crops/` — 各次运行保存的裁剪图

### 5.4 OCR 与结构化输出示例
- `predict_results/20251229_093725/big0_ocr.txt` — 结构化 TXT 输出示例：
  ```
  物品：无
  材质：B-Bi
  比例：2:1
  质量：2g
  技术要求：无
  ```
- `predict_results/20251229_093725/summary_*.txt` — 批量任务汇总报告
- `batch_results/20251229_144257/big0_result.json` — JSON 格式完整结果
- `output/big0.txt` ~ `big3.txt` — 另一次运行的提取结果（多数字段为“无”，反映检测/识别效果仍有提升空间）

### 5.5 日志文件
- `integrate.log` — integrate.py 运行日志（包含 SAHI 切片数、检测对象数、OCR 失败记录等）
- `ocr_integrated.log` — OCR 模块历史日志
- `output/extraction_log.txt` — 提取过程详细日志
- `logs/extraction_*.log` — 大量历史运行日志（100+ 条）

---

## 六、报告建议章节结构

供 Agent 生成正式报告时参考：

1. **摘要** — 系统目标、技术路线、主要成果
2. **引言** — 机械图纸信息提取的背景与意义
3. **相关工作** — 工业图纸 OCR、YOLO 小目标检测、SAHI 等论文方法简述
4. **系统架构** — 整体流程图（YOLO 检测 → ROI 裁剪 → OCR 识别 → 后处理 → 结构化输出）
5. **模型训练**
   - 数据集介绍（规模、类别、标注格式）
   - 训练参数与过程
   - 训练结果分析（插入 `results.png`、`PR_curve`、`confusion_matrix`、`val_batch_pred` 等图）
6. **原始 OCR 模块**
   - PaddleOCR 原理简介
   - 原始预处理与后处理流程
7. **增强改进策略（重点）**
   - SAHI 切片辅助检测
   - 多布局投票 OCR
   - 超分辨率与笔画增强
   - 多尺度 OCR 融合
   - 文本聚类合并与倾斜校正
   - 领域纠错字典
   - 每部分配 1~2 张示意图或对比图
8. **实验与结果**
   - 输入图像示例
   - 检测可视化（YOLO 框 + 类别标签）
   - 裁剪与预处理对比（原图 vs 预处理后）
   - OCR 结构化输出示例
   - 错误案例分析（如输出为“无”或材料识别为“B-Bi”等异常）
9. **问题与改进方向**
   - 数据集过小（30张）导致漏检
   - 环境缺失导致当前无法复现
   - 建议：数据增强、YOLOv8m/l 升级、PaddleOCR 微调、LLM 语义纠错
10. **结论**

---

## 七、报告所需图片清单

### 必用图片（已有，直接引用路径）

| 序号 | 图片内容 | 建议路径 | 用途 |
|------|---------|---------|------|
| 1 | 原始机械图纸输入 | `pic_batch/big0.png` | 展示输入样例 |
| 2 | YOLO 训练指标汇总 | `runs/detect_train/results.png` | 模型训练效果 |
| 3 | PR 曲线 | `runs/detect_train/BoxPR_curve.png` | 检测性能评估 |
| 4 | 混淆矩阵 | `runs/detect_train/confusion_matrix.png` | 类别区分能力 |
| 5 | 训练批次样本 | `runs/detect_train/train_batch0.jpg` | 数据增强与标注质量 |
| 6 | 验证集预测对比 | `runs/detect_train/val_batch0_pred.jpg` | 验证集检测效果 |
| 7 | YOLO 检测可视化 | `output_results/detection_20251226_151426.jpg` | 检测框与类别标签 |
| 8 | OCR 文本位置标注 | `output_results/text_annotated_20251226_151426.jpg` | 检测+识别联合可视化 |
| 9 | 裁剪原图 | `debug_preprocessed/cell_1750_1227_original.jpg` | ROI 原貌 |
| 10 | 预处理后图像 | `debug_preprocessed/cell_1750_1227_preprocessed.jpg` | 预处理效果 |
| 11 | 对比图 | `debug_preprocessed/cell_1750_1227_comparison.jpg` | 原图 vs 预处理并排对比 |
| 12 | 技术要求裁剪 | `output_results/crops/col_20251229_095635_1207_1022.jpg` | 表格/技术要求区域示例 |

### 建议补充制作的示意图

| 序号 | 示意图内容 | 说明 |
|------|-----------|------|
| 13 | **系统架构流程图** | 从输入到输出的全流程，需用 draw.io/PPT/代码绘制 |
| 14 | **SAHI 切片示意图** | 展示大图如何切分为 640×640 重叠切片，再合并 NMS |
| 15 | **多布局投票示意图** | 同一 ROI 加不同 padding → 多路 OCR → 投票合并 |
| 16 | **多尺度预处理对比** | standard / 1.5x / 2x / stroke_enhanced 四宫格对比 |
| 17 | **文本聚类合并示意图** | 展示断句的多个小框如何合并为完整条目 |
| 18 | **原始 vs 改进 预处理流程对比** | 左列原始流程，右列改进流程（代码中已有文字版，需可视化） |

---

## 八、已知问题与限制（报告中需诚实说明）

1. **数据集瓶颈**：仅 ~30 张训练图，导致 `tech_req` 区域漏检严重，部分运行结果全为“无”
2. **环境缺失**：当前代码因依赖问题无法直接运行（`torch.load weights_only`、`PaddleOCR` 版本兼容性等）
3. **OCR 错误案例**：如 `材质：B-Bi`（明显识别错误）、`技术要求：无`（实际存在但未检出）
4. **模型大小**：使用 YOLOv8n/YOLO11n 轻量模型，精度有限，建议升级到 m/l 版本
5. **SAHI 兼容性**：integrate.py 中 SAHI 结果通过 `MockResult` 回传，与标准 YOLO Results 对象存在差异，曾导致 `predict() got an unexpected keyword argument 'cls'` 错误

---

## 九、关键代码片段（供报告引用）

### 9.1 类别定义
```yaml
# dataset/data.yaml
nc: 2
names: ['table', 'tech_req']
```

### 9.2 SAHI 切片逻辑
```python
class SAHIInference:
    def slice_image(self, image):
        # 640x640 切片，20% 重叠
        step_h = int(self.slice_height * (1 - 0.2))
        ...
    def merge_predictions(self, slice_predictions):
        # NMS 合并，坐标转换回原图
```

### 9.3 多布局投票
```python
class MultiLayoutVotingOCR:
    self.padding_ratios = [0.5, 1.0, 1.5, 2.0, 3.0]
    def recognize_with_voting(self, image, region_type):
        # 多 padding 版本识别 → 按标准化文本分组 → 取最佳
```

### 9.4 多尺度预处理触发条件
```python
def multi_scale_preprocess(image, region_type):
    versions = [("standard", v1)]
    if max(image.shape[:2]) < 500: versions.append(("upscale_1.5x", v2))
    if max(image.shape[:2]) < 400: versions.append(("upscale_2x", v3))
    # 笔画增强条件触发
```

---

## 十、附录：目录速查

```
yolo/
├── integrate.py                      # 核心增强脚本
├── INTEGRATE_IMPROVEMENTS.md         # 改进策略说明文档（已有，可直接参考）
├── src/
│   ├── predict.py                    # 原始 YOLO 预测
│   ├── easyocr_utils.py              # 原始 PaddleOCR 工具
│   └── utils/
├── config/
│   ├── predict.yaml                  # 预测配置
│   └── train.yaml                    # 训练配置
├── dataset/data.yaml                 # 2类数据集配置
├── runs/
│   ├── detect_train/                 # 训练结果（含 best.pt、曲线图）
│   └── detect/fresh_training/        # 另一轮训练结果
├── pic_batch/                        # 测试输入图（big0~big3）
├── pic/                              # 更多扫描图（big_scan0~27）
├── predict_results/                  # 历史预测+OCR结果（含 summary）
├── batch_results/                    # 批量处理结果
├── output_results/                   # 检测可视化图、裁剪图、OCR结果
├── debug_preprocessed/               # 预处理对比图（original/preprocessed/comparison）
├── predict/                          # integrate.py 多次运行输出
└── output/                           # 早期提取结果 TXT
```

---

> **使用提示**：将本文档提供给 Agent，指示其基于以上素材生成正式技术报告（Word/Markdown/LaTeX）。若需插入图片，直接引用上述路径即可；若需绘制新的架构图/流程图，Agent 可使用 Python (matplotlib/graphviz) 或描述后由用户手动制作。
