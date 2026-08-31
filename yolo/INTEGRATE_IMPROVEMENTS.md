# integrate.py 改进说明文档

## 一、问题诊断

### 1.1 原始代码存在的问题

| 问题类型 | 具体表现 | 根本原因 |
|---------|---------|---------|
| **YOLO检测效果差** | 技术要求区域检测不到或漏检 | 大图像中小目标检测困难；数据集小(仅30张)；检测阈值设置不合理 |
| **OCR匹配效果差** | 识别出的文字错误多、断句严重 | 图像分辨率低；预处理不足；缺少倾斜校正；无笔画增强 |
| **文本结构化差** | 技术要求条目分割不准确 | 缺少智能的文本聚类合并；序号识别模式不够丰富 |
| **后处理缺失** | 识别结果无纠错、无验证 | 缺少领域知识词典；缺少置信度验证 |

---

## 二、改进策略（基于最新论文研究）

### 2.1 SAHI (Slicing Aided Hyper Inference) - 切片辅助超推理

**参考论文**: 
- "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection" (IEEE, 2022)
- "SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients" (2024)

**核心思想**:
大图像（如机械图纸扫描件，通常2000x3000像素以上）直接输入YOLO时，小目标（如技术要求区域）在特征图中只占很少像素，导致检测困难。SAHI将大图切分为重叠的小切片，分别检测后再合并结果。

**实现细节**:
```python
class SAHIInference:
    def slice_image(self, image):
        # 将图像切分为640x640的重叠切片
        # 重叠比例: 20% (防止目标被切分)
        
    def merge_predictions(self, slice_predictions):
        # 使用NMS合并各切片的预测结果
        # 坐标转换回原始图像坐标系
```

**预期效果**: 在大图像上小目标检测AP提升 **6.8%-14.5%**

**使用方式**:
```bash
# 默认启用SAHI（图像大于1500px时自动触发）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt

# 禁用SAHI
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-sahi
```

---

### 2.2 多布局投票机制 (Multi-Layout Adjustment Voting)

**参考论文**:
- "Optimizing Text Recognition in Borehole Log Images Using a Multi-Layout Adjustment Voting Mechanism" (MDPI Applied Sciences, 2025)

**核心思想**:
OCR识别准确率受画布padding大小影响。通过生成不同padding比例的图像版本，分别识别后投票，可以显著提高准确率。

**实现细节**:
```python
class MultiLayoutVotingOCR:
    def recognize_with_voting(self, image):
        # 使用5种padding比例: [0.5x, 1.0x, 1.5x, 2.0x, 3.0x]
        # 对每种比例进行OCR识别
        # 按文本内容分组，选择出现次数最多且置信度最高的结果
        # 计算平均置信度作为最终置信度
```

**预期效果**: OCR F1-score提升 **1.6%-2.8%** (PaddleOCR)，对低质量图像提升更显著（可达10%）

**使用方式**:
```bash
# 默认启用投票（仅对技术要求区域）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt

# 禁用投票
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-voting
```

---

### 2.3 超分辨率预处理 + 笔画增强

**参考论文**:
- "Optimizing Text Recognition in Mechanical Drawings" (MDPI Machines, 2025) - eDOCr2
- "Extraction of Pharmaceutical Data from Medical Prescriptions Using OCR" (2024)

**核心思想**:
机械图纸中的文字通常笔画很细（<2.5像素），OCR难以识别。通过超分辨率放大 + 笔画膨胀增强，可以显著提高识别率。

**实现细节**:
```python
def upscale_image(image, scale=2.0):
    # 使用LANCZOS4插值（比双线性/双三次更好）
    
def stroke_enhancement(image, min_stroke_width=2.5):
    # 1. 距离变换估计平均笔画宽度
    # 2. 如果笔画太细，使用形态学膨胀增强
    # 3. 保持文字结构不变
```

**预期效果**: 在细笔画文字上CER降低 **10%->0%** (极端情况)

---

### 2.4 多尺度OCR融合 (Multi-Scale OCR Fusion)

**参考论文**:
- 综合多篇论文的预处理策略

**核心思想**:
对同一裁剪区域生成多个预处理版本（原始、1.5x放大、2x放大、笔画增强），分别进行OCR，选择平均置信度最高的结果。

**实现细节**:
```python
def multi_scale_preprocess(image, region_type):
    versions = []
    # 版本1: 标准预处理
    versions.append(("standard", preprocess_image_for_ocr(image, region_type)))
    # 版本2: 1.5x放大（适用于小图）
    # 版本3: 2x放大（适用于极小图）
    # 版本4: 笔画增强（适用于细线文字）
    return versions
```

**使用方式**:
```bash
# 默认启用多尺度
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt

# 禁用多尺度（更快但可能效果稍差）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-multi-scale
```

---

### 2.5 基于位置的文本聚类合并

**参考论文**:
- "eDOCr2: A Framework for Extracting Structured Information from Mechanical Drawings" (MDPI Machines, 2025)

**核心思想**:
OCR通常按字或词检测，导致一个完整句子被分割成多个片段。通过分析文本框的空间位置关系，将同一行的文本合并为完整句子。

**实现细节**:
```python
def cluster_text_boxes(text_items, x_threshold=0.3, y_threshold=0.5):
    # 1. 按y坐标排序
    # 2. 遍历文本框，判断是否与当前cluster在同一行
    #    - y坐标差异 < 平均高度 * 0.5
    #    - x间隔 < 平均宽度 * 3
    # 3. 合并同一cluster的文本，更新边界框
```

**预期效果**: 技术要求条目完整性显著提升，减少断句

---

### 2.6 倾斜校正 + 投影分析

**参考论文**:
- 多篇工业图纸OCR论文的共同策略

**核心思想**:
扫描的机械图纸可能存在轻微倾斜，影响OCR识别。通过最小外接矩形计算倾斜角度，进行旋转校正。

**实现细节**:
```python
def deskew_image(image):
    # 1. 二值化
    # 2. 查找文字轮廓
    # 3. 计算最小外接矩形角度
    # 4. 如果角度>0.5°，进行旋转校正
    # 5. 使用CUBIC插值 + 白色背景填充
```

---

### 2.7 扩展的纠错字典和验证机制

**改进内容**:
1. **扩展纠错字典**: 从原来的4个词条扩展到80+个词条，覆盖常见机械术语、材料牌号、OCR常见错误
2. **技术要求验证器**: 基于关键词库验证识别结果是否是有效的技术要求内容
3. **材料牌号自动修复**: ZL 102 -> ZL102, 5A06 F -> 5A06-F等

---

## 三、预处理流程对比

### 3.1 原始预处理流程
```
图像 -> 灰度转换 -> CLAHE增强 -> 锐化 -> 二值化 -> 去噪 -> BGR转换
```

### 3.2 改进后的预处理流程
```
图像 -> 超分辨率放大(条件触发) -> 倾斜校正 -> 灰度转换 -> 
      去噪(NLMeans) -> CLAHE增强(更强参数) -> 锐化 -> 
      自适应二值化 -> 形态学操作(开+闭) -> BGR转换
      
技术要求区域额外处理:
      -> 笔画增强(条件触发) -> 多布局投票OCR
```

---

## 四、使用方法

### 4.1 基本使用

```bash
# 单图像处理（全部增强功能启用）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt

# 批量处理
python integrate.py --mode batch --input pic_batch/ --model runs/detect_train/weights/best.pt

# 指定输出目录
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --output predict/my_result
```

### 4.2 高级参数

```bash
# 禁用SAHI（如果图像不大或SAHI效果不佳）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-sahi

# 禁用多尺度OCR（提高速度）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-multi-scale

# 禁用多布局投票（提高速度）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-voting

# 全部禁用（最快，但效果最差）
python integrate.py --mode single --input pic_batch/big0.png --model runs/detect_train/weights/best.pt --no-sahi --no-multi-scale --no-voting
```

### 4.3 输出文件说明

```
predict/20260123_143052_enhanced/
├── big0.txt                          # 结构化提取结果
├── big0_visualization.jpg            # 可视化检测结果
├── big0_result.json                  # JSON格式完整结果
├── crops/                            # 裁剪的检测区域
│   ├── tech_req_big0_1234_5678_original.jpg
│   └── table_big0_9012_3456_original.jpg
└── debug_preprocessed/               # 预处理后的调试图像
    ├── tech_req_big0_1234_5678_standard.jpg
    ├── tech_req_big0_1234_5678_upscale_1.5x.jpg
    └── tech_req_big0_1234_5678_stroke_enhanced.jpg
```

---

## 五、性能与效果预期

### 5.1 检测效果提升

| 改进项 | 原始效果 | 预期提升 | 依据 |
|-------|---------|---------|------|
| SAHI切片检测 | 大图上小目标漏检多 | AP +6.8%~14.5% | SAHI论文 |
| 多尺度输入 | 固定640px输入 | 适配不同尺寸图像 | YOLO最佳实践 |
| TTA增强 | 无 | 检测稳定性提升 | YOLO augment参数 |

### 5.2 OCR效果提升

| 改进项 | 原始效果 | 预期提升 | 依据 |
|-------|---------|---------|------|
| 超分辨率 | 原始分辨率 | 小字识别率提升 | PaddleOCR文档 |
| 多布局投票 | 单布局识别 | F1 +1.6%~2.8% | MDPI论文 |
| 笔画增强 | 细线字识别差 | CER 10%->0% | eDOCr2论文 |
| 倾斜校正 | 倾斜文字识别差 | 角度<0.5°不处理 | 工程实践 |
| 纠错字典 | 4个词条 | 80+个词条 | 领域知识 |

### 5.3 后处理效果提升

| 改进项 | 原始效果 | 预期提升 |
|-------|---------|---------|
| 文本聚类合并 | 按字/词断句 | 完整句子识别 |
| 技术要求验证 | 无验证 | 过滤低质量结果 |
| 表格字段提取 | 基础正则 | 扩展正则+回退策略 |

---

## 六、进一步优化建议

### 6.1 数据集层面
1. **增加训练数据**: 当前仅30张，建议至少100-200张
2. **数据增强**: 使用Mosaic、MixUp等增强
3. **SAHI切片微调**: 使用SAHI生成的切片进行额外训练

### 6.2 模型层面
1. **更换更大模型**: 从YOLOv8n升级到YOLOv8m/l
2. **专用检测头**: 为技术要求设计专用的小目标检测头
3. **PaddleOCR微调**: 在机械图纸数据集上微调识别模型

### 6.3 后处理层面
1. **LLM纠错**: 使用大语言模型对识别结果进行语义纠错
2. **模板匹配**: 对常见符号（如直径⌀、粗糙度符号）使用模板匹配
3. **多模型集成**: 同时使用PaddleOCR和EasyOCR，结果投票

---

## 七、参考论文列表

1. Akyon, F.C., et al. (2022). "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." IEEE.
2. MDPI Applied Sciences (2025). "Optimizing Text Recognition in Borehole Log Images Using a Multi-Layout Adjustment Voting Mechanism."
3. MDPI Machines (2025). "Optimizing Text Recognition in Mechanical Drawings" (eDOCr2).
4. PaddleOCR Technical Report (2025). "PaddleOCR 3.0 Technical Report."
5. SOAR (2024). "Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients."
6. ESOD (2024). "Efficient Small Object Detection on High-Resolution Images."
7. Shteriyanov, et al. "Artificial intelligence-based solution for the automatic extraction of pipeline metadata from P&IDs."
