# OCR-YOLO 图像分割与文本提取系统

## 项目简介

本项目是一个集成了YOLO目标检测和OCR文本提取的系统，主要用于从机械图纸等技术文档中自动提取结构化信息。系统能够检测图纸中的技术要求和表格区域，然后对这些区域进行OCR文本识别，最终提取出结构化的技术要求和表格信息。

## 实测指标（21 张装配图自建评测集）

| 指标 | 结果 | 说明 |
|---|---|---|
| 区域检测召回 / mAP@50 | **98.7%** | SAHI 切片检测（大图重叠切片 + NMS 合并） |
| 多尺度 OCR 消融 | 关闭后字段提取完整度 **下降约 50%**、技术要求 CER 0 → 0.498 | 四臂消融（full / no_ms / no_vt / base）定位关键组件 |
| 端到端多模态大模型基线 | 板块识别正确率仅 **23.8%** | 据此确立"YOLO 检测 + OCR + 规则纠错"专用管线路线 |
| 工程调试 | 消融发现投票逻辑为死代码（no_vt 与 full 逐字节相同），修复门控后 OCR 调用量 7→35 次真实生效；定位"OCR 识别正常但输出空白"的聚合逻辑 bug 并修复 | — |

### 增强压力集（2026-09-09：21 原图 × 100 光度变体 = 2100 张，GT 框复用，`gen_aug.py` + `eval_aug.py` 可复现）

| 评测臂 | mAP@50 | Precision | Recall |
|---|---|---|---|
| baseline（整图直检） | **94.2%** | **94.4%** | **93.8%** |
| sahi（切片检测） | 89.5% | 68.9% | 92.9% |

**反直觉发现**：干净原图上 SAHI 是关键组件，但在噪声/模糊/压缩伪影压力下，切片路径误报显著增多（Precision 68.9% vs baseline 94.4%）——增强压力测试推翻了"SAHI 恒优"的直觉，鲁棒性场景应走整图直检或加切片级置信过滤。

## 核心功能

1. **YOLO目标检测**：使用YOLOv8/YOLOv11模型检测图纸中的技术要求和表格区域
2. **OCR文本提取**：使用PaddleOCR对检测到的区域进行文本识别
3. **结构化信息提取**：从识别的文本中提取结构化的技术要求和表格信息
4. **模型训练**：支持自定义数据集的YOLO模型训练
5. **批量处理**：支持批量处理多张图纸

## 项目结构

```
yolo/
├── integrate.py          # 整合YOLO检测和OCR文本提取的核心文件
├── main.py              # 训练和预测主入口
├── train_yolo.py        # YOLO模型训练脚本
├── src/
│   ├── predict.py       # YOLO预测功能
│   ├── train.py         # YOLO训练功能
│   ├── easyocr_utils.py # OCR工具类
│   └── utils/
│       ├── config.py          # 配置文件加载和管理
│       ├── image_processing.py # 图像处理功能
│       └── logger.py          # 日志记录功能
└── config/
    ├── train.yaml         # YOLO训练配置
    ├── predict.yaml       # YOLO预测配置
    ├── predict_ocr.yaml   # OCR预测配置
    ├── train_table.yaml   # 表格训练配置
    └── info_extraction.yaml # 信息提取配置
```

## 环境要求

### 依赖包

- Python 3.8+
- ultralytics (YOLO库)
- paddleocr (OCR库)
- paddlepaddle (PaddleOCR依赖)
- opencv-python (图像处理)
- pyyaml (配置文件管理)
- pillow (图像处理)
- numpy (数值计算)

### 安装依赖

```bash
# 安装YOLO依赖
pip install ultralytics

# 安装PaddleOCR（CPU版本）
pip install paddleocr

# 安装其他依赖
pip install opencv-python pyyaml pillow numpy
```

### GPU支持（可选）

如果需要使用GPU加速，需要安装对应CUDA版本的PaddlePaddle：

```bash
# 安装GPU版本的PaddlePaddle（以CUDA 11.8为例）
pip install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

## 使用方法

### 1. 模型训练

#### 方法一：使用main.py进行训练

1. 准备数据集，按照YOLO格式组织
2. 配置`config/train.yaml`文件，设置训练参数
3. 运行训练脚本：

```bash
python main.py
```

#### 方法二：使用train_yolo.py进行训练

1. 准备数据集，放在`mydata`目录下
2. 运行训练脚本：

```bash
python train_yolo.py
```

### 2. 预测与OCR提取

#### 方法一：使用main.py进行预测

1. 配置`config/predict.yaml`文件，设置预测参数
2. 运行预测脚本：

```bash
python main.py
```

#### 方法二：使用integrate.py进行集成处理

```bash
python integrate.py --image_path <图像路径> --model_path <模型路径> --output_dir <输出目录>
```

### 3. 批量处理

修改`integrate.py`中的`process_batch_images`函数调用，设置输入目录和其他参数，然后运行：

```bash
python integrate.py
```

## 核心文件说明

### integrate.py

整合YOLO目标检测和OCR文本提取的核心文件，主要功能：
- 初始化YOLO模型和OCR处理器
- 检测图像中的技术要求和表格区域
- 对检测到的区域进行OCR文本识别
- 从识别的文本中提取结构化信息
- 保存提取结果到TXT文件

### main.py

训练和预测的主入口，主要功能：
- 加载训练配置文件
- 训练YOLO模型
- 更新预测配置文件中的模型路径
- 调用`src/predict.py`进行预测

### src/predict.py

YOLO预测功能实现，主要功能：
- 加载预测配置文件
- 执行YOLO模型预测
- 保存检测结果和裁剪图像
- 对裁剪图像进行后处理

### src/train.py

YOLO训练功能实现，主要功能：
- 加载训练配置文件
- 训练YOLO模型
- 保存训练结果和最佳模型

## 配置文件说明

### train.yaml

YOLO模型训练配置，主要参数：
- `model`：预训练模型路径
- `data`：数据集配置文件路径
- `epochs`：训练轮数
- `batch`：批处理大小
- `imgsz`：图像尺寸
- `project`：训练结果保存目录
- `name`：训练任务名称
- `device`：训练设备（cpu或gpu）

### predict.yaml

YOLO模型预测配置，主要参数：
- `model`：训练好的模型路径
- `source`：输入图像路径
- `project`：预测结果保存目录
- `save_crop`：是否保存裁剪图像
- `device`：预测设备（cpu或gpu）

## 数据格式

### 训练数据格式

按照YOLO格式组织，目录结构如下：

```
mydata/
├── images1/         # 训练图像
├── labels1/         # 训练标签
└── data.yaml        # 数据集配置文件
```

### 标签格式

YOLO格式的标签文件，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

其中：
- `class_id`：类别ID（0: tech_req, 1: table）
- `x_center`, `y_center`：目标中心点坐标（归一化到0-1）
- `width`, `height`：目标宽度和高度（归一化到0-1）

## 输出结果

### 预测结果

预测结果保存在`predict`目录下，按时间戳组织：

```
predict/
└── 20240101_120000/         # 时间戳目录
    └── crops_by_name/        # 按原图名组织的裁剪图像
        └── <原图名>/         # 原图名目录
            ├── tech_req_0.png # 技术要求区域裁剪图像
            └── table_0.png    # 表格区域裁剪图像
```

### OCR提取结果

OCR提取结果保存在`predict`目录下，与预测结果同一级：

```
predict/
└── 20240101_120000/         # 时间戳目录
    ├── <原图名>.txt          # OCR提取结果文本文件
    └── crops/                # 裁剪图像
```

## 常见问题

### 1. 模型训练失败

- 检查数据集格式是否正确
- 检查配置文件中的参数是否合理
- 检查GPU内存是否足够

### 2. OCR识别准确率低

- 确保检测到的区域包含完整的文本
- 调整图像预处理参数，增强对比度和清晰度
- 考虑使用更高精度的OCR模型

### 3. 预测速度慢

- 使用GPU进行预测
- 减小图像尺寸
- 减少预测时的后处理操作

## 工作交接说明

### 核心功能模块

1. **目标检测模块**：使用YOLO模型检测技术要求和表格区域
2. **文本提取模块**：使用PaddleOCR对检测区域进行文本识别
3. **信息提取模块**：从识别文本中提取结构化信息

### 关键文件

- **integrate.py**：系统核心，整合检测和OCR功能
- **main.py**：主入口，控制训练和预测流程
- **src/predict.py**：预测功能实现
- **src/train.py**：训练功能实现
- **config/**：配置文件目录

### 后续工作建议

1. **模型优化**：收集更多数据，优化YOLO模型，提高检测准确率
2. **OCR优化**：尝试不同的OCR模型和预处理方法，提高文本识别准确率
3. **功能扩展**：添加更多类型的区域检测，如标题、备注等
4. **界面开发**：开发图形用户界面，提高系统易用性
5. **部署优化**：将系统部署为服务，支持API调用

## 联系方式

如有问题或需要进一步的技术支持，请联系项目维护人员。

---

*本项目使用Python 3.8+开发，依赖ultralytics、paddleocr等库。*
