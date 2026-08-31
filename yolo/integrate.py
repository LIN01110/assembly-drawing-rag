#!/usr/bin/env python3
"""
integrate.py - 整合YOLO检测和OCR文本提取的预测脚本（增强版）
支持单图像和批量图像预测，结果输出到predict文件夹

================================================================================
改进策略（基于最新论文研究）：
================================================================================
1. SAHI (Slicing Aided Hyper Inference) - 切片辅助超推理
   论文: "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"
   解决: 大图像中小目标检测问题，对大图纸特别有效
   
2. 多布局投票机制 (Multi-Layout Adjustment Voting)
   论文: "Optimizing Text Recognition in Borehole Log Images Using a Multi-Layout 
         Adjustment Voting Mechanism" (MDPI Applied Sciences 2025)
   解决: 通过调整画布padding并投票，提高OCR准确率1.6-2.8%
   
3. 超分辨率预处理 + 笔画增强
   论文: "Optimizing Text Recognition in Mechanical Drawings" (MDPI Machines 2025)
   解决: 针对机械图纸细线文字，使用笔画膨胀增强
   
4. 多尺度OCR融合 (Multi-Scale OCR Fusion)
   论文: 综合多篇论文的预处理策略
   解决: 对同一区域用不同分辨率/预处理方式识别，取最优结果
   
5. 基于位置的文本聚类合并
   论文: "eDOCr2: A Framework for Extracting Structured Information from 
         Mechanical Drawings"
   解决: 将同一行/列的文本框聚类合并，避免断句
   
6. 倾斜校正 + 投影分析
   论文: 多篇工业图纸OCR论文的共同策略
   解决: 机械图纸扫描时可能存在的倾斜问题
================================================================================
"""

import cv2, json, re, logging, time, sys, os
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

# 设置环境变量，避免连接检查
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========== 扩展的OCR纠错字典 ==========
OCR_CORRECTION_DICT = {
    # 技术要求相关
    "找木典": "技术要求", "找术典": "技术要求", "找术要": "技术要求",
    "技木要求": "技术要求", "找求": "要求", "木典": "要求",
    "技求": "技术要求", "找术": "技术",
    # 机械零件名称
    "阀休": "阀体", "阔体": "阀体", "阔休": "阀体",
    "盒休": "盒体", "盖休": "盖体", "座休": "座体",
    # 材料牌号
    "ZL 102": "ZL102", "ZLIO2": "ZL102", "ZLI02": "ZL102",
    "5A06 F": "5A06-F", "5A06F": "5A06-F", "5AO6": "5A06",
    "45#": "45#钢", "45 #": "45#钢",
    "Q235": "Q235", "Q 235": "Q235",
    "HT200": "HT200", "HT 200": "HT200",
    # 常见OCR错误（数字字母混淆）
    "O0": "0", "0O": "0", "l1": "1", "1l": "1",
    "I1": "1", "1I": "1", "S5": "5", "5S": "5",
    "B8": "8", "8B": "8", "Z2": "2", "2Z": "2",
    # 单位相关
    "m m": "mm", "M M": "MM", "u m": "μm",
    # 技术要求关键词
    "未注": "未注", "倒角": "倒角", "锐边": "锐边",
    "去毛刺": "去毛刺", "毛刺": "毛刺",
    "热处理": "热处理", "调质": "调质", "淬火": "淬火",
    "渗碳": "渗碳", "氮化": "氮化", "氧化": "氧化",
    "镀锌": "镀锌", "镀铬": "镀铬", "发黑": "发黑",
    "喷塑": "喷塑", "喷漆": "喷漆", "涂漆": "涂漆",
    "表面处理": "表面处理", "表面": "表面",
    "硬度": "硬度", "粗糙度": "粗糙度", "公差": "公差",
    "精度": "精度", "铸造": "铸造", "锻造": "锻造",
    "焊接": "焊接", "装配": "装配", "探伤": "探伤",
    "检测": "检测", "检验": "检验", "验收": "验收",
    "退火": "退火", "正火": "正火", "回火": "回火",
    "时效": "时效", "喷砂": "喷砂", "抛丸": "抛丸",
    "打磨": "打磨", "抛光": "抛光", "涂油": "涂油",
    "防锈": "防锈", "防腐": "防腐",
}


# 技术要求关键词库（用于验证和纠错）
TECH_REQ_KEYWORDS = [
    "技术要求", "未注", "倒角", "锐边", "去毛刺", "毛刺",
    "热处理", "调质", "淬火", "渗碳", "氮化", "氧化",
    "镀锌", "镀铬", "发黑", "喷塑", "喷漆", "表面处理",
    "硬度", "HRC", "HB", "HV", "粗糙度", "Ra", "Rz",
    "公差", "精度", "级", "铸造", "锻造", "焊接", "装配",
    "探伤", "检测", "检验", "验收", "比例", "缩放", "放大", "缩小",
    "锐角", "圆角", "圆弧", "半径", "R", "退火", "正火", "回火",
    "时效", "喷砂", "抛丸", "打磨", "抛光", "涂漆", "涂油",
    "防锈", "防腐", "尺寸", "形位", "形位公差", "同轴度",
    "平行度", "垂直度", "对称度", "跳动", "圆度", "圆柱度",
    "平面度", "直线度", "轮廓度", "位置度", "倾斜度",
]


# 表格字段关键词
TABLE_FIELD_KEYWORDS = {
    "物品名称": ["名称", "零件名称", "部件名称", "图名", "物品", "零件"],
    "材料": ["材料", "材质", "牌号", "钢号", "料号"],
    "数量": ["数量", "件数", "个数", "批量"],
    "图号": ["图号", "编号", "件号", "代号", "图样代号"],
    "张数": ["张数", "页数", "共", "第"],
    "重量": ["重量", "质量", "单重", "总重"],
    "比例": ["比例", "SCALE", "scale"],
    "设计": ["设计", "制图", "绘图"],
    "审核": ["审核", "校对", "审查"],
    "工艺": ["工艺", "工艺审核"],
    "标准化": ["标准化", "标准"],
    "批准": ["批准", "审定", "审批"],
}


def correct_ocr_text(text: str) -> str:
    """OCR文本纠错 - 增强版"""
    if not text:
        return text
    
    original = text
    
    # 1. 应用纠错字典
    for k, v in OCR_CORRECTION_DICT.items():
        text = text.replace(k, v)
    
    # 2. 清理多余空格（保留中文字符间的空格）
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. 修复常见数字字母混淆（在特定上下文中）
    text = re.sub(r'ZL\s*(\d+)', r'ZL\1', text)
    text = re.sub(r'(\d+)\s*#', r'\1#', text)
    text = re.sub(r'5A\s*(\d+)', r'5A\1', text)
    text = re.sub(r'HT\s*(\d+)', r'HT\1', text)
    text = re.sub(r'Q\s*(\d+)', r'Q\1', text)
    
    # 4. 修复粗糙度符号
    text = re.sub(r'Ra\s*(\d+\.?\d*)', r'Ra\1', text)
    
    # 5. 修复硬度单位
    text = re.sub(r'(\d+)\s*HRC', r'\1HRC', text)
    text = re.sub(r'(\d+)\s*HB', r'\1HB', text)
    text = re.sub(r'(\d+)\s*HV', r'\1HV', text)
    
    # 6. 修复单位
    text = re.sub(r'(\d+)\s*mm', r'\1mm', text)
    text = re.sub(r'(\d+)\s*°', r'\1°', text)
    
    # 7. 如果文本变化了，记录日志
    if text != original:
        logger.debug(f"OCR纠错: '{original}' -> '{text}'")
    
    return text


def validate_tech_req_text(text: str) -> Tuple[bool, float]:
    """验证文本是否可能是技术要求内容"""
    if not text or len(text) < 2:
        return False, 0.0
    
    score = 0.0
    
    # 检查是否包含技术要求关键词
    for keyword in TECH_REQ_KEYWORDS:
        if keyword in text:
            score += 0.3
    
    # 检查是否包含数字
    if re.search(r'\d', text):
        score += 0.1
    
    # 检查是否包含单位
    units = ['mm', 'cm', 'μm', 'Ra', 'HRC', 'HB', 'HV', '°', '%']
    for unit in units:
        if unit in text:
            score += 0.2
    
    # 检查是否是有效的中文文本
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    if chinese_chars > 0:
        score += min(chinese_chars * 0.05, 0.3)
    
    # 检查长度合理性
    if 2 <= len(text) <= 100:
        score += 0.1
    
    is_valid = score >= 0.2
    return is_valid, min(score, 1.0)


# ==============================================================================
# SAHI (Slicing Aided Hyper Inference) 实现
# 论文: "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"
# ==============================================================================
class SAHIInference:
    """切片辅助超推理 - 专门解决大图像中小目标检测问题"""
    
    def __init__(self, slice_height: int = 640, slice_width: int = 640,
                 overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
    
    def slice_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """将图像切分为重叠的切片"""
        h, w = image.shape[:2]
        slices = []
        
        # 计算步长（考虑重叠）
        step_h = int(self.slice_height * (1 - self.overlap_height_ratio))
        step_w = int(self.slice_width * (1 - self.overlap_width_ratio))
        
        # 生成切片
        y_positions = list(range(0, h - self.slice_height + 1, step_h))
        if not y_positions or y_positions[-1] + self.slice_height < h:
            y_positions.append(h - self.slice_height)
        
        x_positions = list(range(0, w - self.slice_width + 1, step_w))
        if not x_positions or x_positions[-1] + self.slice_width < w:
            x_positions.append(w - self.slice_width)
        
        for y in y_positions:
            for x in x_positions:
                # 确保不越界
                y_end = min(y + self.slice_height, h)
                x_end = min(x + self.slice_width, w)
                y_start = max(0, y_end - self.slice_height)
                x_start = max(0, x_end - self.slice_width)
                
                slice_img = image[y_start:y_end, x_start:x_end]
                
                slices.append({
                    'image': slice_img,
                    'x_offset': x_start,
                    'y_offset': y_start,
                    'width': x_end - x_start,
                    'height': y_end - y_start,
                })
        
        logger.info(f"SAHI切片: 图像 {w}x{h} 被切分为 {len(slices)} 个切片")
        return slices
    
    def merge_predictions(self, slice_predictions: List[List[Dict]], 
                          iou_threshold: float = 0.5) -> List[Dict]:
        """合并各切片的预测结果"""
        all_boxes = []
        
        for slice_idx, predictions in enumerate(slice_predictions):
            for pred in predictions:
                all_boxes.append(pred)
        
        if not all_boxes:
            return []
        
        # 使用NMS合并重叠框
        return self._nms(all_boxes, iou_threshold)
    
    def _nms(self, boxes: List[Dict], iou_threshold: float) -> List[Dict]:
        """非极大值抑制"""
        if not boxes:
            return []
        
        # 按置信度排序
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while boxes:
            current = boxes[0]
            keep.append(current)
            
            remaining = []
            for box in boxes[1:]:
                iou = self._compute_iou(current['bbox'], box['bbox'])
                if iou < iou_threshold:
                    remaining.append(box)
            
            boxes = remaining
        
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


# ==============================================================================
# 多布局投票机制 (Multi-Layout Adjustment Voting)
# 论文: "Optimizing Text Recognition in Borehole Log Images Using a 
#       Multi-Layout Adjustment Voting Mechanism"
#
# 历史 bug（2026-08 消融发现）：投票原先只对 'tech_req' 类生效，
# 而当前检测模型是单类 'panel'，导致投票为死代码（no_vt 与 full 臂逐字节相同）。
# 修复：门控改为类别集合 VOTING_CLASSES，默认包含 panel。
VOTING_CLASSES = {'tech_req', 'panel'}
# ==============================================================================
class MultiLayoutVotingOCR:
    """多布局投票OCR - 通过调整画布padding并投票提高识别率"""
    
    def __init__(self, base_ocr_processor):
        self.ocr = base_ocr_processor
        # 不同的padding比例
        self.padding_ratios = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    def recognize_with_voting(self, image: np.ndarray, 
                              region_type: str = 'general') -> List[Tuple[float, str, List]]:
        """使用多布局投票进行OCR识别"""
        if image.size == 0:
            return []
        logger.info(f"投票OCR执行: region_type={region_type}, paddings={self.padding_ratios}")
        
        h, w = image.shape[:2]
        all_results = []
        
        # 对每个padding比例进行识别
        for ratio in self.padding_ratios:
            padded = self._add_padding(image, ratio)
            results = self.ocr.recognize_text(padded)
            
            for conf, text, pos in results:
                all_results.append({
                    'confidence': conf,
                    'text': text,
                    'position': pos,
                    'padding_ratio': ratio
                })
        
        # 投票合并结果
        merged = self._vote_results(all_results)
        
        return [(r['confidence'], r['text'], r['position']) for r in merged]
    
    def _add_padding(self, image: np.ndarray, ratio: float) -> np.ndarray:
        """添加padding"""
        h, w = image.shape[:2]
        pad_h = int(h * ratio)
        pad_w = int(w * ratio)
        
        if len(image.shape) == 3:
            padded = np.full((h + 2 * pad_h, w + 2 * pad_w, 3), 255, dtype=np.uint8)
            padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        else:
            padded = np.full((h + 2 * pad_h, w + 2 * pad_w), 255, dtype=np.uint8)
            padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        
        return padded
    
    def _vote_results(self, results: List[Dict]) -> List[Dict]:
        """对多布局结果进行投票"""
        if not results:
            return []
        
        # 按文本内容分组
        text_groups = defaultdict(list)
        for r in results:
            # 标准化文本用于比较
            normalized = r['text'].replace(' ', '').lower()
            text_groups[normalized].append(r)
        
        # 选择每组中置信度最高的结果
        merged = []
        for normalized, group in text_groups.items():
            # 选择出现次数最多且置信度最高的
            best = max(group, key=lambda x: (len(group), x['confidence']))
            
            # 计算平均置信度
            avg_conf = np.mean([g['confidence'] for g in group])
            best['confidence'] = avg_conf
            
            merged.append(best)
        
        return merged


class EasyOCRProcessor:
    """PaddleOCR处理器 - 增强版"""
    
    def __init__(self, lang_list=['ch', 'en']):
        from paddleocr import PaddleOCR

        # 离线模型目录
        model_dir = "/workspace/.paddleocr/whl"
        if not os.path.exists(model_dir):
            model_dir = os.path.expanduser("~/.paddleocr/whl")
        
        # 使用更好的OCR配置
        det_model_dir = f"{model_dir}/ch_PP-OCRv4_det_infer" if os.path.exists(f"{model_dir}/ch_PP-OCRv4_det_infer") else None
        rec_model_dir = f"{model_dir}/ch_PP-OCRv4_rec_infer" if os.path.exists(f"{model_dir}/ch_PP-OCRv4_rec_infer") else None
        cls_model_dir = f"{model_dir}/ch_ppocr_mobile_v2.0_cls_infer" if os.path.exists(f"{model_dir}/ch_ppocr_mobile_v2.0_cls_infer") else None
        
        self.reader = PaddleOCR(
            use_angle_cls=True,  # 开启角度分类
            lang="ch",
            use_gpu=False,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
            max_text_length=100,
            drop_score=0.3,
        )

    def recognize_text(self, image: np.ndarray, **kwargs) -> List[Tuple[float, str, List]]:
        """识别图像中的文本"""
        try:
            result = self.reader.ocr(image)
            if result and result[0]:
                ocr_results = []
                for line in result[0]:
                    if line and len(line) >= 2 and line[1]:
                        text = line[1][0]
                        confidence = line[1][1] if len(line[1]) > 1 else 0.0
                        position = line[0] if line[0] else [[0, 0], [0, 0], [0, 0], [0, 0]]
                        corrected_text = correct_ocr_text(text)
                        ocr_results.append((confidence, corrected_text, position))
                return ocr_results
            return []
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return []


class YOLOProcessor:
    """YOLO目标检测处理器 - 增强版（集成SAHI）"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.sahi = SAHIInference(
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        logger.info(f"加载YOLO模型: {model_path}")

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.15,
                       iou_threshold: float = 0.45, use_sahi: bool = True) -> List:
        """检测图像中的对象 - 增强版"""
        try:
            h, w = image.shape[:2]
            
            # 判断是否需要使用SAHI
            if use_sahi and max(h, w) > 1500:
                return self._detect_with_sahi(image, conf_threshold, iou_threshold)
            else:
                return self._detect_standard(image, conf_threshold, iou_threshold)
                
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return []
    
    def _detect_standard(self, image: np.ndarray, conf_threshold: float,
                         iou_threshold: float) -> List:
        """标准检测"""
        h, w = image.shape[:2]
        
        # 根据图像大小选择合适的输入尺寸
        if max(h, w) > 2000:
            imgsz = 1280
        elif max(h, w) > 1000:
            imgsz = 960
        else:
            imgsz = 640
        
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            augment=True,
            agnostic_nms=True,
            max_det=100,
        )
        
        total_boxes = sum(len(r.boxes) for r in results)
        logger.info(f"标准检测: 检测到 {total_boxes} 个对象 (imgsz={imgsz})")
        
        # 如果没有检测到技术要求，降低阈值再试
        if total_boxes > 0:
            has_tech_req = any(
                any(r.names[int(box.cls[0])] == 'tech_req' for box in r.boxes)
                for r in results if hasattr(r, 'boxes') and r.boxes is not None
            )
            
            if not has_tech_req:
                logger.warning("未检测到技术要求区域，尝试降低阈值重新检测")
                results = self.model(
                    image,
                    conf=conf_threshold * 0.5,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    verbose=False,
                    augment=True,
                    agnostic_nms=True,
                    max_det=100,
                )
        
        return results
    
    def _detect_with_sahi(self, image: np.ndarray, conf_threshold: float,
                          iou_threshold: float) -> List:
        """使用SAHI进行切片检测"""
        logger.info("使用SAHI切片检测")
        
        # 切片图像
        slices = self.sahi.slice_image(image)
        
        all_predictions = []
        
        for slice_info in slices:
            slice_img = slice_info['image']
            x_offset = slice_info['x_offset']
            y_offset = slice_info['y_offset']
            
            # 对切片进行检测
            results = self.model(
                slice_img,
                conf=conf_threshold * 0.7,  # SAHI使用稍低的阈值
                iou=iou_threshold,
                imgsz=640,
                verbose=False,
                augment=True,
                agnostic_nms=True,
                max_det=50,
            )
            
            # 转换预测结果到原图坐标
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 转换到原图坐标
                    x1 += x_offset
                    y1 += y_offset
                    x2 += x_offset
                    y2 += y_offset
                    
                    all_predictions.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': cls_id,
                        'confidence': conf,
                        'class_name': result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                    })
        
        # 合并预测结果
        merged = self.sahi.merge_predictions([all_predictions], iou_threshold)
        logger.info(f"SAHI检测完成: 合并后 {len(merged)} 个对象")
        
        # 将合并结果转换回YOLO结果格式
        return self._convert_to_yolo_results(image, merged)
    
    def _convert_to_yolo_results(self, image: np.ndarray, 
                                  predictions: List[Dict]) -> List:
        """将SAHI结果转换为YOLO结果格式"""
        # 创建一个模拟的YOLO结果对象
        class MockResult:
            def __init__(self, image_shape):
                self.boxes = None
                self.names = {}
                self.path = ""
                self.orig_shape = image_shape
        
        if not predictions:
            return [MockResult(image.shape)]
        
        # 按类别分组
        class_groups = defaultdict(list)
        for pred in predictions:
            class_groups[pred['class_id']].append(pred)
        
        # 创建结果
        result = MockResult(image.shape)
        result.names = {pred['class_id']: pred['class_name'] for pred in predictions}
        
        # 创建boxes（简化版）
        import torch
        boxes_data = []
        for pred in predictions:
            boxes_data.append([
                pred['bbox'][0], pred['bbox'][1], 
                pred['bbox'][2], pred['bbox'][3],
                pred['confidence'], pred['class_id']
            ])
        
        if boxes_data:
            boxes_tensor = torch.tensor(boxes_data)
            # 创建简化版的boxes对象（与 ultralytics Boxes 消费方式兼容：
            # 支持 len()、迭代，逐框暴露 .xyxy[0]/.conf[0]/.cls[0]）
            class _SimpleBox:
                def __init__(self, xyxy, conf, cls):
                    self.xyxy = xyxy
                    self.conf = conf
                    self.cls = cls

            class SimpleBoxes:
                def __init__(self, data):
                    self.xyxy = data[:, :4]
                    self.conf = data[:, 4]
                    self.cls = data[:, 5].long()

                def __len__(self):
                    return self.xyxy.shape[0]

                def __iter__(self):
                    for i in range(len(self)):
                        yield _SimpleBox(
                            self.xyxy[i:i + 1], self.conf[i:i + 1], self.cls[i:i + 1]
                        )

            result.boxes = SimpleBoxes(boxes_tensor)
        
        return [result]


# ==============================================================================
# 图像预处理增强
# ==============================================================================
def upscale_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """使用LANCZOS4插值放大图像"""
    if image.size == 0:
        return image
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """图像倾斜校正"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 查找轮廓
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return image
    
    # 计算最小外接矩形角度
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    if abs(angle) < 0.5:
        return image
    
    # 旋转校正
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    
    logger.debug(f"倾斜校正: 角度={angle:.2f}°")
    return rotated


def stroke_enhancement(image: np.ndarray, min_stroke_width: float = 2.5) -> np.ndarray:
    """
    笔画增强 - 针对机械图纸细线文字
    论文: "Optimizing Text Recognition in Mechanical Drawings" (eDOCr2)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算平均笔画宽度（简化版）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 距离变换估计笔画宽度
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    mean_stroke = np.mean(dist_transform[dist_transform > 0]) * 2
    
    logger.debug(f"平均笔画宽度: {mean_stroke:.2f}px")
    
    # 如果笔画太细，进行膨胀增强
    if mean_stroke < min_stroke_width:
        kernel_size = max(1, int((min_stroke_width - mean_stroke) / 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        enhanced = cv2.dilate(binary, kernel, iterations=1)
        
        # 转回白色背景
        enhanced = cv2.bitwise_not(enhanced)
        
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        logger.debug(f"笔画增强: 使用 {kernel_size}x{kernel_size} 核膨胀")
        return enhanced
    
    return image


def multi_scale_preprocess(image: np.ndarray, region_type: str) -> List[np.ndarray]:
    """
    多尺度预处理 - 生成不同预处理版本的图像
    用于多尺度OCR融合
    """
    versions = []
    
    # 版本1: 原始尺寸 + 标准预处理
    v1 = preprocess_image_for_ocr(image, region_type)
    versions.append(("standard", v1))
    
    # 版本2: 放大1.5倍
    if max(image.shape[:2]) < 500:
        upscaled = upscale_image(image, 1.5)
        v2 = preprocess_image_for_ocr(upscaled, region_type)
        versions.append(("upscale_1.5x", v2))
    
    # 版本3: 放大2倍
    if max(image.shape[:2]) < 400:
        upscaled = upscale_image(image, 2.0)
        v3 = preprocess_image_for_ocr(upscaled, region_type)
        versions.append(("upscale_2x", v3))
    
    # 版本4: 笔画增强
    enhanced = stroke_enhancement(image)
    if not np.array_equal(enhanced, image):
        v4 = preprocess_image_for_ocr(enhanced, region_type)
        versions.append(("stroke_enhanced", v4))
    
    return versions


def preprocess_image_for_ocr(image: np.ndarray, region_type: str) -> np.ndarray:
    """根据区域类型预处理图像 - 增强版"""
    if image.size == 0:
        return image
    
    # 1. 放大图像
    if max(image.shape[:2]) < 200:
        image = upscale_image(image, scale=2)
    elif max(image.shape[:2]) < 400:
        image = upscale_image(image, scale=1.5)
    
    # 2. 倾斜校正
    image = deskew_image(image)
    
    # 3. 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 4. 根据区域类型应用不同的预处理
    if region_type == 'tech_req':
        # 技术要求区域：专门优化
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        
    elif region_type in ['table', 'cell', 'col', 'row']:
        denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        result = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
    else:
        denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return result


# ==============================================================================
# 文本聚类合并
# 论文: "eDOCr2: A Framework for Extracting Structured Information from 
#       Mechanical Drawings"
# ==============================================================================
def cluster_text_boxes(text_items: List[Dict], 
                       x_threshold: float = 0.3,
                       y_threshold: float = 0.5) -> List[Dict]:
    """
    基于位置的文本框聚类合并
    将同一行/列的文本合并为完整句子
    """
    if not text_items:
        return []
    
    # 按y坐标排序
    sorted_items = sorted(text_items, key=lambda x: x['sort_key'])
    
    clusters = []
    current_cluster = [sorted_items[0]]
    
    for item in sorted_items[1:]:
        last_item = current_cluster[-1]
        
        # 计算y坐标差异（相对于平均高度）
        avg_height = np.mean([
            item['bbox'][3] - item['bbox'][1],
            last_item['bbox'][3] - last_item['bbox'][1]
        ])
        y_diff = abs(item['sort_key'][0] - last_item['sort_key'][0])
        
        # 计算x坐标差异
        x_gap = item['bbox'][0] - last_item['bbox'][2]
        avg_width = np.mean([
            item['bbox'][2] - item['bbox'][0],
            last_item['bbox'][2] - last_item['bbox'][0]
        ])
        
        # 判断是否在同一个cluster中
        same_line = y_diff < avg_height * y_threshold
        reasonable_gap = x_gap < avg_width * 3  # 允许3倍字宽的间隔
        
        if same_line and reasonable_gap:
            current_cluster.append(item)
        else:
            # 保存当前cluster
            merged = merge_cluster(current_cluster)
            clusters.append(merged)
            current_cluster = [item]
    
    # 处理最后一个cluster
    if current_cluster:
        merged = merge_cluster(current_cluster)
        clusters.append(merged)
    
    return clusters


def merge_cluster(cluster: List[Dict]) -> Dict:
    """合并一个cluster中的文本"""
    if len(cluster) == 1:
        return cluster[0]
    
    # 按x坐标排序
    cluster = sorted(cluster, key=lambda x: x['sort_key'][1])
    
    # 合并文本
    texts = [item['text'] for item in cluster]
    merged_text = ' '.join(texts)
    
    # 计算合并后的边界框
    x1 = min(item['bbox'][0] for item in cluster)
    y1 = min(item['bbox'][1] for item in cluster)
    x2 = max(item['bbox'][2] for item in cluster)
    y2 = max(item['bbox'][3] for item in cluster)
    
    # 计算平均置信度
    avg_conf = np.mean([item['confidence'] for item in cluster])
    
    return {
        'text': merged_text,
        'class': cluster[0]['class'],
        'confidence': avg_conf,
        'position': cluster[0]['position'],
        'sort_key': (cluster[0]['sort_key'][0], cluster[0]['sort_key'][1]),
        'bbox': [x1, y1, x2, y2]
    }


def extract_text_from_regions(detected_objects, image: np.ndarray, 
                              ocr_processor, process_id: str = None,
                              output_dir: Path = None,
                              save_crops: bool = True,
                              save_debug: bool = True,
                              use_multi_scale: bool = True,
                              use_voting: bool = True) -> List[Dict]:
    """从检测区域提取文本 - 增强版"""
    all_text = []
    crop_count = 0
    debug_count = 0
    
    # 创建多布局投票OCR
    voting_ocr = MultiLayoutVotingOCR(ocr_processor) if use_voting else None

    for result_idx, result in enumerate(detected_objects):
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue

        for box_idx, box in enumerate(result.boxes):
            try:
                # 获取边界框信息
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                
                # 针对不同类别使用不同的置信度阈值
                min_conf = {
                    'tech_req': 0.08,
                    'table': 0.15,
                    'cell': 0.15,
                    'col': 0.15,
                    'row': 0.15,
                }.get(class_name, 0.15)
                
                if conf < min_conf:
                    continue

                # 裁剪区域（添加边距）
                margin = 10 if class_name == 'tech_req' else 5
                h, w = image.shape[:2]
                x1_pad = max(0, x1 - margin)
                y1_pad = max(0, y1 - margin)
                x2_pad = min(w, x2 + margin)
                y2_pad = min(h, y2 + margin)
                
                crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size == 0:
                    logger.warning(f"裁剪区域为空: {class_name} at ({x1},{y1},{x2},{y2})")
                    continue

                # 保存原始裁剪图
                prefix = f"{class_name}_{process_id}_{x1}_{y1}" if process_id else f"{class_name}_{x1}_{y1}"
                
                if save_crops and output_dir:
                    try:
                        crops_dir = Path(output_dir) / "crops"
                        crops_dir.mkdir(exist_ok=True, parents=True)
                        crop_path = crops_dir / f"{prefix}_original.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        crop_count += 1
                    except Exception as e:
                        logger.error(f"保存裁剪图失败: {e}")

                # 多尺度OCR识别
                if use_multi_scale:
                    # 生成多个预处理版本
                    versions = multi_scale_preprocess(crop, class_name)
                    
                    best_results = []
                    best_avg_conf = 0
                    
                    for version_name, processed in versions:
                        if save_debug and output_dir:
                            try:
                                debug_dir = Path(output_dir) / "debug_preprocessed"
                                debug_dir.mkdir(exist_ok=True, parents=True)
                                debug_path = debug_dir / f"{prefix}_{version_name}.jpg"
                                cv2.imwrite(str(debug_path), processed)
                                debug_count += 1
                            except Exception:
                                pass
                        
                        # OCR识别
                        if use_voting and class_name in VOTING_CLASSES:
                            ocr_results = voting_ocr.recognize_with_voting(processed, class_name)
                        else:
                            ocr_results = ocr_processor.recognize_text(processed)
                        
                        avg_conf = np.mean([r[0] for r in ocr_results]) if ocr_results else 0
                        
                        if avg_conf > best_avg_conf:
                            best_avg_conf = avg_conf
                            best_results = ocr_results
                            logger.debug(f"使用 {version_name} 的OCR结果 (conf={avg_conf:.2f})")
                else:
                    # 单尺度处理
                    preprocessed = preprocess_image_for_ocr(crop, class_name)
                    
                    if save_debug and output_dir:
                        try:
                            debug_dir = Path(output_dir) / "debug_preprocessed"
                            debug_dir.mkdir(exist_ok=True, parents=True)
                            debug_path = debug_dir / f"{prefix}_preprocessed.jpg"
                            cv2.imwrite(str(debug_path), preprocessed)
                            debug_count += 1
                        except Exception:
                            pass
                    
                    if use_voting and class_name in VOTING_CLASSES:
                        best_results = voting_ocr.recognize_with_voting(preprocessed, class_name)
                    else:
                        best_results = ocr_processor.recognize_text(preprocessed)

                # 处理OCR结果
                for conf_score, text, position in best_results:
                    if not text or len(text.strip()) < 1:
                        continue
                    
                    # 验证技术要求文本
                    if class_name == 'tech_req':
                        is_valid, val_score = validate_tech_req_text(text)
                        if not is_valid and conf_score < 0.5:
                            logger.debug(f"跳过低质量技术要求文本: '{text}' (conf={conf_score:.2f})")
                            continue

                    # 计算边界框中心点
                    if position and len(position) >= 4:
                        points = np.array(position)
                        centroid = points.mean(axis=0)
                        sort_key = (centroid[1], centroid[0])
                    else:
                        sort_key = (y1, x1)

                    all_text.append({
                        'text': text,
                        'class': class_name,
                        'confidence': conf_score,
                        'position': position,
                        'sort_key': sort_key,
                        'bbox': [x1, y1, x2, y2]
                    })

            except Exception as e:
                logger.error(f"处理边界框 {box_idx} 时出错: {e}")
                continue

    # 按类别分组并聚类合并
    class_groups = {}
    for item in all_text:
        cls = item['class']
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(item)
    
    # 合并每个类别中的文本
    merged_text = []
    for cls, items in class_groups.items():
        if cls in ['tech_req', 'table']:
            # 技术要求/表格需要按行聚类合并
            items.sort(key=lambda x: x['sort_key'])
            clusters = cluster_text_boxes(items, x_threshold=0.3, y_threshold=0.5)
            merged_text.extend(clusters)
        else:
            merged_text.extend(items)
    
    # 最终排序
    merged_text.sort(key=lambda x: x['sort_key'])

    logger.info(f"提取完成: 共保存{crop_count}个裁剪图，{debug_count}个调试图像，"
                f"识别到{len(merged_text)}个文本（原始{len(all_text)}个）")
    return merged_text


def extract_structured_info(all_text: List[Dict]) -> Dict:
    """从OCR结果中提取结构化信息 - 增强版"""
    structured_info = {
        "技术要求": [],
        "技术要求_原始": "",
        "表格信息": {},
        "提取统计": {
            "总文本段数": len(all_text),
            "技术要求数": 0,
            "表格信息数": 0
        }
    }

    # 统计各类别的数量
    class_counts = {}
    for item in all_text:
        cls = item['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    structured_info["提取统计"]["类别统计"] = class_counts

    # 分离技术要求和表格信息
    tech_req_items = [item for item in all_text if item['class'] == 'tech_req']
    table_items = [item for item in all_text if item['class'] in ['table', 'cell', 'col', 'row']]

    # 处理技术要求
    if tech_req_items:
        tech_req_items.sort(key=lambda x: x['sort_key'])
        tech_texts = [item['text'] for item in tech_req_items]
        tech_req_full = '\n'.join(tech_texts)
        structured_info["技术要求_原始"] = tech_req_full

        tech_items = extract_tech_items(tech_req_full)
        structured_info["技术要求"] = tech_items
        structured_info["提取统计"]["技术要求数"] = len(tech_items)

    # 处理表格信息
    if table_items:
        table_items.sort(key=lambda x: x['sort_key'])
        table_texts = [item['text'] for item in table_items]
        table_full = ' '.join(table_texts)

        table_fields = extract_table_fields(table_full)
        structured_info["表格信息"] = table_fields
        structured_info["提取统计"]["表格信息数"] = len(table_fields)

    # 单类 panel 模型回退：检测框不分 tech_req/table 子类时，
    # 标题栏/技术要求都包在 panel 内，用全部文本做字段与技术要求提取
    if not tech_req_items and not table_items and all_text:
        all_sorted = sorted(all_text, key=lambda x: x['sort_key'])
        all_lines = '\n'.join(item['text'] for item in all_sorted if item['text'].strip())
        structured_info["原始OCR文本_全部"] = all_lines

        if all_lines.strip():
            tech_fallback = extract_tech_items(all_lines)
            if tech_fallback:
                structured_info["技术要求"] = tech_fallback
                structured_info["提取统计"]["技术要求数"] = len(tech_fallback)
                structured_info["技术要求_原始"] = all_lines

            table_fallback = extract_table_fields(all_lines)
            if table_fallback:
                structured_info["表格信息"] = table_fallback
                structured_info["提取统计"]["表格信息数"] = len(table_fallback)

    return structured_info


def extract_tech_items(tech_req_full: str) -> List[str]:
    """从技术要求文本中提取条目 - 增强版"""
    tech_items = []
    
    if not tech_req_full or not tech_req_full.strip():
        return tech_items
    
    # 预处理
    text = tech_req_full.replace('\r\n', '\n').replace('\r', '\n')
    
    # 尝试多种分割模式
    patterns = [
        (r'(?:^|\n)\s*(\d+)[\.．、]\s*([^\n]*?)(?=\n\s*\d+[\.．、]|$)', 'numbered'),
        (r'[(（](\d+)[)）]\s*([^\n(（)）]*?)(?=[(（]\d+[)）]|$)', 'bracket'),
        (r'([一二三四五六七八九十百千万]+)[\.．、]\s*([^\n]*?)(?=\n[一二三四五六七八九十百千万]+[\.．、]|$)', 'chinese'),
    ]
    
    found_items = False
    
    for pattern, pattern_type in patterns:
        if found_items:
            break
        
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            for idx, match in enumerate(matches):
                if len(match) >= 2:
                    _, content = match[0], match[1]
                    content = content.strip()
                    content = re.sub(r'[\.．、，,；;]+$', '', content).strip()
                    
                    if content and len(content) > 1:
                        tech_items.append(f"{idx + 1}：{content}")
            
            if tech_items:
                found_items = True
    
    # 如果没有匹配到序号模式，尝试按换行分割
    if not tech_items:
        lines = [line.strip() for line in text.split('\n')
                 if line.strip() and len(line.strip()) > 2]
        
        for idx, line in enumerate(lines):
            cleaned_line = re.sub(r'^\d+[\.．、]?\s*', '', line)
            cleaned_line = re.sub(r'^[一二三四五六七八九十百千万]+[\.．、]?\s*', '', cleaned_line)
            cleaned_line = re.sub(r'^[(（]\d+[)）]\s*', '', cleaned_line)
            cleaned_line = cleaned_line.strip()
            
            if cleaned_line and len(cleaned_line) > 2:
                is_valid, _ = validate_tech_req_text(cleaned_line)
                if is_valid or len(cleaned_line) > 5:
                    tech_items.append(f"{idx + 1}：{cleaned_line}")
    
    # 去重
    seen = set()
    unique_items = []
    for item in tech_items:
        if '：' in item:
            content = item.split('：', 1)[1]
        else:
            content = item
        
        if content not in seen:
            seen.add(content)
            unique_items.append(item)
    
    return unique_items


def extract_table_fields(table_full: str) -> Dict:
    """从表格文本中提取字段 - 增强版"""
    fields = {}
    
    if not table_full:
        return fields
    
    patterns = {
        "物品名称": [
            r'(?:物品名称|名称|零件名称|图名|物品|零件)[：:\s]*([^\n：:]+?)(?=\s+(?:材料|材质|数量|图号|比例|重量|$))',
            r'(?:名称)[：:\s]*([\u4e00-\u9fa5A-Za-z0-9\-_]+)',
        ],
        "材料": [
            r'(?:材料|材质|牌号|钢号)[：:\s]*([^\n：:]+?)(?=\s+(?:数量|图号|比例|重量|$))',
            r'(ZL\d+|5A\d+[\-A-Z]*|45#|Q\d+|H\d+|T\d+|\d+Cr[A-Z]*|\d+Mn[A-Z]*|HT\d+)',
        ],
        "数量": [
            r'(?:数量|件数|个数)[：:\s]*(\d+)',
        ],
        "图号": [
            r'(?:图号|编号|件号|代号|图样代号)[：:\s]*([A-Za-z0-9\-._]+)',
        ],
        "张数": [
            r'(\d+)\s*/\s*(\d+)',
            r'共\s*(\d+)\s*张',
            r'第\s*(\d+)\s*张\s*共\s*(\d+)\s*张',
        ],
        "重量": [
            r'(?:重量|质量|单重|总重)[：:\s]*([\d.]+)\s*(?:kg|g|千克|克)?',
        ],
        "比例": [
            r'(?:比例|SCALE)[：:\s]*([\d：:]+)',
            r'(\d+)[：:](\d+)',
        ],
        "设计": [
            r'(?:设计|制图|绘图)[：:\s]*([^\n\s]+)',
        ],
        "审核": [
            r'(?:审核|校对|审查)[：:\s]*([^\n\s]+)',
        ],
        "工艺": [
            r'(?:工艺|工艺审核)[：:\s]*([^\n\s]+)',
        ],
        "标准化": [
            r'(?:标准化|标准)[：:\s]*([^\n\s]+)',
        ],
        "批准": [
            r'(?:批准|审定|审批)[：:\s]*([^\n\s]+)',
        ],
    }
    
    for field_name, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, table_full, re.IGNORECASE)
            if match:
                if match.groups():
                    value = next((g for g in match.groups() if g), match.group(0))
                else:
                    value = match.group(0)
                
                value = re.sub(r'^[：:\s]+', '', value).strip()
                value = re.sub(r'[：:\s]+$', '', value).strip()
                
                if field_name == "张数" and len(match.groups()) >= 2:
                    groups = [g for g in match.groups() if g]
                    if len(groups) >= 2:
                        value = f"{groups[0]}/{groups[1]}"
                    elif len(groups) == 1:
                        value = groups[0]
                
                if field_name == "比例":
                    value = re.sub(r'[:：]', ':', value)
                
                if value and len(value) < 100:
                    fields[field_name] = value
                    break
    
    # 如果没有明确匹配到物品名称，尝试从文本中提取
    if "物品名称" not in fields:
        bracket_match = re.search(r'[\u4e00-\u9fa5]{2,6}\s*[(（][^)）]{2,20}[)）]', table_full)
        if bracket_match:
            fields["物品名称"] = bracket_match.group(0)
        else:
            patterns_item = [
                r'[A-Z][A-Z0-9]{2,10}\s*[\u4e00-\u9fa5]+',
                r'[\u4e00-\u9fa5]{2,8}(?:体|件|盒|盖|座|板|架|套|环|垫|圈|销|键|轴|轮|杆|臂|叉|钩)',
                r'[\u4e00-\u9fa5]{2,6}[\s\-]*[\u4e00-\u9fa5]{2,6}',
            ]
            for pattern in patterns_item:
                match = re.search(pattern, table_full)
                if match:
                    fields["物品名称"] = match.group(0)
                    break
    
    return fields


def save_mechanical_drawing_txt(all_text: List[Dict], structured_info: Dict, 
                                txt_path: Path) -> None:
    """保存机械图纸信息到TXT文件"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 机械图纸信息 ===\n\n")

        field_mapping = [
            ("物品名称", ["物品名称", "物品"]),
            ("材料", ["材料", "材质", "牌号"]),
            ("图号", ["图号", "图纸编号"]),
            ("数量", ["数量", "件数"]),
            ("张数", ["张数"]),
            ("重量", ["重量", "质量"]),
            ("比例", ["比例", "SCALE"]),
            ("设计", ["设计"]),
            ("审核", ["审核"]),
            ("工艺", ["工艺"]),
            ("标准化", ["标准化"]),
            ("批准", ["批准"]),
        ]

        table_info = structured_info.get("表格信息", {})
        for display_name, keys in field_mapping:
            value = ""
            for key in keys:
                if key in table_info:
                    value = table_info[key]
                    break
            f.write(f"{display_name}：{value if value else '无'}\n")

        # 技术要求
        f.write("\n=== 技术要求 ===\n")
        requirements = structured_info.get("技术要求", [])
        if requirements:
            for req in requirements:
                f.write(f"{req}\n")
        else:
            f.write("无明确技术要求\n")

        # 原始OCR文本
        f.write("\n=== 原始OCR文本 ===\n")
        tech_req_raw = (structured_info.get("技术要求_原始", "")
                        or structured_info.get("原始OCR文本_全部", ""))
        if tech_req_raw:
            f.write(tech_req_raw)
            f.write("\n")
        else:
            f.write("无\n")

        # 提取统计
        f.write("\n=== 提取统计 ===\n")
        stats = structured_info.get("提取统计", {})
        for key, value in stats.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")


def visualize_results(image: np.ndarray, detected_objects, all_text: List[Dict],
                      output_path: Path) -> None:
    """可视化检测结果"""
    vis_image = image.copy()
    
    colors = {
        'tech_req': (0, 0, 255),
        'table': (0, 255, 0),
        'cell': (255, 0, 0),
        'col': (255, 255, 0),
        'row': (255, 0, 255),
    }
    
    for result in detected_objects:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue
        
        for box in result.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                
                color = colors.get(class_name, (128, 128, 128))
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception:
                continue
    
    # 绘制OCR文本位置
    for item in all_text:
        pos = item.get('position')
        if pos and len(pos) >= 4:
            pts = np.array(pos, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], True, (0, 255, 255), 1)
    
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f"可视化结果保存到: {output_path}")


def process_single_image(image_path: Path, model_path: str, output_dir: Path,
                         use_sahi: bool = True, use_multi_scale: bool = True,
                         use_voting: bool = True) -> bool:
    """处理单张图像 - 增强版"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法加载图像: {image_path}")
            return False

        logger.info(f"处理图像: {image_path}, 尺寸: {image.shape}")

        yolo_processor = YOLOProcessor(model_path)
        ocr_processor = EasyOCRProcessor()

        # 目标检测（使用SAHI）
        detected_objects = yolo_processor.detect_objects(
            image, use_sahi=use_sahi
        )

        # 提取文本（使用多尺度和投票）
        all_text = extract_text_from_regions(
            detected_objects, image, ocr_processor,
            process_id=image_path.stem,
            output_dir=output_dir,
            save_crops=True,
            save_debug=True,
            use_multi_scale=use_multi_scale,
            use_voting=use_voting
        )

        # 提取结构化信息
        structured_info = extract_structured_info(all_text)

        # 保存结果
        txt_path = output_dir / f"{image_path.stem}.txt"
        save_mechanical_drawing_txt(all_text, structured_info, txt_path)
        
        vis_path = output_dir / f"{image_path.stem}_visualization.jpg"
        visualize_results(image, detected_objects, all_text, vis_path)
        
        json_path = output_dir / f"{image_path.stem}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': str(image_path),
                'structured_info': structured_info,
                'all_text': [{k: v for k, v in item.items() if k != 'sort_key'} 
                            for item in all_text]
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"单图像处理完成: {image_path}")
        return True
    except Exception as e:
        logger.error(f"处理图像 {image_path} 时出错: {e}", exc_info=True)
        return False


def process_batch_images(input_dir: Path, model_path: str, output_dir: Path,
                         use_sahi: bool = True, use_multi_scale: bool = True,
                         use_voting: bool = True) -> bool:
    """批量处理图像 - 增强版"""
    try:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图像文件")
            return False

        logger.info(f"批量处理开始，共 {len(image_files)} 个图像文件")

        yolo_processor = YOLOProcessor(model_path)
        ocr_processor = EasyOCRProcessor()

        success_count = 0
        for image_path in image_files:
            logger.info(f"处理图像: {image_path}")
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"无法加载图像: {image_path}")
                    continue

                detected_objects = yolo_processor.detect_objects(
                    image, use_sahi=use_sahi
                )

                all_text = extract_text_from_regions(
                    detected_objects, image, ocr_processor,
                    process_id=image_path.stem,
                    output_dir=output_dir,
                    save_crops=True,
                    save_debug=True,
                    use_multi_scale=use_multi_scale,
                    use_voting=use_voting
                )

                structured_info = extract_structured_info(all_text)

                txt_path = output_dir / f"{image_path.stem}.txt"
                save_mechanical_drawing_txt(all_text, structured_info, txt_path)
                
                vis_path = output_dir / f"{image_path.stem}_visualization.jpg"
                visualize_results(image, detected_objects, all_text, vis_path)
                
                json_path = output_dir / f"{image_path.stem}_result.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'image': str(image_path),
                        'structured_info': structured_info,
                        'all_text': [{k: v for k, v in item.items() if k != 'sort_key'} 
                                    for item in all_text]
                    }, f, ensure_ascii=False, indent=2)

                success_count += 1
                logger.info(f"图像处理完成: {image_path}")
            except Exception as e:
                logger.error(f"处理图像 {image_path} 时出错: {e}", exc_info=True)

        logger.info(f"批量处理完成，成功处理 {success_count}/{len(image_files)} 个图像")
        return True
    except Exception as e:
        logger.error(f"批量处理时出错: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="整合YOLO检测和OCR文本提取的预测脚本（增强版）"
    )
    parser.add_argument("--mode", type=str, required=True, 
                       choices=['single', 'batch'],
                       help="处理模式: single (单图像) 或 batch (批量)")
    parser.add_argument("--input", type=str, required=True,
                       help="输入图像路径 (single模式) 或输入目录 (batch模式)")
    parser.add_argument("--model", type=str, required=True,
                       help="YOLO模型路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录 (默认: predict/时间戳)")
    parser.add_argument("--no-sahi", action="store_true",
                       help="禁用SAHI切片检测")
    parser.add_argument("--no-multi-scale", action="store_true",
                       help="禁用多尺度OCR")
    parser.add_argument("--no-voting", action="store_true",
                       help="禁用多布局投票")

    args = parser.parse_args()

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("predict") / f"{timestamp}_enhanced"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 功能开关
    use_sahi = not args.no_sahi
    use_multi_scale = not args.no_multi_scale
    use_voting = not args.no_voting
    
    logger.info(f"功能开关: SAHI={use_sahi}, 多尺度={use_multi_scale}, 投票={use_voting}")

    # 根据模式处理
    if args.mode == 'single':
        image_path = Path(args.input)
        if not image_path.exists():
            logger.error(f"输入图像不存在: {image_path}")
            sys.exit(1)
        process_single_image(image_path, args.model, output_dir,
                            use_sahi, use_multi_scale, use_voting)
    else:
        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            sys.exit(1)
        process_batch_images(input_dir, args.model, output_dir,
                            use_sahi, use_multi_scale, use_voting)


if __name__ == "__main__":
    main()
