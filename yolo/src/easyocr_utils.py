"""
PaddleOCR 工具模块 - 用于文本和表格文本检测
"""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EasyOCRProcessor:
    """PaddleOCR 处理器（保持原有类名以确保兼容性）"""

    def __init__(self, languages: List[str] = ['ch_sim', 'en'], gpu: bool = True):  # 默认设为True
        """
        初始化 EasyOCR 处理器

        Args:
            languages: 识别的语言列表 ['ch_sim', 'en'] 等
            gpu: 是否使用GPU（默认True，如果有GPU则使用）
        """
        self.languages = languages
        self.use_gpu = gpu
        self.reader = None
        self.initialized = False

    def initialize(self):
        """初始化 PaddleOCR 读取器"""
        if not self.initialized:
            try:
                from paddleocr import PaddleOCR
                logger.info(f"初始化 PaddleOCR，语言: {self.languages}, GPU: {self.use_gpu}")
                
                # 转换语言列表为PaddleOCR支持的格式
                if 'ch_sim' in self.languages or 'ch' in self.languages:
                    ocr_lang = 'ch'
                elif 'en' in self.languages:
                    ocr_lang = 'en'
                else:
                    ocr_lang = 'ch'
                
                self.reader = PaddleOCR(
                    use_textline_orientation=True,
                    lang=ocr_lang
                )
                self.initialized = True
                logger.info("PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
                raise

    def detect_and_recognize(self, image_path: Path, **kwargs) -> List[Dict[str, Any]]:
        """
        检测和识别图像中的文本

        Args:
            image_path: 图像路径
            **kwargs: 额外参数，如 confidence_threshold, width_ths, add_margin 等

        Returns:
            识别结果列表
        """
        if not self.initialized:
            self.initialize()

        # 默认参数
        conf_threshold = kwargs.get('conf_threshold', 0.5)
        width_ths = kwargs.get('width_ths', 0.6)  # 提高水平文本合并阈值
        add_margin = kwargs.get('add_margin', 0.08)  # 增加文本区域边距

        try:
            logger.info(f"处理图像: {image_path}")

            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return []

            # 图像预处理 - 提高识别准确率
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # 高斯模糊去噪
            denoised = cv2.GaussianBlur(binary, (3, 3), 0)
            
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            # 优化识别参数
            # PaddleOCR的readtext方法参数与EasyOCR不同，需要调整
            results = self.reader.ocr(
                image_rgb,
                det=True,  # 启用文本检测
                rec=True,  # 启用文本识别
                cls=True  # 启用方向分类
            )

            # 格式化结果
            formatted_results = []
            if results and results[0]:
                for i, (bbox, (text, confidence)) in enumerate(results[0]):
                    if confidence < conf_threshold:
                        continue

                    # 文本后处理 - 提高准确率
                    # 去除特殊字符
                    text = re.sub(r'[^\w\u4e00-\u9fa5、：:./()（）\-]', '', text)
                    # 去除前后空格
                    text = text.strip()
                    # 替换全角字符为半角
                    text = text.replace('，', ',').replace('。', '.').replace('；', ';')
                    text = text.replace('：', ':').replace('（', '(').replace('）', ')')
                    text = text.replace('、', ',').replace('\u3000', ' ')
                    
                    if not text:
                        continue

                    # 提取边界框坐标
                    points = [(int(p[0]), int(p[1])) for p in bbox]

                    # 计算边界框
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # 提取文本区域
                    roi = image[y_min:y_max, x_min:x_max]

                    formatted_results.append({
                        'id': i,
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': {
                            'points': points,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max,
                            'width': x_max - x_min,
                            'height': y_max - y_min
                        },
                        'image_size': {
                            'width': image.shape[1],
                            'height': image.shape[0]
                        }
                    })

            logger.info(f"在 {image_path.name} 中检测到 {len(formatted_results)} 个文本区域")
            return formatted_results

        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            return []

    def process_folder(self, source_dir: Path, output_dir: Path, **kwargs):
        """
        处理整个文件夹的图像

        Args:
            source_dir: 源图像文件夹
            output_dir: 输出文件夹
            **kwargs: 传递给 detect_and_recognize 的参数
        """
        if not self.initialized:
            self.initialize()

        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        # 查找所有图像
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(source_dir.glob(ext))

        if not image_paths:
            logger.warning(f"在 {source_dir} 中未找到图像文件")
            return

        logger.info(f"找到 {len(image_paths)} 张图像")

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        processed_count = 0

        for image_path in image_paths:
            try:
                # 处理单张图像
                results = self.detect_and_recognize(image_path, **kwargs)
                
                # 提取表格字段
                table_fields = self.extract_table_fields(results)

                # 保存结果到JSON
                json_path = output_dir / f"{image_path.stem}_ocr.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'processed_time': datetime.now().isoformat(),
                        'results': results,
                        'total_texts': len(results),
                        'table_fields': table_fields  # 新增表格字段
                    }, f, ensure_ascii=False, indent=2)

                # 保存文本区域图像（可选）
                if kwargs.get('save_crops', True):
                    self._save_text_crops(image_path, results, output_dir)

                all_results[image_path.name] = len(results)
                processed_count += 1

                logger.info(f"已处理 {processed_count}/{len(image_paths)}: {image_path.name}")

            except Exception as e:
                logger.error(f"处理失败 {image_path}: {e}")

        # 生成汇总报告
        self._generate_summary(all_results, output_dir)

        logger.info(f"OCR处理完成，处理了 {processed_count}/{len(image_paths)} 张图像")
        return output_dir

    def _save_text_crops(self, image_path: Path, results: List[Dict], output_dir: Path):
        """保存文本区域图像"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return

            crops_dir = output_dir / 'text_crops' / image_path.stem
            crops_dir.mkdir(parents=True, exist_ok=True)

            for i, result in enumerate(results):
                bbox = result['bbox']
                roi = image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']]

                if roi.size > 0:
                    crop_path = crops_dir / f"text_{i}_{result['text'][:20]}.png"
                    cv2.imwrite(str(crop_path), roi)

        except Exception as e:
            logger.error(f"保存文本区域失败: {e}")

    def extract_table_fields(self, ocr_results: List[Dict]) -> Dict[str, str]:
        """
        从OCR结果中提取表格字段
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            提取的表格字段字典
        """
        # 收集所有文本
        all_texts = [result['text'] for result in ocr_results]
        full_text = ' '.join(all_texts)
        
        # 按行分组（基于y坐标）
        rows = {}
        for result in ocr_results:
            y_center = (result['bbox']['y_min'] + result['bbox']['y_max']) / 2
            # 找到最接近的行
            row_key = None
            for y in rows.keys():
                if abs(y_center - y) < 10:  # 10像素内视为同一行
                    row_key = y
                    break
            if row_key is None:
                row_key = y_center
                rows[row_key] = []
            rows[row_key].append(result)
        
        # 按行排序
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        table = {}
        
        # 常见表格字段
        common_fields = [
            # 基础信息
            "物品", "序号", "镀涂", "材质", "GB/T", "比例",
            # 新增字段
            "规格", "型号", "数量", "单位", "尺寸", "重量",
            "厚度", "长度", "宽度", "高度", "直径", "半径",
            "面积", "体积", "颜色", "表面处理", "加工工艺",
            "生产批次", "生产日期", "有效期", "供应商", "制造商",
            "品牌", "产地", "标准", "等级", "类别", "备注",
            "项目", "内容", "参数", "数值", "单位", "要求",
            "检测结果", "合格标准", "检验员", "检验日期",
            "图号", "页码", "版本", "状态", "审批", "复核",
            "设计", "审核", "工艺", "标准化", "批准", "校对", "制图"
        ]
        
        # 从全文中提取字段
        for key in common_fields:
            match = re.search(rf'{key}[:：]?\s*(\S+)', full_text)
            if match:
                table[key] = match.group(1)
        
        # 从行中提取键值对
        for _, row_results in sorted_rows:
            # 按x坐标排序
            row_results.sort(key=lambda x: x['bbox']['x_min'])
            
            # 检查是否包含冒号分隔的键值对
            for i, result in enumerate(row_results):
                text = result['text']
                if '：' in text or ':' in text:
                    # 分割键值对
                    delimiter = '：' if '：' in text else ':'
                    parts = text.split(delimiter, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        table[key] = value
                
                # 检查是否前一个元素是键，当前是值
                if i > 0:
                    prev_text = row_results[i-1]['text']
                    # 检查前一个文本是否是常见字段
                    for common_key in common_fields:
                        if common_key in prev_text and common_key not in table:
                            table[common_key] = text
        
        # 特殊字段处理
        # 张数
        zhang_match = re.search(r'(\d+)/共(\d+)张', full_text)
        if zhang_match:
            table["张数"] = zhang_match.group(0)
        
        # 日期
        date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)', full_text)
        if date_match:
            table["日期"] = date_match.group(1)
        
        # 编号
        code_match = re.search(r'(编号[:：]?\s*\S+)', full_text)
        if code_match:
            table["编号"] = code_match.group(1)
        
        return table
    
    def _generate_summary(self, results: Dict, output_dir: Path):
        """生成汇总报告"""
        summary_path = output_dir / "ocr_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PaddleOCR 处理汇总报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理图像数: {len(results)}\n")
            f.write(f"总检测文本数: {sum(results.values())}\n\n")

            f.write("各图像检测结果:\n")
            f.write("-" * 60 + "\n")
            for img_name, text_count in results.items():
                f.write(f"{img_name}: {text_count} 个文本\n")

            f.write("\n" + "=" * 60 + "\n")


def create_ocr_config():
    """创建 EasyOCR 配置文件"""
    config = {
        'easyocr': {
            'enabled': True,
            'languages': ['ch_sim', 'en'],
            'use_gpu': False,
            'confidence_threshold': 0.5,
            'width_ths': 0.5,  # 水平方向文本合并阈值
            'add_margin': 0.05,  # 文本区域边距
            'save_crops': True,
            'save_json': True,
            'output_format': 'json'
        }
    }
    return config