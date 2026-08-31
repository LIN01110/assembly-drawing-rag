from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> Image.Image:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        加载的图像对象
        
    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: 无法加载图像
    """
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        image = Image.open(image_path)
        image.load()  # 确保图像完全加载
        logger.info(f"成功加载图像: {image_path.name}, 尺寸: {image.size}")
        return image
    except Exception as e:
        raise ValueError(f"无法加载图像 {image_path}: {e}")


def preprocess_image(image: Image.Image, 
                     resize: Optional[Tuple[int, int]] = None, 
                     grayscale: bool = False, 
                     normalize: bool = False) -> Image.Image:
    """
    图像预处理
    
    Args:
        image: 输入图像
        resize: 调整大小的目标尺寸（宽, 高）
        grayscale: 是否转换为灰度图
        normalize: 是否归一化（0-1范围）
        
    Returns:
        预处理后的图像
    """
    processed = image.copy()
    
    # 转换为RGB格式
    if processed.mode != 'RGB':
        processed = processed.convert('RGB')
    
    # 灰度转换
    if grayscale:
        processed = ImageOps.grayscale(processed)
    
    # 调整大小
    if resize:
        processed = processed.resize(resize, Image.LANCZOS)
    
    # 归一化
    if normalize:
        np_image = np.array(processed) / 255.0
        processed = Image.fromarray((np_image * 255).astype(np.uint8))
    
    return processed


def postprocess_crop(crop: Image.Image, 
                     padding: int = 0, 
                     resize: Optional[Tuple[int, int]] = None, 
                     output_format: str = 'PNG',
                     contrast: float = 1.0,
                     brightness: float = 1.0,
                     sharpness: float = 1.0,
                     auto_contrast: bool = False,
                     auto_levels: bool = False) -> Image.Image:
    """
    裁剪图像后处理
    
    Args:
        crop: 裁剪的图像
        padding: 添加的边框大小
        resize: 调整大小的目标尺寸
        output_format: 输出图像格式
        contrast: 对比度调整因子（1.0为原始）
        brightness: 亮度调整因子（1.0为原始）
        sharpness: 锐化调整因子（1.0为原始）
        auto_contrast: 是否自动对比度调整
        auto_levels: 是否自动色阶调整
        
    Returns:
        后处理后的图像
    """
    processed = crop.copy()
    
    # 添加padding
    if padding > 0:
        padded_size = (processed.width + 2 * padding, processed.height + 2 * padding)
        padded = Image.new('RGB', padded_size, color=(255, 255, 255))
        padded.paste(processed, (padding, padding))
        processed = padded
    
    # 自动对比度调整
    if auto_contrast:
        processed = ImageOps.autocontrast(processed)
    
    # 自动色阶调整
    if auto_levels:
        processed = ImageOps.equalize(processed)
    
    # 亮度、对比度、锐化调整
    if contrast != 1.0 or brightness != 1.0 or sharpness != 1.0:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(brightness)
        
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(contrast)
        
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(sharpness)
    
    # 调整大小
    if resize:
        processed = processed.resize(resize, Image.LANCZOS)
    
    return processed


def save_image(image: Image.Image, 
               save_path: Path, 
               format: Optional[str] = None) -> None:
    """
    保存图像
    
    Args:
        image: 要保存的图像
        save_path: 保存路径
        format: 图像格式（可选）
        
    Raises:
        ValueError: 保存图像失败
    """
    try:
        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        image.save(save_path, format=format)
        logger.info(f"成功保存图像: {save_path}")
    except Exception as e:
        raise ValueError(f"保存图像失败 {save_path}: {e}")

def get_image_info(image_path: Path) -> dict:
    """
    获取图像信息
    
    Args:
        image_path: 图像路径
        
    Returns:
        包含图像信息的字典
    """
    image = load_image(image_path)
    return {
        "path": str(image_path),
        "name": image_path.name,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_bytes": image_path.stat().st_size
    }
