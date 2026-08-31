from ultralytics import YOLO
from pathlib import Path
import shutil
import time
from typing import Dict, Any, Optional
import sys
import os
from datetime import datetime
import glob
import tempfile


# 修复相对导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具模块
try:
    from src.utils.image_processing import save_image, load_image, preprocess_image, postprocess_crop
    from src.utils.config import load_config
    from src.utils.logger import get_default_logger
except ImportError as e:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('image_extraction.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"无法导入工具模块: {e}")
else:
    logger = get_default_logger()


def predict(cfg_path: Path, 
           output_dir: Optional[Path] = None,
           preprocess_opts: Optional[Dict[str, Any]] = None,
           postprocess_opts: Optional[Dict[str, Any]] = None) -> Path:
    """
    从2D装配图中提取小图像
    
    Args:
        cfg_path: 配置文件路径
        output_dir: 输出目录（可选，默认使用配置中的设置）
        preprocess_opts: 预处理选项
        postprocess_opts: 后处理选项
        
    Returns:
        提取结果的根目录路径
        
    Raises:
        FileNotFoundError: 配置文件或模型文件不存在
        ValueError: 配置格式错误或预测失败
    """
    start_time = time.time()
    print(f"[DEBUG] 开始预测任务，配置文件: {cfg_path}")
    logger.info(f"开始预测任务，配置文件: {cfg_path}")
    
    try:
        # 加载配置
        print("[DEBUG] 加载配置...")
        cfg = load_config(cfg_path)
        print(f"[DEBUG] 配置内容: {cfg}")
        
        logger.info("成功加载配置")
        
        # 设置输出目录
        if output_dir:
            root_out = output_dir
        else:
            # 在输出目录中加入时间戳，便于区分不同批次的预测结果
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root_out = Path(cfg.get('project', 'predict')) / timestamp
        
        # 创建输出目录（如果不存在）
        print(f"[DEBUG] 创建输出目录: {root_out}")
        root_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {root_out}")
        
        # 配置预测参数
        cfg['save_crop'] = True
        cfg['project'] = str(root_out)
        
        # 验证模型文件存在
        model_path = cfg.pop('model', None)
        if not model_path:
            raise ValueError("配置文件中缺少 'model' 参数")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 执行预测
        logger.info(f"开始预测，配置: {cfg}")
        predict_start_time = time.time()
        
        # 如果提供了预处理选项，先处理图像再预测
        if preprocess_opts:
            logger.info(f"应用预处理选项: {preprocess_opts}")
            
            # 获取所有图像路径
            import glob
            source = cfg.get('source', '')
            if isinstance(source, str):
                # 支持多种文件格式
                image_paths = []
                if Path(source).is_dir():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        image_paths.extend(glob.glob(f"{source}/{ext}"))
                else:
                    # 单个文件
                    image_paths.append(source)
                
                # 创建临时目录存储预处理后的图像
                import tempfile
                temp_dir = Path(tempfile.mkdtemp())
                logger.info(f"创建临时目录存储预处理图像: {temp_dir}")
                
                # 批量预处理所有图像
                preprocessed_paths = []
                for img_path in image_paths:
                    try:
                        img = load_image(Path(img_path))
                        preprocessed = preprocess_image(img, **preprocess_opts)
                        temp_path = temp_dir / Path(img_path).name
                        save_image(preprocessed, temp_path)
                        preprocessed_paths.append(str(temp_path))
                        # 及时释放内存
                        del img
                        del preprocessed
                    except Exception as e:
                        logger.error(f"预处理图像 {img_path} 失败: {e}")
                        preprocessed_paths.append(img_path)  # 使用原始图像
                
                # 更新配置使用预处理后的图像
                original_source = cfg['source']
                cfg['source'] = str(temp_dir)
        
        # 分离自定义参数和YOLO标准参数
        custom_params = ['preprocess', 'postprocess', 'output_format', 'overwrite']
        yolo_cfg = {k: v for k, v in cfg.items() if k not in custom_params}
        
        # 确保使用CPU设备以避免CUDA错误
        yolo_cfg['device'] = 'cpu'
        
        # 执行预测
        results = model.predict(**yolo_cfg)
        
        # 清理临时目录
        if preprocess_opts and 'temp_dir' in locals():
            cfg['source'] = original_source  # 恢复原始配置
            shutil.rmtree(temp_dir)
            logger.info(f"清理临时目录: {temp_dir}")
        
        predict_end_time = time.time()
        logger.info(f"预测完成，耗时: {predict_end_time - predict_start_time:.2f}秒")
        
        # 从配置中读取预处理和后处理选项
        config_preprocess = cfg.get('preprocess', {})
        config_postprocess = cfg.get('postprocess', {})
        output_format = cfg.get('output_format', 'PNG')
        
        # 合并参数（函数参数优先于配置文件）
        final_preprocess = config_preprocess.copy()
        if preprocess_opts:
            final_preprocess.update(preprocess_opts)
        
        final_postprocess = config_postprocess.copy()
        if postprocess_opts:
            final_postprocess.update(postprocess_opts)
        final_postprocess['output_format'] = output_format
        
        logger.info(f"使用的预处理选项: {final_preprocess}")
        logger.info(f"使用的后处理选项: {final_postprocess}")
        
        # 按原图名分文件夹保存裁剪图
        crop_root = root_out / 'crops_by_name'
        crop_root.mkdir(exist_ok=True)
        
        total_crops = 0
        total_images = len(results)
        processed_images = 0
        
        # 使用单进程处理图像
        
        def process_image(result):
            """处理单个图像的函数"""
            orig_path = Path(result.path)
            orig_name = orig_path.stem
            logger.info(f"处理图像: {orig_path.name}")
            
            sub_folder = crop_root / orig_name
            sub_folder.mkdir(exist_ok=True)
            
            # 保存裁剪结果
            result.save_crop(save_dir=sub_folder)
            
            # 后处理和重命名 - 使用rglob递归查找所有子目录中的图像
            crops = list(sub_folder.rglob('*.png')) + list(sub_folder.rglob('*.jpg'))
            image_crops = 0
            
            for crop_idx, crop_path in enumerate(crops):
                try:
                    # 加载裁剪图像
                    crop_img = load_image(crop_path)
                    
                    # 应用后处理
                    crop_img = postprocess_crop(crop_img, **final_postprocess)
                    
                    # 确定输出格式和扩展名
                    output_ext = output_format.lower()
                    
                    # 重命名并保存
                    new_name = f"{orig_name}_{crop_idx}.{output_ext}"
                    new_path = sub_folder / new_name
                    
                    save_image(crop_img, new_path, format=output_format)
                    

                    
                    # 及时释放内存
                    del crop_img
                    
                    # 删除原始裁剪文件
                    if new_path != crop_path:
                        crop_path.unlink()
                    
                    image_crops += 1
                except Exception as e:
                    logger.error(f"处理裁剪图像失败 {crop_path}: {e}", exc_info=True)
            
            return image_crops
        
        # 单进程处理所有图像（避免并行死锁问题）
        logger.info(f"使用单进程处理图像")
        for idx, r in enumerate(results):
            total_crops += process_image(r)
            processed_images += 1
            logger.info(f"进度: {processed_images}/{total_images} 图像已处理")
        
        # 记录结果
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"\n=== 预测任务完成 ===")
        logger.info(f"总处理时间: {elapsed_time:.2f} 秒")
        logger.info(f"处理图像数量: {len(results)}")
        logger.info(f"提取小图像数量: {total_crops}")
        logger.info(f"结果目录: {crop_root}")



        print(f">>> 预测完成，结果：{crop_root}")

        return crop_root
        
    except Exception as e:
        logger.error(f"预测任务失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8图像分割预测工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    try:
        result_dir = predict(Path(args.config))
        print(f"预测完成，结果保存在: {result_dir}")
    except Exception as e:
        print(f"预测失败: {e}")
        exit(1)

# 在 predict 函数末尾（288 行后），加：
