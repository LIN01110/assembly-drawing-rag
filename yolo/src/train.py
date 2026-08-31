from ultralytics import YOLO
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 导入工具模块
try:
    from .utils.config import load_config
    from .utils.logger import get_default_logger
except ImportError as e:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"无法导入工具模块: {e}")
else:
    logger = get_default_logger()


def train(cfg_path: Path, 
         output_dir: Optional[Path] = None,
         resume: bool = False) -> Path:
    """
    训练YOLO模型
    
    Args:
        cfg_path: 配置文件路径
        output_dir: 输出目录（可选，默认使用配置中的设置）
        resume: 是否从上次的检查点继续训练
        
    Returns:
        最佳模型权重的路径
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误或训练失败
    """
    start_time = time.time()
    logger.info(f"开始训练任务，配置文件: {cfg_path}")
    
    try:
        # 加载配置
        cfg = load_config(cfg_path)
        
        # 验证必需的配置项
        required_keys = ['model', 'data', 'epochs', 'batch', 'imgsz']
        missing_keys = [key for key in required_keys if key not in cfg]
        if missing_keys:
            raise ValueError(f"配置缺少必需的键: {missing_keys}")
        
        logger.info("成功加载并验证配置")
        
        # 设置输出目录
        if output_dir:
            cfg['project'] = str(output_dir)
        
        # 设置恢复训练
        if resume:
            cfg['resume'] = True
        
        # 加载模型
        model_path = cfg.pop('model')
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 开始训练 - 显式指定主要参数，避免**cfg可能的解析问题
        logger.info(f"开始训练，原始配置: {cfg}")
        try:
            # 从配置中提取必要参数，或者使用默认值
            data = cfg.pop('data')
            epochs = cfg.pop('epochs', 1)
            batch = cfg.pop('batch', 16)
            imgsz = cfg.pop('imgsz', 640)
            project = cfg.pop('project', 'runs')
            name = cfg.pop('name', 'detect_train')
            exist_ok = cfg.pop('exist_ok', True)
            device = cfg.pop('device', 'cpu')
            
            logger.info(f"使用显式参数训练: data={data}, epochs={epochs}, batch={batch}, imgsz={imgsz}, project={project}, name={name}, exist_ok={exist_ok}, device={device}")
            train_results = model.train(
                data=data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project=project,
                name=name,
                exist_ok=exist_ok,
                device=device,
                verbose=True,  # 总是显示详细信息
                **cfg  # 传递剩余参数
            )
            logger.info(f"训练完成，结果类型: {type(train_results)}")
            logger.info(f"训练完成，结果: {train_results}")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}", exc_info=True)
            raise
        
        # 获取最佳权重路径
        try:
            logger.info(f"获取最佳权重路径，save_dir: {model.trainer.save_dir}")
            
            # 使用与test_direct_train.py相同的方式获取路径
            import os
            weights_dir = os.path.join(model.trainer.save_dir, 'weights')
            best_pt_path = os.path.join(weights_dir, 'best.pt')
            logger.info(f"最佳权重路径: {best_pt_path}")
            
            # 检查目录结构
            logger.info(f"保存目录是否存在: {os.path.exists(model.trainer.save_dir)}")
            if os.path.exists(model.trainer.save_dir):
                logger.info(f"保存目录内容: {os.listdir(model.trainer.save_dir)}")
                logger.info(f"权重目录是否存在: {os.path.exists(weights_dir)}")
                if os.path.exists(weights_dir):
                    logger.info(f"权重目录内容: {os.listdir(weights_dir)}")
            
            if not os.path.exists(best_pt_path):
                raise ValueError(f"训练完成但未生成最佳权重文件: {best_pt_path}")
            
            # 转换为Path对象返回
            best_pt_path = Path(best_pt_path)
        except Exception as e:
            logger.error(f"获取最佳权重路径时发生错误: {e}", exc_info=True)
            raise
        
        # 记录结果
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"\n=== 训练任务完成 ===")
        logger.info(f"总训练时间: {elapsed_time:.2f} 秒")
        logger.info(f"最佳模型权重: {best_pt_path}")
        logger.info(f"训练结果: {train_results}")
        
        return best_pt_path
        
    except Exception as e:
        logger.error(f"训练任务失败: {e}", exc_info=True)
        raise