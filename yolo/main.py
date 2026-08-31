from pathlib import Path
import logging
import yaml
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_DIR = Path('config')
DATA_BIG = Path('big.png')   # 你自己的大图

if __name__ == '__main__':
    logger.info("开始运行程序")
    
    # 1. 训练
    try:
        logger.info("开始训练")
        
        # 加载训练配置
        train_cfg_path = CONFIG_DIR / 'train.yaml'
        logger.info(f"加载训练配置: {train_cfg_path}")
        with open(train_cfg_path, 'r', encoding='utf-8') as f:
            train_cfg = yaml.safe_load(f)
        
        # 提取训练参数
        model_path = train_cfg.get('model', 'yolov8n.pt')
        data = train_cfg.get('data', 'data.yaml')
        epochs = train_cfg.get('epochs', 1)
        batch = train_cfg.get('batch', 16)
        imgsz = train_cfg.get('imgsz', 640)
        project = train_cfg.get('project', 'runs')
        name = train_cfg.get('name', 'detect_train')
        exist_ok = train_cfg.get('exist_ok', True)
        device = train_cfg.get('device', 'cpu')
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 开始训练
        logger.info(f"开始训练，参数: data={data}, epochs={epochs}, batch={batch}, imgsz={imgsz}, project={project}, name={name}, exist_ok={exist_ok}, device={device}")
        train_results = model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            exist_ok=exist_ok,
            device=device,
            verbose=True
        )
        
        logger.info(f"训练完成，结果类型: {type(train_results)}")
        
        # 获取最佳权重路径
        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / 'weights' / 'best.pt'
        logger.info(f"最佳权重路径: {best_pt}")
        
        if not best_pt.exists():
            raise ValueError(f"训练完成但未生成最佳权重文件: {best_pt}")
            
        logger.info(f"训练完成，最佳模型路径: {best_pt}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise
    
    # 2. 更新预测配置里的模型路径（自动）
    predict_cfg = CONFIG_DIR / 'predict.yaml'
    
    # 更新predict.yaml中的模型路径
    logger.info(f"更新预测配置: {predict_cfg}")
    with open(predict_cfg, 'r', encoding='utf-8') as f:
        predict_config = yaml.safe_load(f)
    
    predict_config['model'] = str(best_pt)
    
    with open(predict_cfg, 'w', encoding='utf-8') as f:
        yaml.dump(predict_config, f, default_flow_style=False)
    
    logger.info(f"已更新predict.yaml中的模型路径为: {best_pt}")
    
    # 3. 分割
    logger.info("开始预测")
    from src.predict import predict
    predict(predict_cfg)
    logger.info("预测完成")