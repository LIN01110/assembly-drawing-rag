import yaml
from pathlib import Path
from ultralytics import YOLO


def create_data_yaml():
    """创建YOLO数据配置文件"""
    data_config = {
        'path': str(Path.cwd() / 'mydata'),
        'train': 'images1',
        'val': 'images1',  # 如果没有验证集，暂时用训练集
        'names': {
            0: 'tech_req',
            1: 'table'
        },
        'nc': 2
    }

    with open('mydata/data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True)

    print("✅ 创建 data.yaml 完成")


def train_yolo():
    """训练YOLO模型"""
    print("🚀 开始训练YOLO模型...")

    # 创建数据配置文件
    create_data_yaml()

    # 加载预训练模型
    model = YOLO('yolo11n.pt')  # 使用你现有的模型

    # 训练参数
    train_args = {
        'data': 'mydata/data.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 8,
        'workers': 4,
        'device': 0,  # 使用GPU
        'project': 'runs/detect',
        'name': 'drawing_extraction',
        'exist_ok': True,
        'save': True,
        'save_period': 10,
        'plots': True
    }

    # 开始训练
    results = model.train(**train_args)

    print(f"✅ 训练完成! 最佳模型保存于: {model.trainer.best}")
    return model


if __name__ == "__main__":
    train_yolo()