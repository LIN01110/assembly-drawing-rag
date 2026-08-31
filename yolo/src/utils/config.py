from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
        ValueError: 配置文件为空
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    if not cfg:
        raise ValueError(f"配置文件为空: {config_path}")
    
    logger.info(f"成功加载配置文件: {config_path}")
    return cfg


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置字典
    
    Args:
        config: 原始配置字典
        updates: 要更新的配置项
        
    Returns:
        更新后的配置字典
    """
    config_copy = config.copy()
    config_copy.update(updates)
    return config_copy


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置字典是否包含所有必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        配置是否有效的布尔值
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"配置缺少必需的键: {missing_keys}")
        return False
    return True