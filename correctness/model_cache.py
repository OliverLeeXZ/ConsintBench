import os
import torch
import shutil
import logging
import tempfile
import threading
from typing import Dict, Optional, Union
import hashlib
import time
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_cache')
logger.setLevel(logging.ERROR)  # 设置为ERROR级别，避免过多输出

# 全局缓存锁和状态
_cache_lock = threading.Lock()
_model_download_status = {}  # 记录模型下载状态
_model_load_semaphore = threading.Semaphore(1)  # 限制同时加载模型的数量为1

def get_cache_dir() -> str:
    """获取模型缓存目录"""
    # 首先尝试使用环境变量
    cache_dir = os.environ.get('MODEL_CACHE_DIR')
    
    # 如果未设置，使用默认路径
    if not cache_dir:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "huggingface")
    
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def model_hash(model_name: str) -> str:
    """为模型名称生成一个短的哈希值，用于缓存识别"""
    return hashlib.md5(model_name.encode()).hexdigest()[:10]

def is_model_cached(model_name: str) -> bool:
    """检查模型是否已经被缓存到本地"""
    cache_dir = get_cache_dir()
    model_dir = os.path.join(cache_dir, model_name.replace('/', '_'))
    # 检查目录是否存在并且包含必要的模型文件
    if os.path.exists(model_dir):
        # 对于sentence-transformers模型，检查关键文件
        if os.path.exists(os.path.join(model_dir, "config.json")) or \
           os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) or \
           os.path.exists(os.path.join(model_dir, "sentence_bert_config.json")):
            return True
    return False

def wait_for_model_download(model_name: str, timeout: int = 1800) -> bool:
    """等待模型下载完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        with _cache_lock:
            status = _model_download_status.get(model_name)
            if status == "completed":
                return True
            elif status == "failed":
                return False
        # 短暂等待后继续检查
        time.sleep(1)
    return False

def set_model_download_status(model_name: str, status: str) -> None:
    """设置模型下载状态"""
    with _cache_lock:
        _model_download_status[model_name] = status

def save_model_metadata(model_name: str, metadata: Dict) -> None:
    """保存模型的元数据，如大小、加载时间等"""
    cache_dir = get_cache_dir()
    metadata_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def load_model_with_fallback(model_name: str, device: str, fallback_models: list = None):
    """
    尝试加载模型，如果失败则回退到替代模型
    
    Args:
        model_name: 要加载的模型名称
        device: 使用的设备 (cuda 或 cpu)
        fallback_models: 备用模型列表，如果主模型加载失败
        
    Returns:
        加载的模型对象
    """
    from sentence_transformers import SentenceTransformer
    
    # 设置默认的备用模型
    if fallback_models is None:
        fallback_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    
    # 首先尝试加载指定的模型
    try:
        # 使用信号量限制同时加载的模型数量
        _model_load_semaphore.acquire()
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        _model_load_semaphore.release()
        return model
    except Exception as e:
        _model_load_semaphore.release()
        logger.error(f"加载模型 {model_name} 失败: {e}")
        
        # 尝试备用模型
        for fallback in fallback_models:
            try:
                logger.info(f"尝试加载备用模型: {fallback}")
                _model_load_semaphore.acquire()
                model = SentenceTransformer(fallback, device=device, trust_remote_code=True)
                _model_load_semaphore.release()
                return model
            except Exception as e2:
                _model_load_semaphore.release()
                logger.error(f"加载备用模型 {fallback} 失败: {e2}")
        
        # 如果所有尝试都失败，抛出异常
        raise RuntimeError("无法加载任何嵌入模型，请检查网络连接或模型路径")

def create_local_model_copy(model_name: str, local_path: str) -> None:
    """将远程模型复制到本地路径，以便离线使用"""
    from sentence_transformers import SentenceTransformer
    
    # 先加载模型，这将触发下载
    try:
        model = SentenceTransformer(model_name)
        # 保存到指定路径
        model.save(local_path)
        return True
    except Exception as e:
        logger.error(f"创建本地模型副本失败: {e}")
        return False
