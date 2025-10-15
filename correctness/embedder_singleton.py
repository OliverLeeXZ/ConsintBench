import os
import torch
import gc
import concurrent.futures
import threading
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import EmbedConfig, EMBED_INSTRUCTION
from utils import normalize_text, pick_device, gpu_memory_info
from model_cache import load_model_with_fallback, is_model_cached, get_cache_dir

# 全局模型单例和缓存
_model_instance = None
_model_lock = threading.Lock()
_embeddings_cache = {}
_cache_lock = threading.Lock()

# 模型加载状态
_model_loading = False
_model_ready = False
_model_error = None

# 模型信息
_model_info = {
    "loaded_time": None,
    "device": "cpu",
    "memory_usage": 0,
    "model_name": "",
    "batch_size": 0,
    "embedding_dim": 0
}

def get_model_status():
    """
    获取模型加载状态信息
    
    Returns:
        Dict: 模型状态信息
    """
    global _model_ready, _model_loading, _model_error, _model_info
    
    status = {
        "ready": _model_ready,
        "loading": _model_loading,
        "error": _model_error,
        "info": _model_info.copy() if _model_info else {}
    }
    
    # 如果模型已就绪，添加内存使用信息
    if _model_ready and torch.cuda.is_available():
        try:
            status["info"]["current_gpu_memory"] = gpu_memory_info()
        except:
            pass
            
    return status

def wait_for_model(timeout=60):
    """
    等待模型加载完成
    
    Args:
        timeout: 超时时间（秒）
        
    Returns:
        bool: 是否加载成功
    """
    global _model_ready, _model_loading, _model_error
    
    start_time = time.time()
    while _model_loading and time.time() - start_time < timeout:
        time.sleep(0.5)
        
    return _model_ready

def get_embedder(cfg: EmbedConfig = None, silent: bool = True, timeout: int = 60) -> 'Embedder':
    """
    获取嵌入模型的单例实例，避免重复加载模型
    
    Args:
        cfg: 配置信息，如果为None且单例不存在则会使用默认配置
        silent: 是否抑制输出
        timeout: 等待模型加载的超时时间（秒）
        
    Returns:
        Embedder实例
    """
    global _model_instance, _model_loading, _model_ready, _model_error, _model_info
    
    # 如果模型已经在加载，等待其完成
    if _model_loading:
        if not silent:
            print("[Embedder] 模型正在被其他线程加载，等待...")
            
        # 等待模型加载完成
        start_time = time.time()
        while _model_loading and time.time() - start_time < timeout:
            time.sleep(0.5)
            
        # 检查是否超时
        if _model_loading:
            raise TimeoutError("等待模型加载超时")
            
        # 检查是否有错误
        if _model_error:
            raise RuntimeError(f"模型加载失败: {_model_error}")
            
        # 如果模型已经就绪，直接返回
        if _model_ready and _model_instance:
            return _model_instance
    
    # 如果没有实例，加载新模型
    if _model_instance is None:
        with _model_lock:
            # 二次检查避免竞争条件
            if _model_instance is None:
                try:
                    _model_loading = True
                    _model_ready = False
                    _model_error = None
                    
                    if not silent:
                        print("[Embedder] 开始加载嵌入模型...")
                    
                    # 默认配置
                    if cfg is None:
                        cfg = EmbedConfig(
                            model_name="all-MiniLM-L6-v2",  # 使用更小更稳定的模型
                            batch_size=32,
                            fp16=False,  # 禁用半精度提高稳定性
                            prefer_gpu=True
                        )
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 创建模型实例
                    _model_instance = Embedder(cfg, silent)
                    
                    # 更新模型信息
                    _model_info.update({
                        "loaded_time": time.time(),
                        "device": _model_instance.device,
                        "model_name": cfg.model_name,
                        "batch_size": cfg.batch_size,
                        "loading_time": time.time() - start_time
                    })
                    
                    # 设置模型就绪标志
                    _model_ready = True
                    
                    if not silent:
                        print(f"[Embedder] 模型加载完成，用时 {time.time() - start_time:.2f} 秒")
                        
                except Exception as e:
                    _model_error = str(e)
                    if not silent:
                        print(f"[Embedder] 模型加载失败: {e}")
                    raise
                    
                finally:
                    _model_loading = False
    
    return _model_instance

class Embedder:
    """Class for generating text embeddings using a sentence transformer model."""
    
    def __init__(self, cfg: EmbedConfig, silent: bool = False):
        self.cfg = cfg
        self.device = pick_device(cfg.prefer_gpu)
        self.silent = silent
        self.batch_size = cfg.batch_size
        self.ready = False
        self.load_time = 0
        self.model = None
        self.embedding_dim = None
        
        # 模型加载前预检查
        self._pre_load_check()
        
        # 加载开始时间
        load_start = time.time()
        
        # Optimize CUDA performance if using GPU
        if self.device == "cuda":
            # Set CUDA device index if multiple GPUs
            cuda_device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            if not silent:
                print(f"[Embedder] Using CUDA device: {cuda_device_id}")
            
            # Optimize CUDA for better performance
            torch.backends.cudnn.benchmark = True  # Speed up training with fixed input sizes
            
            # Print GPU info
            if not silent:
                memory_info = gpu_memory_info()
                print(f"[Embedder] GPU memory: {memory_info}")
        
        # 优先使用缓存模型，减少网络请求
        model_path = cfg.model_name
        if not silent:
            print(f"[Embedder] Loading model: {model_path} on {self.device}")
            
        # 检查模型是否已缓存
        if is_model_cached(model_path):
            if not silent:
                print(f"[Embedder] 使用本地缓存模型: {model_path}")
        else:
            if not silent:
                print(f"[Embedder] 本地缓存未找到，尝试加载在线模型")
                
        start_time = time.time()
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用增强的模型加载函数，支持回退到备用模型
                self.model = load_model_with_fallback(
                    model_path, 
                    self.device,
                    fallback_models=[
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "paraphrase-multilingual-MiniLM-L12-v2"
                    ]
                )
                if not silent:
                    print(f"[Embedder] 模型加载成功，用时 {time.time() - start_time:.2f}秒")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"无法加载嵌入模型，已尝试{max_retries}次: {str(e)}") from e
                if not silent:
                    print(f"[Embedder] 加载失败 (尝试 {retry_count}/{max_retries}): {str(e)}, 重试中...")
                time.sleep(1)  # Wait before retry
        
        # Get embedding dimension
        if self.model is not None:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 记录总加载时间
        self.load_time = time.time() - load_start
        
        # 加载后验证模型
        if not self._validate_model():
            raise RuntimeError("模型加载后验证失败")
            
        self.ready = True
        
    def _pre_load_check(self):
        """模型加载前预检查"""
        # 检查设备
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if not self.silent:
                # 检查GPU内存
                mem_info = gpu_memory_info()
                print(f"[Embedder] GPU内存预检查: {mem_info}")
                
            # 尝试清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
        elif self.cfg.prefer_gpu and not torch.cuda.is_available():
            if not self.silent:
                print("[Embedder] 警告: 请求GPU但不可用，使用CPU")
            self.device = "cpu"
    
    def _validate_model(self):
        """模型加载后验证"""
        if self.model is None:
            return False
            
        # 检查模型是否可用
        try:
            # 尝试直接使用模型编码而不是通过self.encode方法（避免循环依赖）
            test_sentence = "This is a test sentence for model validation."
            normalized_test = normalize_text(test_sentence)
            
            # 直接使用模型编码，不通过self.encode
            test_embedding = self.model.encode(
                [normalized_test],
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=False
            )
            
            # 检查嵌入维度
            if self.embedding_dim is None or self.embedding_dim <= 0:
                if not self.silent:
                    print(f"[Embedder] 警告: 嵌入维度无效: {self.embedding_dim}")
                return False
                
            # 检查嵌入结果
            if test_embedding is None or test_embedding.shape[0] != 1 or test_embedding.shape[1] != self.embedding_dim:
                if not self.silent:
                    print(f"[Embedder] 警告: 嵌入结果验证失败, 形状: {test_embedding.shape if test_embedding is not None else None}")
                return False
                
            if not self.silent:
                print(f"[Embedder] 模型验证成功, 嵌入维度: {self.embedding_dim}")
            return True
            
        except Exception as e:
            if not self.silent:
                print(f"[Embedder] 模型验证失败: {e}")
            return False
    
    def _maybe_instr(self, texts: List[str], kind: str) -> List[str]:
        """给文本添加指令前缀（如果配置了使用指令）"""
        if not self.cfg.use_instruction:
            return texts
        prefix = EMBED_INSTRUCTION["query" if kind == "query" else "passage"]
        return [prefix + t for t in texts]
    
    def encode(self, sentences: List[str], kind: str = "passage", batch_size: int = None, convert_to_tensor: bool = True, 
               normalize_embeddings: bool = True, use_instruction: bool = None, show_progress_bar: bool = False,
               instruction: str = None) -> torch.Tensor:
        """
        编码文本为嵌入向量
        
        Args:
            sentences: 要编码的文本列表
            kind: 文本类型，'query'或'passage'
            batch_size: 批处理大小，如果为None则使用配置的批处理大小
            convert_to_tensor: 是否返回张量而不是numpy数组
            normalize_embeddings: 是否对嵌入向量进行归一化
            use_instruction: 是否使用指令，如果为None则使用配置值
            show_progress_bar: 是否显示进度条
            instruction: 自定义指令，如果为None则使用默认指令
            
        Returns:
            编码后的嵌入向量
        """
        # 检查模型是否就绪
        if not self.ready or self.model is None:
            raise RuntimeError("模型尚未就绪，无法编码文本")
            
        # 使用缓存批处理大小，如果未指定
        if batch_size is None:
            batch_size = self.batch_size or 32
            
        # 对文本进行规范化和预处理
        normalized_sentences = [normalize_text(s) for s in sentences]
        
        # 处理指令
        if use_instruction or self.cfg.use_instruction:
            # 使用_maybe_instr方法添加指令
            normalized_sentences = self._maybe_instr(normalized_sentences, kind)
        
        try:
            # 在编码前清理GPU内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                
            # 批处理编码
            if self.cfg.fp16 and self.device == "cuda":
                # 使用半精度加速
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(
                        normalized_sentences, 
                        batch_size=batch_size,
                        convert_to_tensor=True,  # 强制使用张量提高效率
                        normalize_embeddings=normalize_embeddings or True,  # 默认规范化
                        show_progress_bar=show_progress_bar
                    )
            else:
                # 正常精度编码
                embeddings = self.model.encode(
                    normalized_sentences, 
                    batch_size=batch_size,
                    convert_to_tensor=True,  # 强制使用张量提高效率
                    normalize_embeddings=normalize_embeddings or True,  # 默认规范化
                    show_progress_bar=show_progress_bar
                )
                
            # 确保始终返回tensor，避免detach错误
            # 不再需要转换为numpy，因为index.py等地方需要调用detach()
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)
                
        except Exception as e:
            error_msg = f"编码失败: {str(e)}"
            if not self.silent:
                print(f"[Embedder] {error_msg}")
            raise RuntimeError(error_msg)
            
        return embeddings

    def encode_parallel(self, sentences: List[str], kind: str = "passage", max_workers: int = 4, silent: bool = False) -> torch.Tensor:
        """
        使用多线程并行编码文本，提高处理大量文本时的效率
        
        Args:
            sentences: 要编码的文本列表
            kind: 文本类型，'query'或'passage'
            max_workers: 并行工作线程数
            silent: 是否抑制输出消息
            
        Returns:
            编码后的嵌入向量
        """
        # 简化处理: 避免复杂的并行处理，直接使用主线程
        show_progress = not silent
        if show_progress:
            print(f"[Embedder] encode_parallel调用，使用{len(sentences)}个文本，但简化为直接调用")
        
        try:
            # 直接调用encode方法，避免并行处理复杂性
            # 不传递silent参数，因为encode方法不接受该参数
            return self.encode(sentences, kind=kind, show_progress_bar=show_progress)
        except Exception as e:
            # 提供详细的错误信息
            error_msg = f"encode_parallel失败: {str(e)}"
            if show_progress:
                print(f"[Embedder] {error_msg}")
            
            try:
                # 尝试使用更小的批量
                if len(sentences) > 10:
                    if show_progress:
                        print("[Embedder] 尝试使用更小批次")
                    result_parts = []
                    for i in range(0, len(sentences), 10):
                        batch = sentences[i:i+10]
                        # 不传递silent参数
                        part_result = self.encode(batch, kind=kind)
                        result_parts.append(part_result)
                    
                    # 确保所有部分都是张量并正确合并
                    tensor_parts = []
                    for part in result_parts:
                        if isinstance(part, torch.Tensor):
                            tensor_parts.append(part)
                        else:
                            # 转换numpy数组为张量
                            tensor_parts.append(torch.tensor(part))
                            
                    return torch.cat(tensor_parts)
                else:
                    raise RuntimeError("批次已经很小，无法继续减小")
            except Exception as e2:
                raise RuntimeError(f"{error_msg}，尝试使用更小批次也失败: {str(e2)}")
    
    def encode_with_cache(self, sentences: List[str], **kwargs) -> torch.Tensor:
        """使用缓存编码文本"""
        global _embeddings_cache, _cache_lock
        
        # 为需要编码的文本创建键
        keys = []
        sentences_to_encode = []
        indices = []
        
        for i, s in enumerate(sentences):
            # 规范化文本创建缓存键
            key = normalize_text(s)
            keys.append(key)
            
            # 检查是否在缓存中
            if key not in _embeddings_cache:
                sentences_to_encode.append(s)
                indices.append(i)
        
        # 如果有需要编码的文本
        if sentences_to_encode:
            # 编码新文本
            new_embeddings = self.encode(sentences_to_encode, **kwargs)
            
            # 更新缓存
            with _cache_lock:
                for i, idx in enumerate(indices):
                    _embeddings_cache[keys[idx]] = new_embeddings[i]
        
        # 从缓存中获取结果
        embeddings = []
        for key in keys:
            embeddings.append(_embeddings_cache[key])
            
        # 将结果转换为正确的格式
        if kwargs.get("convert_to_tensor", False):
            return torch.stack(embeddings)
        else:
            import numpy as np
            return np.array(embeddings)

    def clear_cache():
        """清除嵌入缓存"""
        global _embeddings_cache, _cache_lock
        with _cache_lock:
            _embeddings_cache.clear()