import torch
import gc
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from config import EmbedConfig, EMBED_INSTRUCTION
from utils import normalize_text, pick_device, gpu_memory_info

class Embedder:
    """Class for generating text embeddings using a sentence transformer model."""
    
    def __init__(self, cfg: EmbedConfig, silent: bool = False):
        self.cfg = cfg
        self.device = pick_device(cfg.prefer_gpu)
        self.silent = silent
        
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
        
        # 直接使用线上模型
        model_path = cfg.model_name
        if not silent:
            print(f"[Embedder] Using online model: {model_path}")
            print(f"[Embedder] Loading model: {model_path} on {self.device}")
        
        try:
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True
            )
            
            # Use mixed precision for faster computation on GPU
            if cfg.fp16 and self.device != "cpu":
                try:
                    self.model = self.model.to(torch.float16)
                    if not silent:
                        print("[Embedder] FP16 enabled for faster inference")
                except Exception as e:
                    if not silent:
                        print(f"[Embedder] FP16 not applied: {e}")
            
            # Increase max sequence length for better context handling
            self.model.max_seq_length = 512
        except Exception as e:
            if not silent:
                print(f"[Embedder] Error loading model: {e}")
                print("[Embedder] Falling back to simpler model: all-MiniLM-L6-v2")
            try:
                # 尝试加载备用模型
                self.model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device=self.device,
                    trust_remote_code=True,
                )
                self.model.max_seq_length = 512
            except Exception as e2:
                if not silent:
                    print(f"[Embedder] Failed to load fallback model: {e2}")
                raise RuntimeError("无法加载嵌入模型，请确保网络连接或提供本地模型文件")

    def _maybe_instr(self, texts: List[str], kind: str) -> List[str]:
        """Add instruction prefix to texts if configured."""
        if not self.cfg.use_instruction:
            return texts
        prefix = EMBED_INSTRUCTION["query" if kind == "query" else "passage"]
        return [prefix + t for t in texts]

    def optimize_batch_size(self, text_count: int) -> int:
        """Dynamically optimize batch size based on available GPU memory and text count."""
        if self.device == "cpu":
            return self.cfg.batch_size
        
        # For small datasets, use default batch size
        if text_count < 100:
            return self.cfg.batch_size
            
        # Check available GPU memory and adjust batch size accordingly
        if self.device == "cuda":
            try:
                mem_info = gpu_memory_info()
                free_mem = mem_info.get("free", 0)
                # Heuristic: ~2MB per text in batch at 512 tokens (conservative estimate)
                # This is very approximate and should be tuned based on actual model size
                text_mem_estimate = 2 * 1024 * 1024  # 2MB per text in bytes
                # Use 70% of available memory at most
                safe_batch_size = max(16, min(self.cfg.batch_size, int(0.7 * free_mem / text_mem_estimate)))
                if not self.silent:
                    print(f"[Embedder] Optimized batch size: {safe_batch_size} (from default {self.cfg.batch_size})")
                return safe_batch_size
            except Exception as e:
                if not self.silent:
                    print(f"[Embedder] Error optimizing batch size: {e}, using default")
                return self.cfg.batch_size
        return self.cfg.batch_size
    
    @torch.inference_mode()
    def encode(self, texts: List[str], kind: str = "passage", batch_size: Optional[int] = None, silent: bool = False) -> torch.Tensor:
        """Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            kind: Type of texts ('query' or 'passage')
            batch_size: Optional batch size override (defaults to config batch size)
            silent: Whether to suppress output messages
            
        Returns:
            Tensor of L2-normalized embeddings (N, dim)
        """
        # Use instance-level silent setting if not overridden
        silent = silent or self.silent
        
        # Prepare texts
        texts = [normalize_text(t) for t in texts]
        texts = self._maybe_instr(texts, kind=kind)
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Optimize batch size if not explicitly provided
        bs = batch_size or self.optimize_batch_size(len(texts))
        if not silent:
            print(f"[Embedder] Encoding {len(texts)} texts with batch size {bs} on {self.device}")
        
        # Perform the encoding
        vecs = self.model.encode(
            texts,
            batch_size=bs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=not silent,
        )
        
        # Log memory usage after encoding
        if self.device == "cuda" and not silent:
            print(f"[Embedder] GPU memory after encoding: {gpu_memory_info()}")
            
        return vecs  # (N, dim), L2-normalized -> cosine similarity can use dot product