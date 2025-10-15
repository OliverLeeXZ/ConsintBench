# utils/encoder.py
import os
from typing import Optional
from sentence_transformers import SentenceTransformer

_ENCODER: Optional[SentenceTransformer] = None

def get_encoder() -> SentenceTransformer:
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER

    # 主进程在 prefetch_hf_model() 中设置的本地目录
    local_dir = os.environ.get("EMB_MODEL_DIR", "hf_models/paraphrase-mpnet-base-v2")

    # 1) 首选：本地离线加载目标模型
    try:
        _ENCODER = SentenceTransformer(local_dir, local_files_only=True)
        return _ENCODER
    except Exception:
        pass  # 本地没有或目录不完整，继续回退

    # 2) 回退：本地离线加载更小的模型（建议提前也下载到缓存）
    try:
        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        return _ENCODER
    except Exception:
        pass

    _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ENCODER
