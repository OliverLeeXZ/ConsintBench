import os
import multiprocessing
from dataclasses import dataclass

# Environment settings
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:37890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:37890")

# Parallel processing settings
MAX_WORKERS = min(multiprocessing.cpu_count(), 4)  # Limit workers to avoid system overload

# GPU settings
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Default to first GPU

# API Keys and credentials
OPENAI_API_KEY = "xx"

# Model settings
OPENAI_CHAT_MODEL = "chatgpt-4o-latest"  # Model for chat completions
OPENAI_KEYWORD_MODEL = "gpt-4o-mini"     # Model for keyword extraction
EMBED_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"  # Default embedding model

EMBED_INSTRUCTION = {
    "query": "Represent this question for retrieving supporting passages: ",
    "passage": "Represent this passage for retrieval: ",
}

@dataclass
class EmbedConfig:
    model_name: str = EMBED_MODEL_NAME  # Default from global config
    batch_size: int = 64
    fp16: bool = True
    prefer_gpu: bool = True
    use_instruction: bool = True
    max_chunk_tokens: int = 250 
    tfidf_max_features: int = 200_000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_max_df: float = 0.95
    tfidf_min_df: int = 1
    # GPU acceleration settings
    cuda_device_id: str = "0"  # Specific CUDA device to use
    optimize_cuda: bool = True  # Whether to optimize CUDA for inference
    dynamic_batch_size: bool = True  # Whether to dynamically adjust batch size based on GPU memory
    
@dataclass
class ProcessingConfig:
    max_workers: int = MAX_WORKERS
    gpu_memory_fraction: float = 0.8  # Maximum fraction of GPU memory to use
    enable_parallel: bool = True      # Whether to enable parallel processing
    log_level: str = "INFO"           # Logging level: DEBUG, INFO, WARNING, ERROR

@dataclass
class IndexPaths:
    artifacts_dir: str = "artifacts"
    q_index: str = "faiss_q.index"
    d_index: str = "faiss_d.index"
    q_meta: str = "q_meta.jsonl"
    d_meta: str = "d_meta.jsonl"
    cfg_json: str = "embed_config.json"
    # Files for keyword index
    d_tfidf_matrix: str = "d_tfidf.npz"
    d_tfidf_vectorizer: str = "d_tfidf_vectorizer.pkl"
@dataclass
class DirectoryPaths:
    questionnaire_dir: str = "questionnaire"     # Directory containing questionnaire JSON files
    discussion_dir: str = "discussion"           # Directory containing raw discussion files
    cleaned_dir: str = "cleaned_discussion"     # Directory containing cleaned discussion files
    embed_dir: str = "embed"                    # Directory containing embeddings
    output_dir: str = "results"                 # Directory for results output
