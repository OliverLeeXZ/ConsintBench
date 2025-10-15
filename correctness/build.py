import os
import json
import time
import torch
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

from config import EmbedConfig, IndexPaths
from embedder_singleton import get_embedder
from index import build_faiss_index, save_faiss, build_tfidf_index, save_tfidf_index
from data_loader import build_discussion_corpus, encode_corpus
from utils import set_torch_deterministic, save_metadata_jsonl, gpu_memory_info

def run_build(
    disc_path: str,
    out_dir: str,
    cfg: EmbedConfig,
    silent: bool = False
):
    """
    Build the search indices from discussion data.
    
    Args:
        disc_path: Path to discussions data file (JSON or CSV)
        out_dir: Output directory for index files
        cfg: Embedding and indexing configuration
        silent: Whether to suppress output messages
    """
    if not silent:
        print("\n===== Building Search Indices with GPU Acceleration =====\n")
    start_time = time.time()
    
    # Set deterministic mode for reproducible results
    set_torch_deterministic(42)
    os.makedirs(out_dir, exist_ok=True)
    paths = IndexPaths(artifacts_dir=out_dir)
    
    # Check GPU availability and memory
    if torch.cuda.is_available() and not silent:
        print(f"\n[GPU] Available: Yes, {torch.cuda.device_count()} device(s)")
        print(f"[GPU] Memory before loading: {gpu_memory_info()}")
    elif not silent:
        print("\n[GPU] Not available, will use CPU (much slower)")

    # Save configuration
    with open(os.path.join(out_dir, paths.cfg_json), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Load discussion data
    if disc_path.lower().endswith(".json"):
        from data_loader import load_discussions_from_aggregate_json
        disc_df = load_discussions_from_aggregate_json(disc_path)
    else:
        disc_df = pd.read_csv(disc_path)

    if not silent:
        print(f"[Build] Discussions rows: {len(disc_df)}")

    # 检查是否已有预加载的模型实例
    preloaded = os.environ.get("EMBEDDER_PRELOADED") == "1"
    if preloaded and not silent:
        print("[Build] 使用已预加载的模型实例")
        embedder = get_embedder(None, silent=silent)  # 使用已加载的模型
    else:
        # 如果没有预加载，使用传入的配置获取嵌入器
        if not silent:
            print("[Build] 未找到预加载模型，创建新实例")
        embedder = get_embedder(cfg, silent=silent)

    # Build discussion chunk corpus
    d_corpus = build_discussion_corpus(disc_df, cfg)

    # Encode (as passages)
    if not silent:
        print(f"[Build] Encoding discussion chunks (as passages): {len(d_corpus)}")
    d_meta, d_vecs = encode_corpus(embedder, d_corpus, kind="passage", silent=silent)

    # FAISS index
    if not silent:
        print("[Build] Building FAISS indexes ...")
    d_index = build_faiss_index(d_vecs)

    # Build TF-IDF keyword index
    if not silent:
        print("[Build] Building TF-IDF keyword index ...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(d_corpus, cfg)

    # Save indices to disk
    if not silent:
        print("\n[Build] Saving indices to disk...")
    save_faiss(d_index, os.path.join(out_dir, paths.d_index))
    save_metadata_jsonl(os.path.join(out_dir, paths.d_meta), d_meta)
    save_tfidf_index(tfidf_vectorizer, tfidf_matrix, out_dir, paths)
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if not silent:
            print(f"[GPU] Memory after processing: {gpu_memory_info()}")
    
    # Print completion summary
    total_time = time.time() - start_time
    if not silent:
        print(f"\n[Done] Build completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"[Done] Artifacts saved to: {out_dir}")
        print(f" - {paths.d_index}")
        print(f" - {paths.d_meta}")
        print(f" - {paths.d_tfidf_matrix}, {paths.d_tfidf_vectorizer}")
        print(f" - {paths.cfg_json}")