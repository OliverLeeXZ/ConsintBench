import os
import json
import numpy as np
from typing import List, Dict, Optional, Set

from config import EmbedConfig, IndexPaths
from embedder import Embedder
from index import load_faiss, search_faiss, load_tfidf_index, search_tfidf
from utils import load_metadata_jsonl

def run_search(
    out_dir: str,
    queries: List[str],
    top_k: int = 5,
    alpha: float = 0.15,                 # engagement weight coefficient
    source_whitelist: Optional[List[str]] = None,  # source filtering
    retrieval_mode: str = "hybrid",      # "keyword" | "semantic" | "hybrid"
    keyword_weight: float = 0.65,        # keyword score weight in hybrid retrieval (0~1)
    prefilter_k: int = 200               # for hybrid/semantic, first use keywords to get candidates then semantic filter/rerank
):
    """
    Run search with multiple retrieval modes and options.
    
    Args:
        out_dir: Directory containing index files
        queries: List of query strings
        top_k: Number of results to return per query
        alpha: Weight coefficient for engagement (0~0.3 common)
        source_whitelist: List of allowed sources (None for no filtering)
        retrieval_mode: "keyword", "semantic", or "hybrid"
        keyword_weight: Weight of keyword scores in hybrid mode (0~1)
        prefilter_k: For hybrid/semantic, number of candidates to prefilter with keywords
        
    Returns:
        List of lists of result texts for each query
    """
    paths = IndexPaths(artifacts_dir=out_dir)
    with open(os.path.join(out_dir, paths.cfg_json), "r", encoding="utf-8") as f:
        cfg = EmbedConfig(**json.load(f))

    d_meta = load_metadata_jsonl(os.path.join(out_dir, paths.d_meta))
    wl = set([s.strip().lower() for s in source_whitelist]) if source_whitelist else None

    use_semantic = retrieval_mode in ("semantic", "hybrid")
    if use_semantic:
        # 检查是否已有预加载的模型实例
        preloaded = os.environ.get("EMBEDDER_PRELOADED") == "1"
        if preloaded:
            # 使用单例模式获取已预加载的模型
            from embedder_singleton import get_embedder
            embedder = get_embedder(None)
        else:
            # 初始化新的嵌入器
            embedder = Embedder(cfg)
        
        d_index = load_faiss(os.path.join(out_dir, paths.d_index))

    # Load keyword index
    use_keyword = retrieval_mode in ("keyword", "hybrid") or prefilter_k > 0
    if use_keyword:
        tfidf_vec, tfidf_X = load_tfidf_index(out_dir, paths)
        kw_topk = max(top_k * 10, prefilter_k)

    results_per_query = []

    # First run keyword retrieval
    kw_pairs_all = None
    if use_keyword:
        kw_pairs_all = search_tfidf(tfidf_vec, tfidf_X, queries, top_k=kw_topk)  # List[List[(doc_idx, score)]]

    # Semantic retrieval (with optional keyword candidate filtering)
    sem_ids_all, sem_scores_all = None, None
    if use_semantic:
        if prefilter_k > 0 and kw_pairs_all:
            candidate_lists = []
            for pairs in kw_pairs_all:
                candidate_lists.append(set([p[0] for p in pairs]))
        else:
            candidate_lists = [None] * len(queries)

        q_vecs = embedder.encode(queries, kind="query")
        base_sem_k = max(top_k * 10, 200)
        ids_list, scores_list = search_faiss(d_index, q_vecs, top_k=base_sem_k)

        if any(c is not None for c in candidate_lists):
            new_ids_list, new_scores_list = [], []
            for cand_set, id_list, sc_list in zip(candidate_lists, ids_list, scores_list):
                if cand_set is None:
                    new_ids_list.append(id_list)
                    new_scores_list.append(sc_list)
                else:
                    tmp_ids, tmp_scores = [], []
                    for idx, sc in zip(id_list, sc_list):
                        if idx in cand_set:
                            tmp_ids.append(idx)
                            tmp_scores.append(sc)
                    if not tmp_ids:  # fallback
                        tmp_ids, tmp_scores = id_list[:top_k], sc_list[:top_k]
                    new_ids_list.append(tmp_ids)
                    new_scores_list.append(tmp_scores)
            ids_list, scores_list = new_ids_list, new_scores_list

        sem_ids_all, sem_scores_all = ids_list, scores_list

    # Merge scores & filter
    for qi, q in enumerate(queries):
        kw_pairs = kw_pairs_all[qi] if kw_pairs_all else []
        kw_scores = {doc: sc for doc, sc in kw_pairs}

        sem_ids = sem_ids_all[qi] if sem_ids_all else []
        sem_raw = sem_scores_all[qi] if sem_scores_all else []
        sem_scores = {doc: sc for doc, sc in zip(sem_ids, sem_raw)}

        union_docs = set(kw_scores.keys()) | set(sem_scores.keys())
        if not union_docs:
            results_per_query.append([])
            continue

        def _minmax_norm(d: Dict[int, float]) -> Dict[int, float]:
            if not d:
                return {}
            vals = np.array(list(d.values()), dtype=np.float32)
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax - vmin < 1e-9:
                return {k: 1.0 for k in d.keys()}
            return {k: (v - vmin) / (vmax - vmin + 1e-9) for k, v in d.items()}

        kw_n = _minmax_norm(kw_scores)
        sem_n = _minmax_norm(sem_scores)

        hits = []
        for doc_idx in union_docs:
            meta = d_meta[doc_idx]
            src = (meta.get("source", "") or "").lower()
            if wl and src not in wl:
                continue
            w = float(meta.get("weight", 0.0))  # log1p(engagement)

            if retrieval_mode == "keyword":
                fused = kw_n.get(doc_idx, 0.0)
            elif retrieval_mode == "semantic":
                fused = sem_n.get(doc_idx, 0.0)
            else:  # hybrid
                fused = keyword_weight * kw_n.get(doc_idx, 0.0) + (1.0 - keyword_weight) * sem_n.get(doc_idx, 0.0)

            fused *= (1.0 + alpha * w)  # engagement weight
            hits.append({
                "id": meta["id"],
                "parent": meta.get("parent"),
                "source": src,
                "engagement": meta.get("engagement", 0),
                "score": float(fused),
                "text": meta["text"]
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        hits = hits[:top_k]
        results_per_query.append([h["text"] for h in hits])

    return results_per_query
