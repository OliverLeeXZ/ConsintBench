import os
import torch
import faiss
import joblib
import numpy as np
from typing import List, Dict, Tuple

from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from config import EmbedConfig, IndexPaths

# ======== FAISS Vector Index ========

def build_faiss_index(vecs: torch.Tensor) -> faiss.IndexFlatIP:
    """Build a FAISS index from embeddings for inner product search."""
    # 安全处理张量转换为numpy
    if isinstance(vecs, torch.Tensor):
        v = vecs.detach().cpu().numpy().astype("float32")
    else:
        # 如果已经是numpy数组，直接使用
        v = vecs.astype("float32") if hasattr(vecs, 'astype') else np.array(vecs, dtype="float32")
    
    dim = v.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(v)
    return index

def save_faiss(index: faiss.IndexFlatIP, path: str):
    """Save a FAISS index to disk."""
    faiss.write_index(index, path)

def load_faiss(path: str) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    return faiss.read_index(path)

def search_faiss(index: faiss.IndexFlatIP, query_vecs: torch.Tensor, top_k: int = 10) -> Tuple[List[List[int]], List[List[float]]]:
    """Search a FAISS index with query vectors."""
    q = query_vecs.detach().cpu().numpy().astype("float32")
    scores, ids = index.search(q, top_k)
    return ids.tolist(), scores.tolist()

# ======== TF-IDF Keyword Index ========

def build_tfidf_index(d_corpus: List[Dict], cfg: EmbedConfig):
    """Build a TF-IDF keyword index from discussion chunk texts."""
    docs = [c["text"] for c in d_corpus]  # order must match d_meta
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        max_features=cfg.tfidf_max_features,
        max_df=cfg.tfidf_max_df,
        min_df=cfg.tfidf_min_df,
        token_pattern=r"(?u)\b\w[\w\-]+\b",  # preserve hyphens/numbers
        norm="l2"
    )
    X = vectorizer.fit_transform(docs)  # csr_matrix (N_docs x V)
    return vectorizer, X

def save_tfidf_index(vectorizer: TfidfVectorizer, X: csr_matrix, out_dir: str, paths: IndexPaths):
    """Save TF-IDF vectorizer and matrix to disk."""
    joblib.dump(vectorizer, os.path.join(out_dir, paths.d_tfidf_vectorizer))
    save_npz(os.path.join(out_dir, paths.d_tfidf_matrix), X)

def load_tfidf_index(out_dir: str, paths: IndexPaths):
    """Load TF-IDF vectorizer and matrix from disk."""
    vec = joblib.load(os.path.join(out_dir, paths.d_tfidf_vectorizer))
    X = load_npz(os.path.join(out_dir, paths.d_tfidf_matrix))
    return vec, X

def search_tfidf(vec: TfidfVectorizer, X: csr_matrix, queries: List[str], top_k: int = 50) -> List[List[Tuple[int, float]]]:
    """Search TF-IDF index for each query, return doc ids and scores.
    
    Returns:
        List of lists of (doc_idx, score) tuples for each query, 
        where score is cosine similarity (dot product with L2 normalization)
    """
    Q = vec.transform(queries)  # (Q x V)
    sims = Q @ X.T              # (Q x N)
    results = []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue
        idxs = row.indices
        vals = row.data
        if len(vals) > top_k:
            top = np.argpartition(vals, -top_k)[-top_k:]
            idxs, vals = idxs[top], vals[top]
        order = np.argsort(-vals)
        pairs = [(int(idxs[j]), float(vals[j])) for j in order]
        results.append(pairs[:top_k])
    return results
