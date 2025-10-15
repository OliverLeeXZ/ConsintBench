import re
import json
import math
import torch
import pandas as pd
from typing import List, Dict, Tuple

from config import EmbedConfig
from embedder_singleton import get_embedder, Embedder
from utils import normalize_text, simple_chunk

# ======== JSON Loading Functions ========

def load_qa_from_questionnaire_json(path: str) -> pd.DataFrame:
    """
    Load Q&A from questionnaire JSON format.
    
    Input format:
    {
      "Question 1": {
        "question": "...",
        "options": [...],
        "answer": "A. ..."
      },
      ...
    }
    Output: DataFrame with question_id, question_text, answer_text columns
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for key, item in raw.items():
        qid_match = re.search(r'\d+', str(key))
        qid = f"Q{qid_match.group(0)}" if qid_match else str(key)
        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("answer", ""))
        opts = item.get("options", [])
        if opts:
            q_for_retrieval = f"{q} Options: " + " ".join([str(o) for o in opts])
        else:
            q_for_retrieval = q
        rows.append({
            "question_id": qid,
            "question_text": q_for_retrieval,
            "answer_text": a
        })
    return pd.DataFrame(rows)

def load_discussions_from_aggregate_json(path: str) -> pd.DataFrame:
    """
    Load discussions from aggregate JSON format.
    
    Input format: list[ {source, discussion, views_count|upvotes_count} ]
    Output: DataFrame with sample_id, discussion_text, source, engagement columns
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for i, item in enumerate(raw):
        src = str(item.get("source", "")).strip().lower()
        txt = normalize_text(str(item.get("discussion", "")))
        eng = item.get("views_count", None)
        if eng is None:
            eng = item.get("upvotes_count", 0)
        try:
            eng = int(str(eng).replace(",", "").strip())
        except Exception:
            try:
                eng = float(eng)
            except Exception:
                eng = 0
        rows.append({
            "sample_id": f"D{i:06d}",
            "discussion_text": txt,
            "source": src,
            "engagement": eng
        })
    return pd.DataFrame(rows)

# ======== Corpus Building Functions ========

def build_question_corpus(df: pd.DataFrame) -> List[Dict]:
    """Build corpus from questions dataframe."""
    corpus = []
    for _, r in df.iterrows():
        qid = str(r.get("question_id", ""))
        qtext = normalize_text(str(r.get("question_text", "")))
        atext = normalize_text(str(r.get("answer_text", "")))
        if not qtext:
            continue
        corpus.append({
            "id": qid,
            "type": "question",
            "text": qtext,
            "answer_text": atext
        })
        if atext:
            corpus.append({
                "id": f"{qid}::with_answer",
                "type": "question_plus_answer",
                "text": f"{qtext} Answer: {atext}",
                "answer_text": atext
            })
    return corpus

def build_discussion_corpus(df: pd.DataFrame, cfg: EmbedConfig) -> List[Dict]:
    """Build corpus from discussions dataframe."""
    corpus = []
    for idx, r in df.iterrows():
        sid = str(r.get("sample_id", idx))
        dtext = normalize_text(str(r.get("discussion_text", "")))
        src = (str(r.get("source", "")).lower() or "")
        eng = r.get("engagement", 0)
        try:
            eng = float(eng)
        except Exception:
            eng = 0.0
        if not dtext:
            continue
        chunks = simple_chunk(dtext, max_tokens=cfg.max_chunk_tokens)
        w = math.log1p(max(0.0, eng))  # pre-compute weight
        for c_idx, ch in enumerate(chunks):
            corpus.append({
                "id": f"{sid}::chunk{c_idx}",
                "type": "discussion_chunk",
                "parent": sid,
                "text": ch,
                "source": src,
                "engagement": eng,
                "weight": w
            })
    return corpus

def encode_corpus(embedder: Embedder, corpus: List[Dict], kind: str, silent: bool = False, max_workers: int = 4) -> Tuple[List[Dict], torch.Tensor]:
    """Encode corpus texts using the embedder with multi-threading."""
    texts = [c["text"] for c in corpus]
    vecs = embedder.encode_parallel(texts, kind=kind, max_workers=max_workers, silent=silent)
    return corpus, vecs

def has_embeddings(json_path: str, expected_dim: int = 1024) -> bool:
    """Check if JSON file already contains embedding vectors (kept for compatibility)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "embedding" in first:
                emb = first["embedding"]
                if isinstance(emb, list) and len(emb) == expected_dim:
                    return True
        if isinstance(data, dict):
            rows = data.get("rows") or data.get("data")
            if rows and isinstance(rows, list):
                first = rows[0]
                if isinstance(first, dict) and "embedding" in first:
                    emb = first["embedding"]
                    if isinstance(emb, list) and len(emb) == expected_dim:
                        return True
        return False
    except Exception:
        return False