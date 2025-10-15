from collections import defaultdict
from typing import List, Any, Optional
import pickle

import torch
from sentence_transformers import SentenceTransformer, util

__all__ = [
    "get_sentence_model",
    "cope",
    "normalize_branch_terms",
]
from dotenv import load_dotenv

from utils.logging_utils import log

load_dotenv(override=True)
import os
import json
from utils.call_llm import get_prompt, call_4o

def match_each_to_catalog_dict(B_list, A_list, model):

    A_emb = model.encode(A_list, convert_to_tensor=True)
    B_emb = model.encode(B_list, convert_to_tensor=True)
    if A_emb.size(1) != B_emb.size(1):
        raise ValueError(f"向量维度不一致: A={A_emb.size(1)}, B={B_emb.size(1)}")
    sim_mat = util.cos_sim(B_emb, A_emb)  # [len(B), len(A)]
    result = {}
    for i, b_item in enumerate(B_list):
        j = torch.argmax(sim_mat[i]).item()
        score = sim_mat[i, j].item()
        result[b_item] = A_list[j]

    return result

def safe_read_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取代码块中的 JSON
        if "```" in content:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except:
                    pass
    return None

def categorize_products(product_list, template_path="prompts/model_categorize.yaml"):
    # 1. 读取模板
    system_prompt, user_prompt = get_prompt(template_path,context={"name_list":f"{product_list}"})
    # 3. 调用大模型
    text = call_4o(system_prompt, user_prompt, "")
    
    log.info(f"原始产品列表{product_list}")
    log.info(f"LLM建议产品列表{text}")
    # 4. 尝试解析 JSON
    result = safe_read_json(text)
    if not isinstance(result, list):
        log.error("[WARN] 解析失败，返回空列表")
        return [], text

    cleaned = []
    for item in result:
        s = str(item).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned

# ===== 懒加载模型 =====
_model_cache: Optional[SentenceTransformer] = None
def get_sentence_model(name: str = "paraphrase-mpnet-base-v2") -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(name)
    return _model_cache

# ===== 按相似度分组 =====
def merge_node(words: List[str], model: SentenceTransformer, cosine_threshold: float = 0.8) -> List[List[str]]:
    if not words:
        return []
    embeddings = model.encode(words, convert_to_tensor=True)
    shared_pool: List[List[str]] = []
    in_pool = {w: -1 for w in words}

    def union(w1: str, w2: str):
        if in_pool[w1] == -1 and in_pool[w2] == -1:
            shared_pool.append([w1, w2])
            idx = len(shared_pool) - 1
            in_pool[w1] = in_pool[w2] = idx
        elif in_pool[w1] == -1:
            shared_pool[in_pool[w2]].append(w1)
            in_pool[w1] = in_pool[w2]
        elif in_pool[w2] == -1:
            shared_pool[in_pool[w1]].append(w2)
            in_pool[w2] = in_pool[w1]

    for i, w1 in enumerate(words):
        for j in range(i + 1, len(words)):
            w2 = words[j]
            if util.cos_sim(embeddings[i], embeddings[j]).item() > cosine_threshold:
                union(w1, w2)

    for g in shared_pool:
        g.sort(key=len)
    return shared_pool

# ===== 业务清洗 =====
def cope(data: List[List[Any]], model: Optional[SentenceTransformer]) -> List[List[Any]]:
    if not data:
        return data
    rows = [list(r) for r in data]
    # 2) 品牌归一
    brand_list = defaultdict(int)
    for r in rows:
        brand_list[r[0]] += 1
    max_brand = max(brand_list, key=brand_list.get)
    for r in rows:
        r[0] = max_brand
    if model is None:
        model = get_sentence_model()
    ncol = len(rows[0])
    end_col = max(2, ncol - 3)
    for col in range(1, end_col):
        uniq = []
        for r in rows:
            if col < len(r):
                v = str(r[col])
                if v not in uniq:
                    uniq.append(v)
        if not uniq:
            continue
        groups = merge_node(uniq, model)
        if not groups:
            continue
        if col==1:
            for g in groups:
                if len(g)>1:
                    categorized_list=categorize_products(g)
                    result = match_each_to_catalog_dict(g, categorized_list, model)
                    # print(rows[i][1] for i in range(len(rows)))
                    for r in rows:
                        old_val = str(r[1])
                        if old_val in result:
                            r[1] = result[old_val]
                        else:
                            continue
            continue
        rep_map = {}
        for g in groups:
            rep = sorted(g, key=lambda w: (len(w), w))[0]
            for w in g:
                rep_map[w] = rep
        for r in rows:
            if col < len(r) and r[col] in rep_map:
                r[col] = rep_map[r[col]]
        # print(f"[Col {col}] 唯一词数: {len(uniq)}, 分组数: {len(groups)} 示例: {groups[:3]}")

    return rows

# ===== 核心归一化 =====
def normalize_branch_terms(branches: List[List[Any]], model: Optional[SentenceTransformer] = None) -> List[List[Any]]:
    return cope(branches, model)

# ===== 文件归一化 =====
if __name__ == "__main__":
    def load_branches(pkl_path: str) -> List[List[Any]]:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        branches = obj["branches"] if isinstance(obj, dict) and "branches" in obj else obj
        if not isinstance(branches, list):
            raise ValueError(f"文件 {pkl_path} 的 branches 非 list：{type(branches)}")
        return branches

    branches = load_branches("../datas/branch/branch_from_disc/Apple_discussion_branch.pkl")

    normalized= normalize_branch_terms(
        branches, model=get_sentence_model()
    )
