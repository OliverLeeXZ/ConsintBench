import asyncio
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from utils.cope_json import extract_brand_and_product_from_questionnaire
from utils.llm_judge import main
from utils.tree_node import TreeNode, load_tree_from_json
from utils.logging_utils import log
from utils.cope_json import *

# ===================== 配置 =====================

# 作为唯一键的列（用于判定“同一行”）
KEY_COLS = ['brand', 'product_series', 'generate_questionnaire_model', 'task_type']
file_path = 'tree_result.csv'

# 初始化 DataFrame
if os.path.exists(file_path):
    tree_result = pd.read_csv(file_path)
    for k in KEY_COLS:
        if k not in tree_result.columns:
            tree_result[k] = ""
else:
    tree_result = pd.DataFrame(columns=KEY_COLS)


# ===================== 基础工具函数 =====================

def _ensure_columns(row: Dict[str, Any]):
    """确保 row 中的列都在 DataFrame 里存在。"""
    global tree_result
    for col in row.keys():
        if col not in tree_result.columns:
            tree_result[col] = pd.NA

def _slug(text: str) -> str:
    """将任意文本转为文件名安全的 slug。"""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(text)).strip('_') or "UNK"

def _sort_and_save():
    """
    排序并保存到 CSV：
    1) 先按 brand 升序
    2) 再按 product_series（ALL 在前）
    3) 同 brand & model 内 is_root=True 在最上
    """

    global tree_result, file_path
    #print("----------------sort开始------------------")
    #print(tree_result)

    if 'is_root' not in tree_result.columns:
        tree_result['is_root'] = False

    # product_series 排序键：ALL 最前
    ps_key = tree_result['product_series'].astype(str).apply(lambda x: 0 if x == 'ALL' else 1)
    root_key = tree_result['is_root'].apply(lambda x: 0 if bool(x) else 1)

    tree_result = tree_result.assign(
        _ps_key=ps_key,
        _root_key=root_key
    ).sort_values(
        by=['brand', '_ps_key', 'product_series', '_root_key', 'generate_questionnaire_model', 'task_type'],
        ascending=[True, True, True, True, True, True]
    ).drop(columns=['_ps_key', '_root_key'])

    # 固定列顺序保存（KEY_COLS 优先）
    fixed = [c for c in KEY_COLS if c in tree_result.columns]
    others = [c for c in tree_result.columns if c not in fixed]
    tree_result = tree_result[fixed + others]

    tree_result.to_csv(file_path, index=False)

    #print("----------------sort结束------------------")
    #print(tree_result)

def insert_or_update(row: Dict[str, Any]):
    """
    Upsert：row 至少包含 KEY_COLS，其余指标列将自动扩展。
    额外约定：若包含 'is_root' / 'root_scope' 字段，将一并保存。
    """

    global tree_result
    #print(f"进入insert_or_update{tree_result}")

    _ensure_columns(row)    #确保包含需要的所有列

    #print(f"after ensure{tree_result}")
    
    if 'is_root' not in tree_result.columns:
        tree_result['is_root'] = False
    if 'is_root' not in row:
        row['is_root'] = False  # 默认 False
    if 'root_scope' not in tree_result.columns:
        tree_result['root_scope'] = ""

    # 复合主键匹配, 判断是否有匹配行
    #cond = pd.Series([True] * len(tree_result))     
    cond = pd.Series(True, index=tree_result.index)
    
    #print(f"after cond{tree_result}")

    #KEY_COLS = ['brand', 'product_series', 'generate_questionnaire_model', 'task_type']
    for k in KEY_COLS:
        if k not in row:
            raise ValueError(f"缺少关键键字段：{k}")
        cond = cond & (tree_result[k].astype(str) == str(row[k]))


    #print(f"after cond2{tree_result}")
    #print(cond)

    # 更新或插入
    # 如果有匹配行, 更新该行的数据
    if cond.any():
        idx = tree_result.index[cond][0]

        for k, v in row.items():
            tree_result.at[idx, k] = v
        log.info(f"[Upsert] 更新：{row['brand']} | {row['product_series']} | {row['generate_questionnaire_model']} | {row['task_type']}")
        #print(tree_result)
        #print(f"更新{row}")
    else:
        tree_result = pd.concat([tree_result, pd.DataFrame([row])], ignore_index=True)
        log.info(f"[Upsert] 插入：{row['brand']} | {row['product_series']} | {row['generate_questionnaire_model']} | {row['task_type']}")
        #print(tree_result)
        #print(f"插入{row}")

    _sort_and_save()


# ===================== 业务计算函数 =====================

def get_metrics(root: TreeNode) -> Tuple[list, float, float, float]:
    """返回：(2-7层广度分数组, 平均深度, 总分, llm占位)"""
    breadth_vals = root.get_weight_topic()[2:8]  # 2-7层广度分
    avg_depth = root.get_avg_leaf_depth()
    total = root.get_total_w()
    llm = 0.0
    return breadth_vals, avg_depth, total, llm

def write_brand_root(brand: str, product_series:str,root1: Dict[str, Any]) -> None:
    """写入每个品牌的 root 总览行（is_root=True，四个主键置为 ALL）。"""
    breadth_vals = root1["breadth_vals_2"]
    row = {
        "brand": brand,
        "product_series": product_series,
        "generate_questionnaire_model": "ALL",
        "task_type": "ALL",
        "total_score": root1["total_2"],
        "avg_depth": root1["avg_depth_2"],
        "breadth_L2": breadth_vals[0] if len(breadth_vals) > 0 else pd.NA,
        "breadth_L3": breadth_vals[1] if len(breadth_vals) > 1 else pd.NA,
        "breadth_L4": breadth_vals[2] if len(breadth_vals) > 2 else pd.NA,
        "breadth_L5": breadth_vals[3] if len(breadth_vals) > 3 else pd.NA,
        "breadth_L6": breadth_vals[4] if len(breadth_vals) > 4 else pd.NA,
        "breadth_L7": breadth_vals[5] if len(breadth_vals) > 5 else pd.NA,
    }

    # 如果已经写入过了就不写了
    #---------------------------------------------------------------------
    global tree_result

    _ensure_columns(row)    #确保包含需要的所有列
    
    if 'is_root' not in tree_result.columns:
        tree_result['is_root'] = False
    if 'is_root' not in row:
        row['is_root'] = False  # 默认 False
    if 'root_scope' not in tree_result.columns:
        tree_result['root_scope'] = ""

  
    cond = pd.Series(True, index=tree_result.index)
    
    for k in KEY_COLS:
        if k not in row:
            raise ValueError(f"缺少关键键字段：{k}")
        cond = cond & (tree_result[k].astype(str) == str(row[k]))

    if cond.any():
        return
    #--------------------------------------------------------------------
    
    insert_or_update(row)

def write_model_root(brand: str, model: str, rootm: Dict[str, Any]) -> None:
    """写入每个型号的 root 总览行（is_root=True，GQM/TT=ALL）。"""
    if rootm is None:
        return
    breadth_vals = rootm["breadth_vals_2"]
    row = {
        "brand": brand,
        "product_series": model,
        "generate_questionnaire_model": "ALL",
        "task_type": "ALL",
        "total_score": rootm["total_2"],
        "avg_depth": rootm["avg_depth_2"],
        "breadth_L2": breadth_vals[0] if len(breadth_vals) > 0 else pd.NA,
        "breadth_L3": breadth_vals[1] if len(breadth_vals) > 1 else pd.NA,
        "breadth_L4": breadth_vals[2] if len(breadth_vals) > 2 else pd.NA,
        "breadth_L5": breadth_vals[3] if len(breadth_vals) > 3 else pd.NA,
        "breadth_L6": breadth_vals[4] if len(breadth_vals) > 4 else pd.NA,
        "breadth_L7": breadth_vals[5] if len(breadth_vals) > 5 else pd.NA,
    }
    insert_or_update(row)

def write_metrics_for_file(filename: str, brand_root: Dict[str, Any]) -> None:
    brand, product_series, generate_questionnaire_model, task_type = extract_brand_and_product_from_questionnaire(filename)
    #-------------------------------------------------------------------------
    #首先检测这个内容是否已经完整地存在于excel之中了，若是则不再进行打分
    row = {
        "brand": brand,
        "product_series": product_series,
        "generate_questionnaire_model": generate_questionnaire_model,
        "task_type": task_type,
    }

    global tree_result

    _ensure_columns(row)    #确保包含需要的所有列
    
    if 'is_root' not in tree_result.columns:
        tree_result['is_root'] = False
    if 'is_root' not in row:
        row['is_root'] = False  # 默认 False
    if 'root_scope' not in tree_result.columns:
        tree_result['root_scope'] = ""

  
    cond = pd.Series(True, index=tree_result.index)
    
    for k in KEY_COLS:
        if k not in row:
            raise ValueError(f"缺少关键键字段：{k}")
        cond = cond & (tree_result[k].astype(str) == str(row[k]))

    if cond.any():
        return
    #--------------------------------------------------------------------------

    #print(f"{str(Path("datas/tree/lighted")/filename)}")
    #root2 = load_tree_from_json(str(Path("datas/tree/lighted")/filename))
    #print(f"./datas/tree/lighted/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^lighted_tree.json")
    root2 = load_tree_from_json(f"./datas/tree/lighted/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^lighted_tree.json")
    breadth_base, avg_depth_base, total_base, llm_base = [value for key, value in brand_root.items()]
    breadth_vals_2, avg_depth_2, total_2, llm_2 = get_metrics(root2)
    breadth_ratio = [
        (breadth_vals_2[i] / breadth_base[i] * 100) if i < len(breadth_vals_2) and breadth_base[i] != 0 else 0
        for i in range(len(breadth_vals_2))
    ]

    total_2=total_2/total_base*100
    #未启动LLM评分
    #llm_judge = asyncio.run(main(brand, product_series, generate_questionnaire_model, task_type))
    llm_judge={"LLM_judge":0, "depth":0, "model":0, "usage_scenario":0,"aspect":0, "feeling":0, "comparison":0, "tendency":0}

    judge_fields: Dict[str, Any] = {}
    if isinstance(llm_judge, dict):
        judge_fields = llm_judge
    elif isinstance(llm_judge, (list, tuple)):
        base_names = ["LLM_judge", "depth", "model", "usage_scenario",
                      "aspect", "feeling", "comparison", "tendency"]
        for i, v in enumerate(llm_judge):
            name = base_names[i] if i < len(base_names) else f"judge_{i}"
            judge_fields[name] = v
    else:
        judge_fields = {"LLM_judge": llm_judge}

    row = {
        "brand": brand,
        "product_series": product_series,
        "generate_questionnaire_model": generate_questionnaire_model,
        "task_type": task_type,
        "total_score": total_2,
        "avg_depth": avg_depth_2,
        "breadth_L2": breadth_ratio[0] if len(breadth_ratio) > 0 else pd.NA,
        "breadth_L3": breadth_ratio[1] if len(breadth_ratio) > 1 else pd.NA,
        "breadth_L4": breadth_ratio[2] if len(breadth_ratio) > 2 else pd.NA,
        "breadth_L5": breadth_ratio[3] if len(breadth_ratio) > 3 else pd.NA,
        "breadth_L6": breadth_ratio[4] if len(breadth_ratio) > 4 else pd.NA,
        "breadth_L7": breadth_ratio[5] if len(breadth_ratio) > 5 else pd.NA,
    }
    
    row.update(judge_fields)

    insert_or_update(row)

def excel():
    # 1) 收集文件
    lighted_dir = "datas/tree/lighted"
    file_list = os.listdir(lighted_dir)

    # 2) 搜集所有的lighted树，涵盖品牌+产品系列
    brand_list = []
    product_series_list=[]
    generate_questionnaire_model_list=[]
    task_type_list=[]
    for file in file_list:
        brand, product_series, generate_questionnaire_model, task_type = extract_brand_and_product_from_questionnaire(file)
        brand_list.append(brand)
        product_series_list.append(product_series)
        generate_questionnaire_model_list.append(generate_questionnaire_model)
        task_type_list.append(task_type)

        
    print(brand_list)
    print(product_series_list)

    #3) 遍历所有lighted tree
    for i in range(len(brand_list)):
        brand_key=brand_list[i]
        product_series_key=product_series_list[i]
        generate_questionnaire_model_key=generate_questionnaire_model_list[i]
        task_type_key=task_type_list[i]
        
        brand_tree_path = f"datas/tree/{brand_key}^{product_series_key}^discussion_tree.json"

        if not os.path.exists(brand_tree_path):
            log.warning(f"[Skip brand root] 未找到讨论树：{brand_tree_path}")
            breadth_vals_b = [1, 1, 1, 1, 1, 1]
            avg_depth_b = 0.0
            total_b = 0.0
            llm_b = 0.0
        else:
            brand_root_tree = load_tree_from_json(brand_tree_path)    #确实读取的是自己的brand_product_tree
            breadth_vals_b, avg_depth_b, total_b, llm_b = get_metrics(brand_root_tree)

        brand_root = {
            "breadth_vals_2": breadth_vals_b,
            "avg_depth_2": avg_depth_b,
            "total_2": total_b,
            "llm_2": llm_b,
        }
        # 填写ground_truth得分情况
        write_brand_root(brand_key, product_series_key , brand_root)

        #4) 遍历所有相关questionnaire, 进行打分
        json_files=get_json_filenames("./datas/questionnaire")


        for json_file in json_files:
            json_brand,json_product_series ,json_model,json_task=extract_brand_and_product_from_questionnaire(json_file)
            if json_brand==brand_key and json_product_series==product_series_key and json_model==generate_questionnaire_model_key and json_task==task_type_key:
                write_metrics_for_file(json_file, brand_root)


if __name__=="__main__":
    excel()


