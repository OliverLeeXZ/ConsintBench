# -*- coding: utf-8 -*-
"""
树的高层装载逻辑：按“已有 tree.pkl → 有 merged.pkl → 有原始 branch.pkl”三段式。
"""
import os
from dotenv import load_dotenv
from .file_io  import get_path, save_pickle, load_pickle
from .tree_node import create_tree

load_dotenv(override=True)

def load_or_build_tree(branch_file: str, model, cope_func):
    merged_path = get_path("merged", f"{branch_file}_merged.pkl")
    tree_path   = get_path("tree",   f"{branch_file}_tree.pkl")

    branch_dir  = f"{branch_file.split('_')[1]}_branch"
    branch_path = get_path(branch_dir, f"{branch_file}.pkl")

    if os.path.exists(tree_path):
        data = load_pickle(merged_path)
        root = load_pickle(tree_path)
    elif os.path.exists(merged_path):
        data = load_pickle(merged_path)
        root = create_tree(data)
        save_pickle(root, tree_path)
    elif os.path.exists(branch_path):
        raw_data = load_pickle(branch_path)
        data     = cope_func(raw_data["branches"], model, ground_truth=True)
        save_pickle(data, merged_path)
        root = create_tree(data)
        save_pickle(root, tree_path)
    else:
        raise FileNotFoundError(f"未找到原始分支文件: {branch_path}")

    return root, data
