# -*- coding: utf-8 -*-
"""
树结构、构建、评估、点亮、剪枝、微调、可视化等。
仅读取 .env；Graphviz 可按你系统 PATH 配置（Windows 建议把 Graphviz/bin 加入 PATH）。
"""
import os
import json
import sys
import threading
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from graphviz import Digraph
from sentence_transformers import SentenceTransformer, util

load_dotenv(override=True)


# 线程安全的模型单例类
class ModelSingleton:
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer('paraphrase-mpnet-base-v2')
        return self._model


# 全局模型实例
_model_singleton = ModelSingleton()


def _get_model():
    """保持原有接口不变，内部使用线程安全的单例"""
    return _model_singleton.get_model()


DEFAULT_TREE_JSON = "datas/tree/tree_data.json"
DEFAULT_TREE_IMG = "datas/visualize/tree_visual.pdf"


class TreeNode:
    def __init__(self, name: str, w: float = 0.0, discussion: str = ""):
        self.name = name
        self.w = w
        self.children = []
        self.parent = None
        self.discussion = []
        self.lighted = False
        self.depth = 0
        if discussion:
            self.discussion.append(discussion)

    def get_total_w(self) -> float:
        """修复递归逻辑错误"""
        total_w = self.w
        for child in self.children:
            total_w += child.get_total_w()
        return total_w

    def add_child(self, child: "TreeNode"):
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "w": self.w,
            "discussion": self.discussion,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children]
        }

    @staticmethod
    def from_dict(data: dict, parent: "TreeNode" = None) -> "TreeNode":
        node = TreeNode(data["name"], data.get("w", 0.0))
        node.discussion = data.get("discussion", [])
        node.depth = data.get("depth", 0)
        node.parent = parent
        for child_data in data.get("children", []):
            node.add_child(TreeNode.from_dict(child_data, parent=node))
        return node

    def get_avg_leaf_depth(self) -> float:
        total_d, count = self._dfs_depth(0)
        return total_d / count if count else 0.0

    def _dfs_depth(self, depth: int) -> Tuple[float, int]:
        if not self.children:
            return depth, 1
        total_d, total_n = 0.0, 0
        for c in self.children:
            d, n = c._dfs_depth(depth + 1)
            total_d += d
            total_n += n
        return total_d, total_n

    def get_weight_topic(self, max_depth: int = 8) -> List[float]:
        weights = [0.0] * max_depth

        def dfs(node: "TreeNode", depth: int):
            if depth >= max_depth:
                return
            weights[depth] += node.w
            for c in node.children:
                dfs(c, depth + 1)

        dfs(self, 0)
        return weights


# ---------- 构建/保存/加载 ----------
def create_tree(sentence_list: list) -> TreeNode:
    root = TreeNode("root")
    if not hasattr(root, "depth"):
        root.depth = 0

    for raw_path in sentence_list:
        if not raw_path or not isinstance(raw_path, (list, tuple)):
            continue

        path = list(raw_path)

        # 1) 拆分名称链与元信息
        names: List[str] = []
        meta: List[object] = []
        entered_meta = False
        for x in path:
            if not entered_meta and isinstance(x, str):
                val = x.strip()
                if val:
                    names.append(val)
                else:
                    continue
            else:
                entered_meta = True
                meta.append(x)
        if not names:
            continue

        discussion_text: str = ""
        weight_val: Optional[float] = None
        for x in meta:
            if discussion_text == "" and isinstance(x, str):
                discussion_text = x.strip()
            elif weight_val is None and isinstance(x, (int, float)):
                try:
                    weight_val = float(x)
                except Exception:
                    pass
            if discussion_text and (weight_val is not None):
                break

        pos = root
        for depth_idx, name in enumerate(names, start=1):
            found = next((c for c in getattr(pos, "children", []) if c.name == name), None)
            if not found:
                new_node = TreeNode(name)
                new_node.depth = depth_idx
                pos.add_child(new_node)
                pos = new_node
            else:
                if not hasattr(found, "depth") or found.depth != depth_idx:
                    found.depth = depth_idx
                pos = found

        if discussion_text:
            if not hasattr(pos, "discussion"):
                pos.discussion = []
            if discussion_text not in pos.discussion:
                pos.discussion.append(discussion_text)
            p = pos.parent
            while p is not None:
                if not hasattr(p, "discussion"):
                    p.discussion = []
                if discussion_text not in p.discussion:
                    p.discussion.append(discussion_text)
                p = p.parent

        if weight_val is not None:
            if not hasattr(pos, "w"):
                pos.w = 0.0
            pos.w = round(float(pos.w) + weight_val, 6)
            p = pos.parent
            while p is not None:
                if not hasattr(p, "w"):
                    p.w = 0.0
                p.w = round(float(p.w) + weight_val, 6)
                p = p.parent
    return root


def save_tree_to_json(root: TreeNode, file_path: str = DEFAULT_TREE_JSON):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(root.to_dict(), f, ensure_ascii=False, indent=2)


def load_tree_from_json(file_path: str = DEFAULT_TREE_JSON) -> TreeNode:
    if not os.path.exists(file_path):
        return TreeNode("root")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TreeNode.from_dict(data)


# ---------- 评估/点亮 ----------
def eval_grade(node: "TreeNode", branch: List[str]) -> float:
    current = node
    branch_mark = 0.0
    branch_index = 0
    while branch_index < len(branch):
        if not current.children:
            break
        children_names = [child.name for child in current.children]
        leaf_word = branch[branch_index]
        chosen_name = choose_node(leaf_word, children_names, branch_index)
        if not chosen_name:
            break
        for child in current.children:
            if child.name == chosen_name:
                current = child
                if not child.lighted:
                    branch_mark += child.w
                break
        branch_index += 1
    return branch_mark


def choose_node(leaf: str, layer_words: List[str], branch_index: int) -> str:
    """
    动态阈值选择相似词（前三层严格，后续放宽）
    现在线程安全，支持并发调用
    """
    if not layer_words:
        return ""

    threshold = 0.65 if branch_index >= 3 else 0.75
    model = _get_model()

    try:
        embed_leaf = model.encode(leaf, convert_to_tensor=True)
        embeddings = model.encode(layer_words, convert_to_tensor=True)
        sims = util.cos_sim(embed_leaf, embeddings)[0]
        best_idx = int(sims.argmax().item())
        best_score = float(sims[best_idx].item())
        return layer_words[best_idx] if best_score >= threshold else ""
    except Exception:
        # 如果出现任何异常，返回空字符串
        return ""


def lighten(node: TreeNode, branch: list):
    choose_branch = []
    current_node = node
    current_node_children_words = [child.name for child in current_node.children]
    branch_index = 0
    while current_node_children_words and branch_index < len(branch):
        leaf = branch[branch_index]
        layer_words = current_node_children_words
        choose_leaf = choose_node(leaf, layer_words, branch_index)
        if choose_leaf != '':
            choose_branch.append(choose_leaf)
            for child in current_node.children:
                if child.name == choose_leaf:
                    current_node = child
                    child.lighted = True
                    break
            current_node_children_words = [child.name for child in current_node.children]
            branch_index += 1
        else:
            break

    print(f"本轮点亮的Branch为:{ choose_branch}")
        
    return choose_branch


def christmas(original_root: TreeNode) -> Optional[TreeNode]:
    lighted_nodes: List[TreeNode] = []

    def collect(node: TreeNode):
        if node.lighted:
            lighted_nodes.append(node)
        for c in node.children:
            collect(c)

    collect(original_root)
    if not lighted_nodes:
        return None

    def get_depth(n: TreeNode) -> int:
        d = 0
        p = n.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

    root0 = min(lighted_nodes, key=get_depth)

    def build_subtree(orig: TreeNode) -> TreeNode:
        n = TreeNode(orig.name, orig.w, "")
        n.lighted = orig.lighted
        for c in orig.children:
            if c.lighted:
                n.add_child(build_subtree(c))
        return n

    return build_subtree(root0)


# ---------- 微调/剪枝/路径 ----------
def fine_tune(root: TreeNode) -> TreeNode:
    if root.name == "whole":
        root.w = 0.0
    for c in root.children:
        fine_tune(c)
    return root


def prune(root: TreeNode) -> TreeNode:
    """
    删除深度 7 上名字为 None 的节点；
    删除深度 6 且无子节点的 None 节点
    """

    def _prune(n: TreeNode):
        # 先递归处理子节点
        for c in list(n.children):
            _prune(c)

        # 删除符合条件的节点
        children_to_remove = []
        for child in n.children:
            if ((child.depth == 7 and str(child.name).lower() == "none") or
                    (child.depth == 6 and str(child.name).lower() == "none" and not child.children)):
                children_to_remove.append(child)

        for child in children_to_remove:
            n.children.remove(child)

    _prune(root)
    return root


def get_top_25_paths(root: TreeNode, top_k: int = int(os.getenv('QUE_NUM'))) -> List[Tuple[List[TreeNode], float]]:
    results: List[Tuple[List[TreeNode], float]] = []
    visited_ids = set()

    def dfs(n: TreeNode, acc: float, path: List[TreeNode], best: dict):
        new_acc = acc + (0.0 if id(n) in visited_ids else float(n.w))
        path.append(n)
        if not n.children:
            if new_acc > best["w"]:
                best["w"] = new_acc
                best["path"] = path.copy()
        else:
            for c in n.children:
                dfs(c, new_acc, path, best)
        path.pop()

    for _ in range(top_k):
        best = {"w": float("-inf"), "path": []}
        dfs(root, 0.0, [], best)
        if not best["path"]:
            break
        results.append((best["path"], best["w"]))
        for node in best["path"]:
            visited_ids.add(id(node))
    #print(results)
    #exit()
    return results


# ---------- 可视化 ----------
def visualize_tree(root: TreeNode, output_filename: str = DEFAULT_TREE_IMG):
    dot = Digraph(format="pdf")
    dot.attr('graph',  # 图属性
             fontname='Microsoft YaHei',  # 微软雅黑
             rankdir='TB',  # 图形布局从上到下
             bgcolor='white',
             splines='curved',  # 曲线连接节点
             nodesep='0.5',  # 同一等级节点之间距离
             ranksep='0.8',  # 不同等级节点距离
             concentrate='false')  # 不合并边，每条边单独显示
    dot.attr('node',  # 节点属性
             shape='box',
             style='rounded,filled,setlinewidth(1.5)',
             color='black',
             fillcolor='lightblue:lightcyan',
             gradientangle='90',
             fontname='Microsoft YaHei',
             fontsize='14',
             penwidth='2')
    dot.attr('edge',
             color='black',
             arrowsize='0.8',
             penwidth='1.5',
             fontname='Microsoft YaHei')

    def _trav(n: TreeNode):
        nid = str(id(n))
        dot.node(nid, f"{n.name} ({n.w:.2f})")
        for c in n.children:
            cid = str(id(c))
            _trav(c)
            dot.edge(nid, cid)

    _trav(root)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    dot.render(filename=output_filename.rsplit('.', 1)[0], cleanup=True)
