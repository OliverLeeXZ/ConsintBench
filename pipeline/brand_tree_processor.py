# -*- coding: utf-8 -*-
"""
品牌分支和树处理模块
功能：专门处理品牌讨论数据的分支生成和树结构构建
从主程序中独立出来的模块，用于处理品牌数据的核心流程
"""

import asyncio
import os
import json
import time
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.logging import logging_redirect_tqdm

# 导入自定义模块
from Disscussion2branch import run_discussion2branch
from utils.tree_node import (
    create_tree, save_tree_to_json, visualize_tree,
    fine_tune, prune, TreeNode
)
from utils.merge_utils import normalize_branch_terms, get_sentence_model
from utils.logging_utils import log


class BrandTreeConfig:
    """品牌树处理配置类"""
    DEFAULT_QPS = 8.0  # 默认请求频率限制
    DEFAULT_CONCURRENCY = 10  # 默认并发数
    DEFAULT_MODEL = "chatgpt-4o-latest"  # 默认模型

    # 目录配置
    DATA_DIR = "datas"
    DISCUSSION_DIR = f"{DATA_DIR}/discussion"
    BRANCH_DIR = f"{DATA_DIR}/branch/branch_from_disc"
    MERGED_DIR = f"{DATA_DIR}/merged"
    TREE_DIR = f"{DATA_DIR}/tree"
    VISUALIZE_DIR = f"{DATA_DIR}/visualize"


class GlobalLimiter:
    """全局请求频率限制器"""

    def __init__(self, qps: Optional[float]):
        """
        初始化限速器
        Args:
            qps: 每秒请求数限制，None表示无限制
        """
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        """等待以满足限速要求"""
        if self.interval <= 0:
            return

        with self.lock:
            now = time.time()
            if now < self.next_t:
                sleep_time = self.next_t - now
                time.sleep(sleep_time)
                now = time.time()
            self.next_t = now + self.interval


# ========= 线程安全的句向量模型单例 =========
_sentence_model_lock = threading.Lock()
_sentence_model_singleton = None


def safe_get_sentence_model():
    """
    线程安全地获取句向量模型
    避免多线程并发初始化导致的meta tensor问题
    """
    global _sentence_model_singleton
    if _sentence_model_singleton is not None:
        return _sentence_model_singleton

    with _sentence_model_lock:
        if _sentence_model_singleton is None:
            log.info("初始化句向量模型...")
            _sentence_model_singleton = get_sentence_model()
            log.info("句向量模型初始化完成")
    return _sentence_model_singleton


# ========= 工具函数 =========
def ensure_parent_dir(file_path: Union[str, Path]) -> None:
    """确保文件的父目录存在"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(f"创建目录失败: {file_path} -> {e}")


def load_branches_from_pickle(pkl_path: str) -> List[List[Any]]:
    """从pickle文件加载分支数据"""
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        # 处理不同的数据结构
        if isinstance(obj, dict) and "branches" in obj:
            branches = obj["branches"]
        else:
            branches = obj

        if not isinstance(branches, list):
            raise ValueError(f"文件 {pkl_path} 的branches不是list类型: {type(branches)}")

        return branches
    except Exception as e:
        log.error(f"加载分支文件失败 {pkl_path}: {e}")
        raise


def normalize_and_save_branches(
        branches: List[List[Any]],
        output_file: str,
        *,
        model=None,
        levels: int = 7,
        cosine_threshold: float = 0.8,
) -> None:
    """归一化分支数据并保存"""
    if model is None:
        model = safe_get_sentence_model()

    log.info(f"开始归一化分支数据: levels={levels}, threshold={cosine_threshold}")
    start_time = time.time()

    try:
        normalized_branches = normalize_branch_terms(branches, model=model)
        elapsed_time = time.time() - start_time
        log.info(f"归一化完成，用时 {elapsed_time:.2f} 秒，条目数: {len(normalized_branches)}")
        ensure_parent_dir(output_file)

        # 保存归一化结果
        with open(output_file, "wb") as f:
            pickle.dump(normalized_branches, f)
        log.info(f"归一化结果已保存: {output_file}")

    except Exception as e:
        log.error(f"归一化处理失败: {e}")
        raise


def build_tree_from_merged_file(merged_file: str) -> Optional[TreeNode]:
    """从归一化文件构建树结构"""
    try:
        with open(merged_file, "rb") as f:
            branches = pickle.load(f)

        if not isinstance(branches, list) or not branches:
            log.warning(f"归一化分支数据为空或类型异常: {type(branches)}")
            return None

        log.info("开始构建树结构...")
        root = create_tree(branches)
        root = prune(root)
        root = fine_tune(root)
        log.info("树结构构建完成")

        return root

    except Exception as e:
        log.error(f"从文件构建树结构失败 {merged_file}: {e}")
        return None


class BrandProcessor:
    """品牌数据处理器"""

    def __init__(self, config: BrandTreeConfig = None):
        self.config = config or BrandTreeConfig()

    def parse_brand_files(self) -> List[Dict[str, str]]:
        """解析品牌文件列表"""
        brand_list = []

        if not os.path.exists(self.config.DISCUSSION_DIR):
            log.error(f"讨论数据目录不存在: {self.config.DISCUSSION_DIR}")
            return brand_list

        file_list = os.listdir(self.config.DISCUSSION_DIR)
        log.info(f"发现 {len(file_list)} 个文件")

        for file in file_list:
            try:
                # 解析文件名格式: brand^product_type.json
                base_name = os.path.splitext(os.path.basename(file))[0]
                parts = base_name.split("^")

                if len(parts) >= 2:
                    brand_info = {
                        "brand": parts[0],
                        "product_type": parts[1],
                        "filename": file
                    }
                    brand_list.append(brand_info)
                else:
                    log.warning(f"跳过格式异常的文件: {file}")

            except Exception as e:
                log.error(f"处理文件名时出错 {file}: {e}")
                continue

        log.info(f"成功解析 {len(brand_list)} 个品牌文件")
        return brand_list

    def process_single_brand(
            self,
            brand_info: Dict[str, str],
            model_name: str,
            *,
            progress_position: Optional[int] = None,
            external_limiter: Optional[GlobalLimiter] = None
    ) -> Optional[str]:
        """处理单个品牌的分支生成和归一化"""
        brand = brand_info['brand']
        product_type = brand_info['product_type']

        merged_file = f"{self.config.MERGED_DIR}/{brand}^{product_type}^discussion_branch_merged.pkl"
        output_file = f"{self.config.BRANCH_DIR}/{brand}^{product_type}^discussion_branch.pkl"

        if os.path.exists(merged_file):
            log.info(f"[跳过] 合并文件已存在: {merged_file}")
            return merged_file

        if os.path.exists(output_file):
            log.info(f"发现现有分支文件: {output_file}")
            try:
                branches = load_branches_from_pickle(output_file)
                log.info(f"分支条目数: {len(branches)}")

                model = safe_get_sentence_model()
                normalize_and_save_branches(branches, merged_file, model=model)
                return merged_file

            except Exception as e:
                log.error(f"处理现有分支文件失败: {e}")
                return None

        log.info(f"开始生成分支文件: {brand}^{product_type}")
        start_time = time.time()
        try:
            discussion_file = f"{self.config.DISCUSSION_DIR}/{brand}^{product_type}.json"

            result_info = run_discussion2branch(
                round_start=0,
                round_end=726,
                discussion_json_filename=discussion_file,
                brand=brand,
                product_series=product_type,
                model_name=model_name,
                max_workers=50,
                external_rate_limiter=external_limiter,
                show_progress=True,
                progress_position=progress_position,
                verbose=False
            )

            elapsed_time = time.time() - start_time
            log.info(f"分支生成完成，用时 {elapsed_time:.2f} 秒")
            log.debug(f"生成结果: {json.dumps(result_info, ensure_ascii=False, indent=2)}")

        except Exception as e:
            log.error(f"生成分支失败: {e}")
            return None

        # 检查生成的文件
        if not os.path.exists(output_file):
            log.warning(f"生成后未找到文件: {output_file}")
            return None

        # 加载并归一化
        try:
            branches = load_branches_from_pickle(output_file)
            log.info(f"生成的分支条目数: {len(branches)}")

            model = safe_get_sentence_model()
            normalize_and_save_branches(branches, merged_file, model=model)
            return merged_file

        except Exception as e:
            log.error(f"处理生成的分支文件失败: {e}")
            return None

    def process_tree_construction(self, brand: str, product_type: str, merged_file: str) -> Optional[TreeNode]:
        """
        处理树结构构建
        Args:
            brand: 品牌名称
            product_type: 产品类型
            merged_file: 合并文件路径
        Returns:
            构建成功返回树根节点，否则返回None
        """
        tree_file = f"{self.config.TREE_DIR}/{brand}^{product_type}^discussion_tree.json"
        visualize_file = f"{self.config.VISUALIZE_DIR}/{brand}^{product_type}^discussion_tree.pdf"

        # 检查文件是否已存在
        if os.path.exists(tree_file) and os.path.exists(visualize_file):
            log.info(f"[跳过] 树文件和可视化文件已存在: {brand}")
            return build_tree_from_merged_file(merged_file)

        # 只有树文件存在，需要补充可视化
        if os.path.exists(tree_file) and not os.path.exists(visualize_file):
            log.info(f"树文件已存在，开始生成可视化: {visualize_file}")
            root = build_tree_from_merged_file(merged_file)
            if root is None:
                log.warning("无法重建树结构，跳过可视化")
                return None

            ensure_parent_dir(visualize_file)
            try:
                visualize_tree(root, visualize_file)
                log.info(f"树可视化已保存: {visualize_file}")
            except Exception as e:
                log.warning(f"可视化生成失败: {e}")
            return root

        # 检查合并文件是否存在
        if not os.path.exists(merged_file):
            log.error(f"合并文件不存在: {merged_file}")
            return None

        # 构建新的树结构
        log.info(f"开始构建树结构: {brand}")
        start_time = time.time()

        root = build_tree_from_merged_file(merged_file)
        if root is None:
            return None

        # 保存树结构JSON
        ensure_parent_dir(tree_file)
        try:
            save_tree_to_json(root, tree_file)
            log.info(f"树JSON已保存: {tree_file}")
        except Exception as e:
            log.error(f"保存树JSON失败: {e}")
            return None

        # 生成可视化
        ensure_parent_dir(visualize_file)
        try:
            visualize_tree(root, visualize_file)
            log.info(f"树可视化已保存: {visualize_file}")
        except Exception as e:
            log.warning(f"可视化生成失败: {e}")

        elapsed_time = time.time() - start_time
        log.info(f"树构建完成，用时 {elapsed_time:.2f} 秒")
        return root


class BrandTreeProcessor:
    """品牌树处理主控制器"""

    def __init__(self, config: BrandTreeConfig = None):
        self.config = config or BrandTreeConfig()
        self.brand_processor = BrandProcessor(self.config)

    async def run_branch_processing(self) -> Dict[str, Dict[str, str]]:
        """运行分支处理流程"""
        log.info("=== 开始分支处理阶段 ===")
        brand_list = self.brand_processor.parse_brand_files()
        if not brand_list:
            log.warning("未找到有效的品牌文件")
            return {}

        total = len(brand_list)
        log.info(f"准备处理 {total} 个品牌")

        concurrency = min(total, self.config.DEFAULT_CONCURRENCY)
        limiter = GlobalLimiter(self.config.DEFAULT_QPS)

        brand_to_merged = {}
        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_brand = {}
                for i, brand_info in enumerate(brand_list):
                    brand_name = brand_info['brand']
                    log.info(f"=== [{i + 1}/{total}] 提交任务: {brand_name} ===")
                    future = executor.submit(
                        self.brand_processor.process_single_brand,
                        brand_info,
                        self.config.DEFAULT_MODEL,
                        progress_position=i,
                        external_limiter=limiter
                    )
                    future_to_brand[future] = brand_info

                completed = 0
                for future in as_completed(future_to_brand):
                    brand_info = future_to_brand[future]
                    brand_name = brand_info['brand']
                    completed += 1
                    try:
                        merged_path = future.result()
                        if merged_path:
                            brand_to_merged[brand_name] = {
                                'merged_path': merged_path,
                                'product_type': brand_info['product_type']
                            }
                            log.info(f"[{completed}/{total}] 品牌 {brand_name} 处理完成: {merged_path}")
                        else:
                            log.error(f"[{completed}/{total}] 品牌 {brand_name} 处理失败")
                    except Exception as e:
                        log.error(f"[{completed}/{total}] 品牌 {brand_name} 处理异常: {e}")

        log.info(f"分支处理完成，成功处理 {len(brand_to_merged)}/{total} 个品牌")
        return brand_to_merged

    async def run_tree_processing(self, brand_to_merged: Dict[str, Dict[str, str]]) -> None:
        """运行树处理流程"""
        log.info("=== 开始树处理阶段 ===")

        for brand_name, brand_data in brand_to_merged.items():
            merged_path = brand_data['merged_path']
            product_type = brand_data['product_type']

            log.info(f"处理品牌树结构: {brand_name}")

            tree_root = self.brand_processor.process_tree_construction(brand_name, product_type, merged_path)
            if tree_root:
                # 提取Top-N分支
                self.brand_processor.extract_top_branches(brand_name, product_type, tree_root)

                # 生成问卷
                try:
                    await self._generate_questionnaire_async(brand_name, product_type)
                    log.info(f"品牌 {brand_name} TOP问卷生成完成")
                except Exception as e:
                    log.error(f"品牌 {brand_name} TOP问卷生成失败: {e}")
            else:
                log.warning(f"品牌 {brand_name} 树构建失败")

    async def process_brand_and_trees(self) -> Dict[str, Dict[str, str]]:
        """
        处理品牌分支和树结构的完整流程
        这是从原main()中提取出来的核心功能

        Returns:
            品牌到合并文件的映射字典
        """
        log.info("========== 开始品牌分支和树处理流程 ==========")

        try:
            # 运行分支处理
            brand_to_merged = await self.run_branch_processing()

            # 如果有成功处理的品牌，继续树处理
            if brand_to_merged:
                await self.run_tree_processing(brand_to_merged)
                log.info("品牌分支和树处理流程完成")
            else:
                log.warning("没有成功处理的品牌，跳过树处理")

            return brand_to_merged

        except Exception as e:
            log.error(f"品牌分支和树处理流程失败: {e}")
            raise


# ========= 便捷使用函数 =========
async def process_brands_and_trees(config: BrandTreeConfig = None) -> Dict[str, Dict[str, str]]:
    """
    便捷函数：处理品牌分支和树结构

    Args:
        config: 配置对象，如果不提供则使用默认配置

    Returns:
        品牌到合并文件的映射字典
    """
    processor = BrandTreeProcessor(config)
    return await processor.process_brand_and_trees()


# ========= 测试运行 =========
async def main():
    """测试运行函数"""
    try:
        result = await process_brands_and_trees()
        print(f"处理完成，成功处理 {len(result)} 个品牌")
        for brand_name, info in result.items():
            print(f"品牌: {brand_name}, 产品类型: {info['product_type']}, 文件: {info['merged_path']}")
    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())