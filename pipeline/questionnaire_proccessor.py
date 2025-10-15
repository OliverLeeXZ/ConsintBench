# -*- coding: utf-8 -*-
"""
问卷和Top分支处理模块
功能：专门处理Top分支提取和问卷生成
从主程序中独立出来的模块，用于处理问卷相关的功能
实现文件级并发：每个并发线程处理一个完整的品牌文件
"""

import asyncio
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# 导入自定义模块
from branch2questionnaire import branch_to_questionnaire
from utils.tree_node import TreeNode, get_top_25_paths, load_tree_from_json
from utils.logging_utils import log

# 加载环境变量
load_dotenv(override=True)


class QuestionnaireConfig:
    """问卷处理配置类"""
    DEFAULT_MODEL = "chatgpt-4o-latest"  # 默认模型
    DEFAULT_MAX_CONCURRENT = 5  # 默认最大并发数
    DEFAULT_SEMAPHORE_LIMIT = 8  # 默认信号量限制

    # 目录配置
    DATA_DIR = "datas"
    TREE_DIR = f"{DATA_DIR}/tree"
    TOP_BRANCHES_DIR = f"{DATA_DIR}/branch/branch_from_top"
    QUESTIONNAIRE_DIR = f"{DATA_DIR}/questionnaire"


class QuestionnaireProcessor:
    """问卷和Top分支处理器"""

    def __init__(self, config: QuestionnaireConfig = None):
        self.config = config or QuestionnaireConfig()

    def ensure_parent_dir(self, file_path: str) -> None:
        """确保文件的父目录存在"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning(f"创建目录失败: {file_path} -> {e}")

    def extract_top_branches_sync(self, brand: str, product_type: str, tree_root: TreeNode) -> bool:
        """
        同步提取Top-25分支路径
        Args:
            brand: 品牌名称
            product_type: 产品类型
            tree_root: 树根节点
        Returns:
            成功返回True，失败返回False
        """
        try:
            output_dir = Path(self.config.TOP_BRANCHES_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{brand}^{product_type}^top_branches.pkl"

            if output_file.exists():
                log.info(f"[跳过] Top-{os.getenv('QUE_NUM', '25')}分支文件已存在: {output_file}")
                return True

            # 提取Top-N路径
            top_paths_with_discussion = [
                [*(node.name for node in path), getattr(path[-1], "discussion", []) or []]
                for path, _ in get_top_25_paths(tree_root)
            ]

            # 保存结果
            with open(output_file, "wb") as f:
                pickle.dump(top_paths_with_discussion, f)

            log.info(f"Top-{os.getenv('QUE_NUM', '25')}分支路径已保存: {output_file}")
            return True

        except Exception as e:
            log.error(f"提取Top-{os.getenv('QUE_NUM', '25')}分支失败: {e}")
            return False

    def load_tree_from_file(self, brand: str, product_type: str) -> Optional[TreeNode]:
        """
        从文件加载树结构
        Args:
            brand: 品牌名称
            product_type: 产品类型
        Returns:
            树根节点或None
        """
        try:
            tree_file = f"{self.config.TREE_DIR}/{brand}^{product_type}^discussion_tree.json"

            if not os.path.exists(tree_file):
                log.error(f"树文件不存在: {tree_file}")
                return None

            return load_tree_from_json(tree_file)

        except Exception as e:
            log.error(f"加载树文件失败 {brand}^{product_type}: {e}")
            return None

    def generate_questionnaire_sync(self, brand_name: str, product_type: str) -> bool:
        """
        同步生成问卷
        Args:
            brand_name: 品牌名称
            product_type: 产品类型
        Returns:
            成功返回True，失败返回False
        """
        try:
            log.info(f"开始为品牌 {brand_name} 生成问卷...")

            branch_to_questionnaire(
                product_brand=brand_name,
                product_type=product_type,
                model_name=self.config.DEFAULT_MODEL
            )

            log.info(f"品牌 {brand_name} 问卷生成完成")
            return True

        except Exception as e:
            log.error(f"问卷生成失败 {brand_name}: {e}")
            return False

    def process_single_brand_file_sync(
            self,
            brand_name: str,
            product_type: str,
            tree_root: Optional[TreeNode] = None
    ) -> bool:
        """
        同步处理单个品牌文件的完整流程（Top分支提取 + 问卷生成）
        这个函数将在单个线程中顺序执行两个步骤

        Args:
            brand_name: 品牌名称
            product_type: 产品类型
            tree_root: 树根节点，如果不提供会尝试从文件加载
        Returns:
            成功返回True，失败返回False
        """
        thread_name = threading.current_thread().name
        log.info(f"[线程 {thread_name}] 开始处理品牌文件 {brand_name}")

        try:
            # 如果没有提供树根节点，尝试从文件加载
            if tree_root is None:
                tree_root = self.load_tree_from_file(brand_name, product_type)
                if tree_root is None:
                    log.error(f"[线程 {thread_name}] 无法获取品牌 {brand_name} 的树结构")
                    return False

            # 步骤1: 提取Top分支
            log.info(f"[线程 {thread_name}] 开始提取品牌 {brand_name} 的Top分支")
            top_branches_success = self.extract_top_branches_sync(brand_name, product_type, tree_root)

            if not top_branches_success:
                log.error(f"[线程 {thread_name}] 品牌 {brand_name} Top分支提取失败")
                return False

            log.info(f"[线程 {thread_name}] 品牌 {brand_name} Top分支提取完成")

            # 步骤2: 生成问卷
            log.info(f"[线程 {thread_name}] 开始生成品牌 {brand_name} 的问卷")
            questionnaire_success = self.generate_questionnaire_sync(brand_name, product_type)

            if not questionnaire_success:
                log.error(f"[线程 {thread_name}] 品牌 {brand_name} 问卷生成失败")
                return False

            log.info(f"[线程 {thread_name}] 品牌 {brand_name} 问卷生成完成")

            log.info(f"[线程 {thread_name}] 品牌文件 {brand_name} 处理完成")
            return True

        except Exception as e:
            log.error(f"[线程 {thread_name}] 处理品牌文件 {brand_name} 失败: {e}")
            return False

    async def process_single_brand_file_async(
            self,
            brand_name: str,
            product_type: str,
            tree_root: Optional[TreeNode] = None
    ) -> bool:
        """
        异步处理单个品牌文件的完整流程
        将同步处理放在线程池中执行

        Args:
            brand_name: 品牌名称
            product_type: 产品类型
            tree_root: 树根节点，可选
        Returns:
            成功返回True，失败返回False
        """
        return await asyncio.to_thread(
            self.process_single_brand_file_sync,
            brand_name,
            product_type,
            tree_root
        )

    async def process_multiple_brand_files_concurrent(
            self,
            brand_data_list: List[Dict[str, str]],
            max_concurrent: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        文件级并发处理多个品牌文件
        每个并发线程处理一个完整的品牌文件（Top分支 + 问卷）

        Args:
            brand_data_list: 品牌数据列表，每个元素包含brand和product_type
            max_concurrent: 最大并发数，如果不指定则使用配置默认值
        Returns:
            品牌名称到处理结果的映射
        """
        if max_concurrent is None:
            max_concurrent = self.config.DEFAULT_MAX_CONCURRENT

        log.info(f"开始文件级并发处理 {len(brand_data_list)} 个品牌文件，最大并发数: {max_concurrent}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_file_with_semaphore(brand_data):
            async with semaphore:
                brand_name = brand_data['brand']
                product_type = brand_data['product_type']

                try:
                    success = await self.process_single_brand_file_async(brand_name, product_type)

                    if success:
                        log.info(f"品牌文件 {brand_name}_{product_type} 处理成功")
                    else:
                        log.warning(f"品牌文件 {brand_name}_{product_type} 处理失败")

                    return brand_name+"_"+product_type, success

                except Exception as e:
                    log.error(f"处理品牌文件  {brand_name}_{product_type} 时发生异常: {e}")
                    return  brand_name+"_"+product_type, False

        # 创建所有任务
        tasks = [
            asyncio.create_task(
                process_single_file_with_semaphore(brand_data),
                name=f"brand_file_{brand_data['brand']}"
            )
            for brand_data in brand_data_list
        ]

        # 等待所有任务完成
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        results = {}
        for i, task_result in enumerate(task_results):
            if isinstance(task_result, Exception):
                log.error(f"任务 {tasks[i].get_name()} 执行异常: {task_result}")
                # 尝试从任务名称中提取品牌名
                try:
                    brand_name = tasks[i].get_name().replace("brand_file_", "")
                    results[brand_name] = False
                except:
                    pass
                continue

            brand_name, success = task_result
            results[brand_name] = success

        success_count = sum(1 for success in results.values() if success)
        log.info(f"文件级并发处理完成，成功: {success_count}/{len(brand_data_list)}")

        return results

    def get_existing_tree_brands(self) -> List[Dict[str, str]]:
        """
        获取已存在树文件的品牌列表
        Returns:
            品牌信息列表
        """
        brands = []

        if not os.path.exists(self.config.TREE_DIR):
            log.warning(f"树目录不存在: {self.config.TREE_DIR}")
            return brands

        tree_files = [f for f in os.listdir(self.config.TREE_DIR) if f.endswith('discussion_tree.json')]

        for tree_file in tree_files:
            try:
                # 解析文件名: brand^product_type^discussion_tree.json
                base_name = os.path.splitext(tree_file)[0]  # 移除.json
                parts = base_name.split('^')

                if len(parts) >= 2:
                    brand_info = {
                        'brand': parts[0],
                        'product_type': parts[1],
                        'tree_file': tree_file
                    }
                    brands.append(brand_info)
                else:
                    log.warning(f"跳过格式异常的树文件: {tree_file}")

            except Exception as e:
                log.error(f"解析树文件名失败 {tree_file}: {e}")
                continue

        log.info(f"发现 {len(brands)} 个已存在的树文件")
        return brands

    async def process_all_existing_tree_files(self, max_concurrent: Optional[int] = None) -> Dict[str, bool]:
        """
        文件级并发处理所有已存在树文件的品牌
        Args:
            max_concurrent: 最大并发数，如果不指定则使用配置默认值
        Returns:
            处理结果映射
        """
        log.info("开始文件级并发处理所有已存在树文件的品牌")

        brands = self.get_existing_tree_brands()
        if not brands:
            log.info("没有找到已存在的树文件")
            return {}

        return await self.process_multiple_brand_files_concurrent(brands, max_concurrent)

    def check_files_status(self, brand_data_list: List[Dict[str, str]]) -> Dict[str, Dict[str, bool]]:
        """
        检查品牌文件的处理状态
        Args:
            brand_data_list: 品牌数据列表
        Returns:
            品牌文件状态映射
        """
        status_map = {}

        for brand_data in brand_data_list:
            brand = brand_data['brand']
            product_type = brand_data['product_type']

            # 检查top分支文件
            top_file = Path(self.config.TOP_BRANCHES_DIR) / f"{brand}^{product_type}^top_branches.pkl"
            top_exists = top_file.exists()

            # 检查问卷文件 (这里假设问卷文件的命名规则，可能需要根据实际情况调整)
            questionnaire_file = Path(self.config.QUESTIONNAIRE_DIR) / f"{brand}^{product_type}_questionnaire.json"
            questionnaire_exists = questionnaire_file.exists()

            status_map[brand] = {
                'top_branches': top_exists,
                'questionnaire': questionnaire_exists,
                'completed': top_exists and questionnaire_exists
            }

        return status_map

    def get_incomplete_brands(self) -> List[Dict[str, str]]:
        """
        获取需要处理的不完整品牌列表
        Returns:
            需要处理的品牌信息列表
        """
        all_brands = self.get_existing_tree_brands()
        if not all_brands:
            return []

        status_map = self.check_files_status(all_brands)

        incomplete_brands = []
        for brand_data in all_brands:
            brand = brand_data['brand']
            if not status_map[brand]['completed']:
                incomplete_brands.append(brand_data)

        log.info(f"发现 {len(incomplete_brands)} 个需要处理的不完整品牌文件")
        return incomplete_brands


# ========= 便捷使用函数 =========
async def process_single_brand_file(
        brand_name: str,
        product_type: str,
        tree_root: Optional[TreeNode] = None,
        config: QuestionnaireConfig = None
) -> bool:
    """
    便捷函数：处理单个品牌文件的完整工作流程

    Args:
        brand_name: 品牌名称
        product_type: 产品类型
        tree_root: 树根节点，可选
        config: 配置对象，可选
    Returns:
        处理是否成功
    """
    processor = QuestionnaireProcessor(config)
    return await processor.process_single_brand_file_async(brand_name, product_type, tree_root)


async def process_all_brand_files_concurrent(
        config: QuestionnaireConfig = None,
        max_concurrent: Optional[int] = None
) -> Dict[str, bool]:
    """
    便捷函数：文件级并发处理所有品牌文件

    Args:
        config: 配置对象，可选
        max_concurrent: 最大并发数，可选
    Returns:
        品牌名称到处理结果的映射
    """
    processor = QuestionnaireProcessor(config)
    return await processor.process_all_existing_tree_files(max_concurrent)


def extract_top_branches_sync(
        brand_name: str,
        product_type: str,
        tree_root: TreeNode,
        config: QuestionnaireConfig = None
) -> bool:
    """
    便捷函数：同步提取Top分支

    Args:
        brand_name: 品牌名称
        product_type: 产品类型
        tree_root: 树根节点
        config: 配置对象，可选
    Returns:
        提取是否成功
    """
    processor = QuestionnaireProcessor(config)
    return processor.extract_top_branches_sync(brand_name, product_type, tree_root)


def generate_questionnaire_sync(
        brand_name: str,
        product_type: str,
        config: QuestionnaireConfig = None
) -> bool:
    """
    便捷函数：同步生成问卷

    Args:
        brand_name: 品牌名称
        product_type: 产品类型
        config: 配置对象，可选
    Returns:
        生成是否成功
    """
    processor = QuestionnaireProcessor(config)
    return processor.generate_questionnaire_sync(brand_name, product_type)


# ========= 测试运行 =========
async def main():
    """测试运行函数"""
    try:
        results = await process_all_brand_files_concurrent(max_concurrent=3)
        print(f"处理完成，结果:")
        for brand_name, success in results.items():
            status = "成功" if success else "失败"
            print(f"  品牌: {brand_name} - {status}")



    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())