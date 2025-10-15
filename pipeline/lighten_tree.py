import json
import pickle
from pathlib import Path
import numpy as np
from utils.cope_json import extract_brand_and_product_from_questionnaire
from utils.logging_utils import log
from utils.tree_node import  visualize_tree, load_tree_from_json, eval_grade, lighten, christmas, save_tree_to_json, TreeNode
from dotenv import load_dotenv
load_dotenv(override=True)
import os

def compare_with_branch(filename: str):

    brand, product_series, generate_questionnaire_model, task_type = extract_brand_and_product_from_questionnaire(
        filename)
    tree_root = load_tree_from_json(f"datas/tree/{brand}^{product_series}^discussion_tree.json")
    try:
        lighted_tree_file = Path(
            f"datas/tree/lighted/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^lighted_tree.json")
        ques_data_file = Path(f"datas/branch/branch_from_ques/{filename}")
        if not lighted_tree_file.exists():
            with open(ques_data_file, 'rb') as f:
                ques_data = pickle.load(f)
            log.info(f"问卷中共有 {len(ques_data)} 条数据")
            if len(ques_data) != int(os.getenv("QUE_NUM"))*4:
                log.error(f"{filename}中仅有{len(ques_data)}条问题，跳过点亮")
                return
            for i in range(int(os.getenv("QUE_NUM"))):
                branches_mark = []
                base_idx = 4 * i
                for j in range(4):
                    branch_data = ques_data[base_idx + j]
                    if branch_data=="None":
                        branch_mark=0
                    else:
                        branch_mark = eval_grade(tree_root, branch_data)
                    branches_mark.append(branch_mark)
                max_mark_index = int(np.argmax(branches_mark))
                max_mark = branches_mark[max_mark_index]
                log.info(f"无重叠情况下，第{i}个问题最终得分为{max_mark:.4f}")
                lighten(tree_root, ques_data[base_idx + max_mark_index])
            tree_root.lighted = True
            lighted_tree = christmas(tree_root)
            log.info(f"点亮树的总分为:{lighted_tree.get_total_w()}")
            save_tree_to_json(lighted_tree, str(lighted_tree_file))
        else:
            log.info(f"点亮树已存在:{brand}^{product_series}^{generate_questionnaire_model}^{task_type}")


        with open(lighted_tree_file, 'r', encoding='utf-8') as f:
            lighted_tree_json = json.load(f)
        lighted_tree = TreeNode("root").from_dict(lighted_tree_json)

        #检测是否已经生成了visual
        visualize_file=f"./datas/visualize/lighted/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^lighted_tree.pdf"
        if not os.path.exists(visualize_file):
            visualize_tree(lighted_tree, visualize_file)
            log.info(f"生成可视化:{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^lighted_tree.pdf")

    except Exception as e:
        log.error(e)

if __name__ == '__main__':
    compare_with_branch("Samsung^Galaxy Book Pro^chatgpt-4o-latest^direct^sentence_list.pkl")
