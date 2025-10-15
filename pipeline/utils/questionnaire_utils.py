# -*- coding: utf-8 -*-
"""
问卷→点亮：保留 process_questionnaire 签名与“25题×4选1点亮”的主逻辑。
"""
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

def process_questionnaire(tree_root, data, ques_path, cope, load_pickle, eval_grade, lighten, christmas):
    ques_data = load_pickle(ques_path)
    ques_data = cope(ques_data, None, ground_truth=False)
    for i in range(25):
        marks = [eval_grade(tree_root, data, ques_data[4 * i + j]) for j in range(4)]
        best = int(np.argmax(marks))
        lighten(tree_root, data, ques_data[4 * i + best])
    tree_root.lighted = True
    return christmas(tree_root)
