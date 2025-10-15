import os
import pickle
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_pickle(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, file_path):
    ensure_dir(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def generate_timestamp() -> str:
    return datetime.now().strftime("%m月_%d日_%H时_%M分")

def get_path(category: str, filename: str):
    base_dirs = {
        "discussion": "datas/discussion",
        "merged": "datas/merged",
        "tree": "datas/tree",
        "visual": "datas/visualize",
        "questionnaire": "datas/questionnaire",
        "comparison": "datas/comparison",
        "discussion_branch": "datas/branch/branch_from_disc",
        "questionnaire_branch": "datas/branch/branch_from_ques",
    }
    if category not in base_dirs:
        raise ValueError(f"未知类别: {category}")
    return os.path.join(base_dirs[category], filename)
