from dotenv import load_dotenv
from .file_io import ensure_dir
from .tree_node import visualize_tree as _vis

load_dotenv(override=True)

def visualize_tree(tree_root, save_path: str):
    ensure_dir(save_path)
    _vis(tree_root, save_path)
