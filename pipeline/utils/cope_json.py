import os
import json
from .logging_utils import log


def safe_read(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log.info(f"Successfully loaded file: {file_path}, 读取后类型为{type(data)}")
        return data
    except FileNotFoundError:
        log.error(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError as e:
        log.error(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
    return None


def safe_read_json_str(content):
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        log.error(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
    return None


def get_json_filenames(discussion_path):
    """获取指定目录下所有.json文件的文件名列表"""
    json_files = []
    if os.path.exists(discussion_path) and os.path.isdir(discussion_path):
        for filename in os.listdir(discussion_path):
            if filename.endswith('.json'):
                json_files.append(filename)
    return json_files


def extract_brand_and_product(filename):
    """
    从JSON文件名中提取品牌和产品系列
    文件名格式：{Brand}^{Product_series}.json
    """
    # 去除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 按照^符号分割
    parts = name_without_ext.split('^')

    # 验证格式是否正确
    if len(parts) == 2:
        brand = parts[0]
        product_series = parts[1]
        return brand, product_series
    else:
        # 格式不正确时返回None
        log.error(f"警告：文件名 '{filename}' 格式不符合要求")
        return None, None



def extract_brand_and_product_from_questionnaire(filename):
    """
    从JSON文件名中提取品牌和产品系列
    文件名格式：{Brand}^{Product_series}^{...}.json
    """
    # 去除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 按照^符号分割
    parts = name_without_ext.split('^')

    # 验证格式是否正确
    if len(parts) == 5:
        brand = parts[0]
        product_series = parts[1]
        model=parts[2]
        task_type=parts[3]
        return brand, product_series,model, task_type
    else:
        # 格式不正确时返回None
        log.error(f"警告：文件名 '{filename}' 格式不符合要求")
        return None, None




