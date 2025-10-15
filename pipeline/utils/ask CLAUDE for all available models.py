
from dotenv import load_dotenv
load_dotenv(override=True)

import requests

def get_claude_models(api_key, base_url="https://api.anthropic.com"):
    """
    获取Claude可用的模型列表
    
    参数:
        api_key: 你的Anthropic API密钥
        base_url: Claude API的基础URL
    
    返回:
        模型列表或错误信息
    """
    # API端点 - 用于列出可用模型
    url = f"{base_url}/v1/models"
    
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"  # 指定API版本
    }
    
    try:
        # 发送GET请求
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 如果响应状态码不是200，会抛出异常
        
        # 解析JSON响应
        data = response.json()
        
        # 返回模型列表
        if "data" in data:
            return [model["id"] for model in data["data"]]
        else:
            return "未找到模型数据"
            
    except requests.exceptions.RequestException as e:
        return f"请求出错: {str(e)}"
    except Exception as e:
        return f"处理响应时出错: {str(e)}"

# 使用示例
if __name__ == "__main__":
    # 替换为你的API密钥
    api_key = os.getenv("CLAUDE_API_KEY", "")
          
    # 获取模型列表
    models = get_claude_models(api_key)
    
    # 打印结果
    if isinstance(models, list):
        print("可用的Claude模型:")
        for model in models:
            print(f"- {model}")
    else:
        print(models)

