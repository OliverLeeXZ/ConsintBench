import openai
from openai import OpenAI
import os
from dotenv import load_dotenv  # 用于加载环境变量，需要安装python-dotenv包
load_dotenv(override=True)
import os



def test_openai_connection(api_key=None):
    """
    测试与OpenAI API的连接情况
    
    参数:
        api_key: OpenAI API密钥，如果为None则尝试从环境变量加载
    
    返回:
        字典，包含连接状态和相关信息
    """
    try:
        # 加载环境变量（如果有）
        load_dotenv()
        
        # 确定API密钥
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            # 尝试从环境变量获取
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {
                    "status": "error",
                    "message": "未提供API密钥，请通过参数或环境变量OPENAI_API_KEY设置"
                }
            client = OpenAI()
        
        # 发送一个简单的请求测试连接
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请返回'连接成功'以确认API可正常访问"}
            ],
            max_tokens=10
        )
        
        # 检查响应
        if response.choices and response.choices[0].message.content.strip() == "连接成功":
            return {
                "status": "success",
                "message": "与OpenAI API连接成功",
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            return {
                "status": "warning",
                "message": "连接成功但响应不符合预期",
                "response": response.model_dump()
            }
    
    except openai.AuthenticationError:
        return {
            "status": "error",
            "message": "认证失败，请检查API密钥是否正确"
        }
    except openai.APIConnectionError:
        return {
            "status": "error",
            "message": "无法连接到OpenAI API，请检查网络连接"
        }
    except openai.RateLimitError:
        return {
            "status": "error",
            "message": "已达到API速率限制，请稍后再试"
        }
    except openai.APIError as e:
        return {
            "status": "error",
            "message": f"API错误: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"发生未知错误: {str(e)}"
        }

# 示例用法
if __name__ == "__main__":
    # 可以直接调用（使用环境变量中的API密钥）
    result = test_openai_connection()
    
    # 或者指定API密钥
    # result = test_openai_connection(api_key="your_api_key_here")
    
    print(f"状态: {result['status']}")
    print(f"消息: {result['message']}")
    
    # 打印更多详细信息（如果有）
    if result["status"] == "success":
        print(f"使用模型: {result['model']}")
        print(f"使用情况: {result['usage']}")
    elif "response" in result:
        print(f"响应详情: {result['response']}")
