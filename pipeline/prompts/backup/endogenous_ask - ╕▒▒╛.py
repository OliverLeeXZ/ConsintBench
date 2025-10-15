import asyncio
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)

from typing import Any, Dict, List, Optional


import json
def safe_read_json_str(content):
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
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
        print(f"警告：文件名 '{filename}' 格式不符合要求")
        return None, None





from tenacity import retry, stop_after_attempt, wait_exponential
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
async def safe_agent_invoke(agent, prompt, timeout=90):
    """带重试和超时的安全调用函数"""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        log.error(f"调用超时 ({timeout}秒), 正在重试")
        raise  # 触发重试
    except Exception as e:
        log.error(f"调用发生错误:{e}, 正在重试...")
        raise



from langgraph.prebuilt import create_react_agent
# -------------------------------Invoke LLM---------------------------

import yaml
from jinja2 import Environment, StrictUndefined

jinja_env = Environment(undefined=StrictUndefined)

async def LLM_ask(brand, product_series, model_name, washer_model_name):
    model = init_chat_model(model_name, model_provider="openai")

    with open(f"prompts/llm_direct_create_question.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 系统提示词的template
    template = jinja_env.from_string(config["system_prompt"])
    sys_msg = template.render(**{'brand': brand, "product_series":product_series})

    # 用户提示词的template
    template = jinja_env.from_string(config["template"])  # 用户逐步输入内容
    ques_generate_prompt = template.render(**{'brand': brand, "product_series":product_series})

    agent = create_react_agent(
        model=model,  tools=[],   prompt=sys_msg,  )


    result = await safe_agent_invoke(agent, ques_generate_prompt)

    #print("最原始返回内容")
    #print(result)

    content = result["messages"][1].content

    # 第一种解决方案
    content = content.strip('` \n')

    if content.startswith('json'):
        content = content[4:] 
    

    #第二种解决方案，给大模型加入提示词->Do not wrap the json codes in JSON markers

    print(content)

    #尝试用json格式来解析该文本，如果解析失败，则进行一次清洗
    try:
        questionnaire=safe_read_json_str(content)
        print("json格式问卷读取成功")
        print(questionnaire)

        return questionnaire
    except Exception as e:
        
        print(f"出现了{e}, 需要进行一次清洗")

        with open(f"prompts/json_washer.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    
        # 系统提示词的template
        washer_sys_msg = config["system_prompt"]

        # 用户提示词的template
        template = jinja_env.from_string(config["template"])  # 用户逐步输入内容
        json_washer_prompt = template.render(**{
            'json_str': content,
        })

        model_washer = init_chat_model(washer_model_name, model_provider="openai")
        #洗json工
        json_washer = create_react_agent(
        model=model_washer,
        tools=[],
        prompt=washer_sys_msg,  # 系统提示词
        )

        washed_result = await safe_agent_invoke(json_washer, json_washer_prompt)

        washed_content = washed_result["messages"][1].content
        print("清洗后的结果")
        print(washed_content)

        questionnaire=safe_read_json_str(washed_content)
        print("json格式问卷清洗成功")
        print(questionnaire)      
        return questionnaire

    
async def main():
    #o3 GPT5 o4-mini GPT4.1 GPT4.5
    #Claude 3.5 Claude 3.7
    #Grok3
    model_names=["chatgpt-4o-latest","o3-2025-04-16", #"gpt-5",
                 #"o4-mini-2025-04-16","gpt-4.1-2025-04-14", "GPT-4.5",
                 "claude-opus-4-20250514", "claude-sonnet-4-20250514",
                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"
                 ]
    

    Discussion_path="./datas/discussion"

    json_files = get_json_filenames(Discussion_path)
    brand_product_list = []

    ##先遍历Discussion
    for filename in json_files:
        brand, product_series = extract_brand_and_product(filename)
        if not (brand and product_series):
            print("格式解析错误")
            break

        # 再遍历模型
        for model_name in model_names:
            filename=f"./datas/questionnaire/{brand}^{product_series}_{model_name}_direct_questionnaire.json"

            if os.path.exists(filename):
                print(f"文件 {filename} 已存在，跳过生成")
                continue  
            
            washer_model_name="chatgpt-4o-latest"

            print(f"使用{model_name}针对{brand}的{product_series}系列产品, without discussion 生成问卷")
            questionnaire=await LLM_ask(brand, product_series, model_name, washer_model_name)

            

            with open(filename, 'w', encoding='utf-8') as f:
                # ensure_ascii=False确保中文等特殊字符正常显示
                json.dump(questionnaire, f, ensure_ascii=False, indent=4)
            print(f"问卷已成功保存到 {filename}")

    
if __name__=="__main__":
    asyncio.run(main())
    




