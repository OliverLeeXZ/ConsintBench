import asyncio
import os
import pickle
import sys
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)
from .logging_utils import log
# ---------------------------------全局变量-----------------------------------------
keywords = ["product model", "usage scenario", "Aspect", "Feeling", "Comparison", "Tendency"]

import json
def safe_read(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        log.error(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError as e:
        log.error(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
    return None

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

import langchain
from langgraph.prebuilt import create_react_agent
# -------------------------------Invoke LLM---------------------------
# 读取大模型提示词
import yaml
from jinja2 import Environment, StrictUndefined
model_name = "chatgpt-4o-latest"
model = init_chat_model(model_name, model_provider="openai")
jinja_env = Environment(undefined=StrictUndefined)

def fetch_mark(content):
    import re
    pattern = r"Final Score: (\d+(\.\d+)?)\."
    match = re.search(pattern, content)

    if match:
        score = match.group(1)  # 获取第一个捕获组（数字部分）
        try:
            score = float(score)
            # 检测socre是否是整数
            if score % 1 == 0 and score >= 0 and score <= 10:
                # 以int形式返回
                return int(score)
            else:
                log.error("提取到错误范围的Final Score: {score}")
                return -1

        except:
            log.error("提取到的Final Score并非数值: {score}")
            return -1

    # 没找到匹配项
    else:
        log.error("未在正则表达式中匹配到Final Score")
        return -1


async def topic_judge_total(brand, json_questionnaire):
    with open(f"prompts/judge_topic_total.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sys_msg = config['system_prompt']

    template = jinja_env.from_string(config["template"])  # 用户逐步输入内容

    agent = create_react_agent(
        model=model,
        tools=[],
        prompt=sys_msg,  # 系统提示词
    )

    mark_prompt = template.render(**{
        'brand': brand,
        "questionnaire": json_questionnaire,
    })

    try:
        mark = -1
        # 没有正确输出结果就一直重试
        while mark == -1:
            result = await safe_agent_invoke(agent, mark_prompt)

            content = result["messages"][1].content
            mark = fetch_mark(content)
        return mark, content

    except Exception as e:
        log.error(f"等待持续超时: {e}")
        return [], []


async def topic_judge_depth(brand, json_questionnaire):
    with open(f"prompts/judge_topic_depth.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sys_msg = config['system_prompt']

    template = jinja_env.from_string(config["template"])  # 用户逐步输入内容

    agent = create_react_agent(
        model=model,
        tools=[],
        prompt=sys_msg,  # 系统提示词
    )

    mark_prompt = template.render(**{
        'brand': brand,
        "questionnaire": json_questionnaire,
    })
    try:
        mark = -1
        # 没有正确输出结果就一直重试
        while mark == -1:
            result = await safe_agent_invoke(agent, mark_prompt)

            content = result["messages"][1].content
            mark = fetch_mark(content)

        return mark, content
    except Exception as e:
        log.error(f"等待持续超时: {e}")
        return [], []


def get_keyword(index: int, Brand: str = "Tesla", product_type="vehicles",
                exclude_type="software systems (such as FSD)", aspects=["price", "autonomous driving"],
                desc=["fast", "convenient"], Compare: str = "BYD"):
    if index == 0:
        keyword = keywords[0]
        # here it refers to
        explanation = f"the specific product model discussed. The product model must only include {product_type} of the {Brand} brand, and excludes {exclude_type}."
    elif index == 1:
        keyword = keywords[1]
        explanation = "the exact usage scenario discussed, but do not regard the time of use or a specific country/city as the usage scenario."
    elif index == 2:
        keyword = keywords[2]
        explanation = "the product's aspect discussed, such as {aspects[0]} and {aspects[1]}."
    elif index == 3:
        keyword = keywords[3]
        explanation = f"an adjective describing the discussed aspect(e.g., '{desc[0]}', '{desc[1]}')"

    elif index == 4:
        keyword = keywords[4]
        explanation = f"the other {keywords[0]}s that the discussed aspect is compared against; such {keywords[0]}s may be from other brands, such as {Compare}."
    elif index == 5:
        keyword = keywords[5]
        explanation = f"What tendency does the discussion advise or guess {Brand} to have?"

    return keyword, explanation


async def topic_judge_breadth(brand, json_questionnaire, index):
    global keywords
    with open(f"prompts/judge_topic_6_template.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    keyword, explanation = get_keyword(index)
    # 系统提示词的template
    template = jinja_env.from_string(config["system_prompt"])
    sys_msg = template.render(**{
        'brand': brand,
        "questionnaire": json_questionnaire,
        "keyword": keyword,
        "explanation": explanation,
    })

    # 用户提示词的template
    template = jinja_env.from_string(config["template"])  # 用户逐步输入内容
    mark_prompt = template.render(**{
        'brand': brand,
        "questionnaire": json_questionnaire,
        "keyword": keyword,
        "explanation": explanation,
    })

    agent = create_react_agent(
        model=model,
        tools=[],
        prompt=sys_msg,  # 系统提示词
    )

    try:
        mark = -1
        # 没有正确输出结果就一直重试
        while mark == -1:
            result = await safe_agent_invoke(agent, mark_prompt)

            content = result["messages"][1].content

            mark = fetch_mark(content)

        return mark, content
    except Exception as e:
        log.error(f"等待持续超时: {e}")
        return [], []

"""
async def main(brand, product_series, generate_questionnaire_model, task_type):
    global keywords
    grade8 = []
    with open(f"datas/questionnaire/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^questionnaire.json", "rb") as f:
        json_questionnaire=json.load(f)

    grade, illustration = await topic_judge_total(json_questionnaire)
    grade8.append(grade)
    grade, illustration = await topic_judge_depth(json_questionnaire)
    grade8.append(grade)
    grades = []
    illustrations = []
    for index in range(6):
        grade, illustration = await topic_judge_breadth(json_questionnaire, index)
        grades.append(grade)
        illustrations.append(illustration)
        grade8.append(grade)
    return grade8
"""
async def main(brand, product_series, generate_questionnaire_model, task_type):
    global keywords
    # 读取问卷数据
    with open(f"datas/questionnaire/{brand}^{product_series}^{generate_questionnaire_model}^{task_type}^questionnaire.json", "rb") as f:
        json_questionnaire = json.load(f)
    
    # 创建所有需要并发执行的任务
    tasks = [
        topic_judge_total(brand, json_questionnaire),       # 任务1：总体评判
        topic_judge_depth(brand, json_questionnaire),       # 任务2：深度评判
        topic_judge_breadth(brand, json_questionnaire, 0),  # 任务3：广度评判-索引0
        topic_judge_breadth(brand, json_questionnaire, 1),  # 任务4：广度评判-索引1
        topic_judge_breadth(brand, json_questionnaire, 2),  # 任务5：广度评判-索引2
        topic_judge_breadth(brand, json_questionnaire, 3),  # 任务6：广度评判-索引3
        topic_judge_breadth(brand, json_questionnaire, 4),  # 任务7：广度评判-索引4
        topic_judge_breadth(brand, json_questionnaire, 5)   # 任务8：广度评判-索引5
    ]
    
    # 并发执行所有任务，gather会按任务列表顺序返回结果
    results = await asyncio.gather(*tasks)
    
    # 提取每个任务的grade，保持原始顺序
    grade8 = [result[0] for result in results]
    
    
    return grade8


if __name__=="__main__":
    asyncio.run(main("Apple", "apple_vision_pro", "chatgpt-4o-latest", "top"))

