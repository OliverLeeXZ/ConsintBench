import asyncio
import json
import os
import pickle
import re
import sys
import random
from asyncio import Semaphore

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from utils.call_llm import get_prompt, call_4o
from utils.logging_utils import log

from utils.cope_json import *

load_dotenv(override=True)

def clean_questionnaire(questionnaire: List[str]) -> List[str]:
    cleaned = []
    for i, text in enumerate(questionnaire, start=1):
        s = text.replace("```json", "").replace("```", "")
        s = re.sub(r',(\s*[\]}])', r'\1', s)
        s = re.sub(r'"Question\s*\d+"\s*:', f'"Question {i}":', s)
        cleaned.append(s)
    return cleaned

"""
def _generate_questionnaire(
    sentence_with_discussion: List[Any],
    prompt_question_yaml: str,
    model_name: str,
    concurrency: int = 25,
    max_retries: int = 4,
    timeout_sec: float = 60.0,
) -> List[str]:

    # 函数内的主函数
    async def _runner() -> List[str]:
        sem = Semaphore(concurrency)
        async def run_one(idx: int, item: Any) -> str:
            delay = 0.5
            for attempt in range(1, max_retries + 1):
                try:
                    system_prompt, user_prompt = get_prompt(
                        prompt_question_yaml, {"data": str(item)}
                    )
                    # 将同步的 call_4o 放到线程池，并加超时
                    return await asyncio.wait_for(
                        asyncio.to_thread(call_4o, system_prompt, user_prompt, model_name),
                        timeout=timeout_sec,
                    )
                except Exception as e:
                    # 可根据需要更细分异常类型（如 HTTP 429/5xx），这里统一重试
                    if attempt == max_retries:
                        # 失败时写日志并返回一个可被 clean_questionnaire 处理的占位
                        log.error(f"[问卷生成失败] idx={idx}, err={e}")
                        return f'Question {idx+1}: ERROR: {e}'
                    await asyncio.sleep(delay + random.random() * 0.3)
                    delay *= 2

        async def guarded(idx: int, item: Any) -> str:
            async with sem:
                return await run_one(idx, item)

        tasks = [asyncio.create_task(guarded(i, it)) for i, it in enumerate(sentence_with_discussion)]
        results: List[str] = await asyncio.gather(*tasks)
        return results

    raw_results = asyncio.run(_runner())
    return clean_questionnaire(raw_results)
"""

#单任务版本
"""
def _generate_questionnaire(
    sentence_with_discussion: List[Any],
    prompt_question_yaml: str,
    model_name: str,
    brand:str,
    product_series:str,
    max_retries: int = 4,
    timeout_sec: float = 60.0,
) -> List[str]:

    num_questions=os.getenv('QUE_NUM')       #根据指定数量的branch生成指定数量的问题

    # 函数内的主函数
    async def _runner() -> List[str]:
        sem = Semaphore(1)                                 # 只有1个并发
        async def run_one(idx: int, item: Any) -> str:
            delay = 0.5
            for attempt in range(1, max_retries + 1):
                try:
                    system_prompt, user_prompt = get_prompt(
                        prompt_question_yaml, {"data": str(item), "num_questions":num_questions,
                                               "brand":brand,"product_series":product_series}
                    )
                    # 将同步的 call_4o 放到线程池，并加超时
                    return await asyncio.wait_for(
                        asyncio.to_thread(call_4o, system_prompt, user_prompt, model_name),
                        timeout=timeout_sec,
                    )
                except Exception as e:
                    # 可根据需要更细分异常类型（如 HTTP 429/5xx），这里统一重试
                    if attempt == max_retries:
                        # 失败时写日志并返回一个可被 clean_questionnaire 处理的占位
                        log.error(f"[TOP问卷生成失败] idx={idx}, err={e}")
                        return f'Question {idx+1}: ERROR: {e}'
                    await asyncio.sleep(delay + random.random() * 0.3)
                    delay *= 2

        async def guarded(idx: int, item: Any) -> str:
            async with sem:
                return await run_one(idx, item)

        #tasks = [asyncio.create_task(guarded(i, it)) for i, it in enumerate(sentence_with_discussion)]

        # 转成唯一一个task
        #tasks = [asyncio.create_task(guarded(i, it)) for i, it in enumerate(sentence_with_discussion)]
        #results: List[str] = await asyncio.gather(*tasks)

        # 直接等待问卷生成
        raw_questionnaire=await(  guarded(0, sentence_with_discussion)   )
        return raw_questionnaire

    #raw_results = asyncio.run(_runner())     #开启执行
    raw_questionnaire = asyncio.run(_runner())
    print(raw_questionnaire)
    return clean_questionnaire(raw_questionnaire)
"""

# 单任务简化版本
def _generate_questionnaire(
    sentence_with_discussion: List[Any],
    prompt_question_yaml: str,
    model_name: str,
    brand: str,
    product_series: str,
    max_retries: int = 4,
    timeout_sec: float = 60.0,
) -> List[str]:
    num_questions = os.getenv('QUE_NUM')  # 根据指定数量的branch生成指定数量的问题

    # 函数内的主函数
    async def _runner() -> List[str]:
        delay = 0.5
        for attempt in range(1, max_retries + 1):
            try:
                # 获取提示词
                system_prompt, user_prompt = get_prompt(
                    prompt_question_yaml, {
                        "data": str(sentence_with_discussion),
                        "num_questions": num_questions,
                        "brand": brand,
                        "product_series": product_series
                    }
                )
                
                # 调用模型生成问卷，带超时处理
                return await asyncio.wait_for(
                    asyncio.to_thread(call_4o, system_prompt, user_prompt, model_name),
                    timeout=timeout_sec,
                )
            except Exception as e:
                if attempt == max_retries:
                    # 达到最大重试次数，记录日志并返回错误信息
                    log.error(f"[TOP问卷生成失败] err={e}")
                    return [f'Question 1: ERROR: {e}']
                
                # 重试前等待
                await asyncio.sleep(delay + random.random() * 0.3)
                delay *= 2

    # 执行异步任务
    questionnaire = asyncio.run(_runner())

    #print(raw_questionnaire)
    # 进行问卷的JSON格式清洗

    try:
        questionnaire=json.loads(questionnaire)

        return questionnaire
    
    except Exception as e:
        
        print(f"出现了{e}, 需要进行一次清洗")

        system_prompt,user_prompt = get_prompt("prompts/json_washer.yaml",{'json_str': questionnaire})
        washer_model_name="chatgpt-4o-latest"
        washed_result=call_4o(system_prompt,user_prompt,washer_model_name)


        print("清洗后的结果")
        print(washed_result)

        questionnaire=safe_read_json_str(washed_result)
        print("json格式问卷清洗成功")
        print(questionnaire)      
        return questionnaire

    
    #return clean_questionnaire(raw_questionnaire)




# 老版本功能：逐个输入branch, 每个branch转换成一个问题
#  新版本功能：输入branch list, 直接转换成一个问卷
def branch_to_questionnaire(
    product_brand: str,
    product_type: str,
    prompt_sentence_yaml: str = "prompts/branches_combine_sentence.yaml",    #先把关键词重新生成成句子
    #prompt_question_yaml: str = "prompts/create_question.yaml",                        # 原本上传的是discussion+关键词 
    prompt_question_yaml: str = "prompts/create_question_2.yaml",
    sentence_out_dir: str = "datas/combined/branches",                                        # 关键句子的输出
    questionnaire_out_dir: str = "datas/questionnaire",                                          # 问卷的输出
    model_name: Optional[str] = None,                                                                  # 用于生成问卷的模型
    print_branches: bool = False,
) ->None:
    global branch,sentence_list
    
    model_name = model_name if model_name else os.getenv("OPENAI_MODEL")
    branch_file=f"datas/branch/branch_from_top/{product_brand}^{product_type}^top_branches.pkl"
    sentence_pkl = Path(sentence_out_dir) / f"{product_brand}^{product_type}^{model_name}^top^sentence_list.pkl"
    q_pkl = Path(questionnaire_out_dir) / f"{product_brand}^{product_type}^{model_name}^top^questionnaire.pkl"
    q_json=Path(questionnaire_out_dir)/f"{product_brand}^{product_type}^{model_name}^top^questionnaire.json"

    if q_json.exists():
        log.info(f"[跳过] 问卷JSON文件已存在：{q_json}")
        return

    # 打开含有  branch+discussion的文件
    with open(branch_file, "rb") as f:
        obj = pickle.load(f)
    branch = obj["branches"] if isinstance(obj, dict) and "branches" in obj else obj

    if sentence_pkl.exists():
        log.info(f"[加载] 已存在句子列表文件：{sentence_pkl}")          #已经用ChatGPT把关键词组装回概括句
        with open(sentence_pkl, "rb") as f:
            sentence_with_discussion = pickle.load(f)
    else:
        if not isinstance(branch, list):
            raise ValueError(f"分支文件格式异常，应为 list 或包含 'branches' 的 dict，实际：{type(branch)}")
        if print_branches:
            for b in branch:
                log.info(b)

        #branch_8 = [b[1:-7] for b in branch]
        #由于出现过剪枝，每个branch的长度不同
        branch_8 = [b[1:-1] for b in branch]
        branch_7_str = json.dumps(branch_8, ensure_ascii=False)
    
        #print("branch_8")
        #print(branch_8)

        # 一次prompt20个，可以接受
        system_prompt, user_prompt = get_prompt(prompt_sentence_yaml,{"data":branch_7_str,"product_brand": product_brand,"product_type": "vehicles","other_brand":"BYD"})
        sentence_raw = call_4o(system_prompt, user_prompt, model_name)

        try:
            sentence_list = json.loads(sentence_raw)
        except Exception:
            sentence_list = [x.strip() for x in str(sentence_raw).splitlines() if x.strip()]

        #print("sentence_list")
        #print(sentence_list)
        #print(len(sentence_list))
        if len(sentence_list)!=len(branch_8):
            log.error(f"branch2sentence 数量错误, branches:{len(branch_8)}, sentences:{len(sentence_list)}")
            return

        # 直接送入sentence_list即可
        #sentence_with_discussion = [
        #    [sentence_list[i], branch[i][-1]] for i in range(min(len(sentence_list), len(branch)))
        #]

        
        Path(sentence_out_dir).mkdir(parents=True, exist_ok=True)
        with open(sentence_pkl, "wb") as f:
            pickle.dump(sentence_list, f)
            #pickle.dump(sentence_with_discussion, f)
        log.info(f"[保存] 句子列表已保存：{sentence_pkl}")

    #questionnaire = _generate_questionnaire(sentence_with_discussion, prompt_question_yaml, model_name)
    questionnaire = _generate_questionnaire(sentence_list, prompt_question_yaml, model_name,product_brand,product_type)
    Path(questionnaire_out_dir).mkdir(parents=True, exist_ok=True)
    #print(questionnaire)

    #parsed_dicts = [json.loads(q_str) for q_str in questionnaire]
    #cleaned_questionnaire = {}
    #for d in parsed_dicts:
    #    cleaned_questionnaire.update(d)

    with open(q_json, 'w', encoding='utf-8') as f:
        json.dump(questionnaire, f, ensure_ascii=False, indent=4)
    log.info(f"问卷JSON已成功保存到 {q_json}")


if __name__ == "__main__":
    branch_to_questionnaire(product_brand="Acer",product_type="Acer monitor", model_name = "chatgpt-4o-latest")
    #branch_to_questionnaire(model_name="chatgpt-4o-latest")
    # with open("datas/branch/branch_from_top/Tesla_top_branches.pkl", "rb") as f:
    #     branch = pickle.load(f)
    # print(branch[-4])
    # for b in branch:
    #     print(b[-2])

#
#     # system_prompt, user_prompt = get_prompt("prompts/branches_combine_sentence.yaml", {"data": , "product_brand": "Tesla","product_type": "vehicles", "other_brand": "BYD"})
#     # sentence_raw = call_4o(system_prompt, user_prompt, "chatgpt-4o-latest")
