import os
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import yaml
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

load_dotenv(override=True)
import anthropic


__all__ = [
    "generate_prompt",
    "get_client",
    "call_4o",
    "to_format_template",
    "build_user_prompt",
    "get_prompt",
]
def get_prompt(prompt_sentence_yaml: str,context: Dict[str, str]):
    with open(prompt_sentence_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    #system_prompt: str = config['system_prompt']
    # 先构建系统提示词
    template_raw: str = config['system_prompt']
    system_prompt = build_user_prompt(
        template_raw,
        **context
    )
    
    template_raw: str = config["template"]
    user_prompt = build_user_prompt(
        template_raw,
        **context
    )
    return system_prompt,user_prompt
def generate_prompt(template: str, context: dict) -> str:
    try:
        return template.format(**context)
    except KeyError as e:
        missing_key = e.args[0]
        raise ValueError(f"生成 prompt 失败：缺少占位符 {missing_key} 的值。") from e

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    if not api_key:
        raise EnvironmentError("未检测到 OPENAI_API_KEY（或 openai_api_key），请在 .env 配置。")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

def _retryable(e: Exception) -> bool:
    if isinstance(e, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(e, APIError):
        code = getattr(e, "status_code", None)
        return code in (429, None) or (isinstance(code, int) and code >= 500)
    return False

def _sleep_with_backoff(attempt: int, base: float, cap: float):
    delay = min(cap, (base ** (attempt - 1)))
    delay *= (1.0 + random.random())
    time.sleep(delay)

def call_4o(prompt: str, user_content: Union[str, List[Dict[str, Any]]],model_name:str) -> str:
    model        = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")
    temperature  = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    timeout_s_openai    = float(os.getenv("OPENAI_TIMEOUT_S", "180"))
    timeout_s_claude    = float(os.getenv("CLAUDE_TIMEOUT_S", "180"))
    max_retries  = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
    backoff_base = float(os.getenv("OPENAI_BACKOFF_BASE", "0.8"))
    backoff_cap  = float(os.getenv("OPENAI_BACKOFF_CAP", "20"))
    if model_name:
        model=model_name
    openai_model_list = ["gpt-5-chat-latest","gpt-5-2025-08-07","gpt-5","gpt-4-0613","gpt-4","gpt-3.5-turbo","gpt-5-nano","gpt-5","gpt-5-mini-2025-08-07","gpt-5-mini","gpt-5-nano-2025-08-07","gpt-3.5-turbo-instruct","gpt-3.5-turbo-instruct-0914","gpt-4-1106-preview","gpt-3.5-turbo-1106","gpt-4-0125-preview","gpt-4-turbo-preview","gpt-3.5-turbo-0125","gpt-4-turbo","gpt-4-turbo-2024-04-09","gpt-4o","gpt-4o-2024-05-13","gpt-4o-mini-2024-07-18","gpt-4o-mini","gpt-4o-2024-08-06","chatgpt-4o-latest","gpt-4o-realtime-preview-2024-10-01","gpt-4o-audio-preview-2024-10-01","gpt-4o-audio-preview","gpt-4o-realtime-preview","gpt-4o-realtime-preview-2024-12-17","gpt-4o-audio-preview-2024-12-17","gpt-4o-mini-realtime-preview-2024-12-17","gpt-4o-mini-audio-preview-2024-12-17","gpt-4o-mini-realtime-preview","gpt-4o-mini-audio-preview","computer-use-preview","o3-mini","o3-mini-2025-01-31","gpt-4o-2024-11-20","computer-use-preview-2025-03-11","gpt-4o-search-preview-2025-03-11","gpt-4o-search-preview","gpt-4o-mini-search-preview-2025-03-11","gpt-4o-mini-search-preview","gpt-4o-transcribe","gpt-4o-mini-transcribe","gpt-4o-mini-tts","o3-2025-04-16","o4-mini-2025-04-16","o3","o4-mini","gpt-4.1-2025-04-14","gpt-4.1","gpt-4.1-mini-2025-04-14","gpt-4.1-mini","gpt-4.1-nano-2025-04-14","gpt-4.1-nano","gpt-image-1","codex-mini-latest","o3-pro","gpt-4o-realtime-preview-2025-06-03","gpt-4o-audio-preview-2025-06-03","o3-pro-2025-06-10","o4-mini-deep-research","o3-deep-research","o3-deep-research-2025-06-26","o4-mini-deep-research-2025-06-26","gpt-5-chat-latest","gpt-5-2025-08-07"]
    qwen_model_list=["qwen3-32b","qwen3-8b"]
    ds_model_list=["deepseek-reasoner", "deepseek-chat"]
    #by_model_list=["Qwen2.5-7B-Instruct","Qwen2.5-3B-Instruct","Qwen2.5-14B-Instruct","Qwen3-8B"]
    by_model_list_str = os.getenv("BY_MODEL_LIST", "")
    by_model_list = by_model_list_str.split(",") if by_model_list_str else []
    #print(by_model_list[-1])
    #print(model)    



    last_err: Optional[Exception] = None
    #if model in openai_model_list or model=="GPT-4.5":
    if model in openai_model_list:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt}]
        if isinstance(user_content, str):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.extend(user_content)
        client = get_client()
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    timeout=timeout_s_openai,
                    messages=messages
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt >= max_retries or not _retryable(e):
                    break
                _sleep_with_backoff(attempt, backoff_base, backoff_cap)
        raise RuntimeError(f"调用 OpenAI 失败：{last_err}")

    #----------------------------QWEN系列模型----------------------------
    elif model in qwen_model_list:
    
        qwen_api_key= os.getenv("QWEN_API_KEY", "")
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

        client = OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_base_url,
        )
        #print("client创建成功")
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    timeout=timeout_s_openai,
                    messages=[{"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},],
                    extra_body={"enable_thinking": False},
                )

                #os.environ["HTTP_PROXY"]=ori_http_proxy
                #os.environ["HTTPS_PROXY"]=ori_https_proxy
                return (resp.choices[0].message.content or "").strip()

            except Exception as e:
                last_err = e
                if attempt >= max_retries or not _retryable(e):
                    break
                _sleep_with_backoff(attempt, backoff_base, backoff_cap)

                #os.environ["HTTP_PROXY"]=ori_http_proxy
                #os.environ["HTTPS_PROXY"]=ori_https_proxy
                
        raise RuntimeError(f"调用 QWEN 失败：{last_err}") 

    elif model in by_model_list:
        client = OpenAI(
        api_key=os.getenv("BY_API_KEY", "BY_API_KEY"),
        base_url=os.getenv("BY_BASE_URL", "")    #从.env文件中获取闭源模型的URL
        )
        model_name = client.models.list().data[0].id      #本地端口每次只开放一个模型
        if model not in model_name:
            raise
        #print("All valid models:", client.models.list())
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    timeout=timeout_s_openai,
                    messages=[{"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},],
                    extra_body={"enable_thinking":False},
                )
                print(resp)

                return (resp.choices[0].message.content or "").strip()

            except Exception as e:
                last_err = e
                if attempt >= max_retries or not _retryable(e):
                    break
                _sleep_with_backoff(attempt, backoff_base, backoff_cap)


                
        raise RuntimeError(f"调用 closed-source model {model_name} 失败：{last_err}") 


    elif model in ds_model_list:
        ds_api_key= os.getenv("DS_API_KEY", "")
        qwen_base_url="https://api.deepseek.com"

        messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt}]
        if isinstance(user_content, str):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.extend(user_content)

        client = OpenAI(
            api_key=ds_api_key,
            base_url=qwen_base_url,
        )
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    timeout=timeout_s_openai,
                    messages=messages
                )
                #reasoning_content = response.choices[0].message.reasoning_content
                
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt >= max_retries or not _retryable(e):
                    break
                _sleep_with_backoff(attempt, backoff_base, backoff_cap)
        raise RuntimeError(f"调用 DeepSeek 失败：{last_err}")      

        

    else:
        
        client = anthropic.Client(
            api_key=os.getenv("CLAUDE_API_KEY", ""))
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt+user_content}]
        for attempt in range(1, max_retries + 1):
            try:
                # 调用 Claude 模型进行对话生成
                print(f"Claude-{model}调用超时时间为{timeout_s_claude}")
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=1,
                    messages=messages,
                    timeout=timeout_s_claude, 
                )
                return resp.content[0].text
            except Exception as e:
                last_err = e
                # 错误处理：重试逻辑
                if attempt >= max_retries or not _retryable(e):
                    print(f"Error occurred: {e}. No more retries.")
                    break


_JINJA_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\|[^}]*)?\}\}")
_JINJA_BLOCK = re.compile(r"\{\%.*?\%\}", re.DOTALL)

def to_format_template(tpl: str) -> str:
    s = _JINJA_BLOCK.sub("", tpl)
    s = _JINJA_VAR.sub(lambda m: "{" + m.group(1) + "}", s)
    return s

def build_user_prompt(tpl_raw: str, **kwargs) -> str:
    """
    统一的 user_prompt 生成入口：
      - 模板可为两种任一：Jinja2 风格（{{ var }}) 或 Python 格式化风格（{var}）
      - 参数数量与名称不定，全部从 **kwargs 传入
      - 内部统一走 generate_prompt(tpl_fmt, kwargs)
    用法示例：
      user_prompt = build_user_prompt(
          template_raw,
          user_discussion=data_txt,
          product_brand=brand,
          output_num=output_num,
          platform=platform,
          metric=metric,
      )
    """
    tpl_fmt = to_format_template(tpl_raw) if ("{{" in tpl_raw and "}}" in tpl_raw) else tpl_raw
    return generate_prompt(tpl_fmt, kwargs)

if __name__ == "__main__":
    #resp=call_4o("You are a kind Agent","hello ","chatgpt-4o-latest")
    #resp=call_4o("You are a kind Agent","hello ","GPT-4.5")
    #print(resp)
    #resp=call_4o("","hello ","claude-3-haiku-20240307")
    #print(resp)
    #resp=call_4o("","what's your version","deepseek-reasoner")   #deepseek-chat
    #resp=call_4o("","what's your version","Qwen2.5-7B-Instruct")   #deepseek-chat
    #resp=call_4o("","hello","gpt-5")
    #print(resp)
    #resp=call_4o("","hello","gpt-5-2025-08-07") 
    resp=call_4o("","hello","Llama-3.2-8B-Instruct") 
    print(resp)
