from __future__ import annotations
import os
import re
import json
import sys
from contextlib import nullcontext

import yaml
import pickle
import threading
import time
from typing import Any, Tuple, Optional, List, Dict, Callable, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm.contrib.logging import logging_redirect_tqdm
load_dotenv(override=True)
from utils.call_llm import generate_prompt, call_4o, build_user_prompt
from utils.logging_utils import log
# 进度条
from tqdm import tqdm

import datetime

_LLM_LOG_LOCK = threading.Lock()

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _log_llm_output(
    *,
    brand: str,
    index: int,
    attempt: int,
    stage: str,                 # "try" 或 "empty_retry"
    platform: str,
    metric: int,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_text: str,
    log_dir: str = "datas/llm_logs"
) -> None:
    """把一次大模型输出落盘为 JSONL（每行一个 JSON 记录）"""
    _ensure_dir(log_dir)
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "brand": brand,
        "index": index,
        "attempt": attempt,
        "stage": stage,
        "platform": platform,
        "metric": metric,
        "model": model_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output": output_text,
    }
    path = os.path.join(log_dir, f"{brand}.txt")  # 每个品牌一个txt（JSONL）
    line = json.dumps(rec, ensure_ascii=False)
    with _LLM_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ==================== 限速器接口定义（可选外部传入） ====================
class _LimiterProto(Protocol):
    def wait(self) -> None: ...


class _RateLimiter:
    """简单的线程安全 QPS 限速器（可作为默认内部限速器）"""
    def __init__(self, qps: Optional[float]):
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.time()
            self.next_time = now + self.interval


# ==================== JSON 安全读取 ====================
def safe_read(file_path: str) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            log.warning(f"期望是 list，但获得了 {type(data)}，将尝试包装为单元素列表。")
            data = [data]
        log.info(f"Successfully loaded file: {file_path}")
        return data  # type: ignore[return-value]
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON format - {e}")
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



# ==================== 评论数据提取 ====================
def extract_comments_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    twitter_data: List[Dict[str, Any]] = []
    reddit_data: List[Dict[str, Any]] = []
    web_data: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source = item.get('source', '')
        body = item.get('body', {}) or {}
        if source == 'twitter':
            twitter_data.append({
                'retweet_count': int(body.get('retweet_count', 0) or 0),
                'favorite_count': int(body.get('favorite_count', 0) or 0),
                'views_count': int(body.get('views_count', 0) or 0),
                'reply_count': int(body.get('reply_count', 0) or 0),
                'full_text': body.get('full_text', '') or ''
            })
        elif source == 'reddit':
            reddit_data.append({
                'title': body.get('title', '') or '',
                'upvote': int(body.get('upvote', 0) or 0),
                'content': body.get('content', '') or '',
            })
        elif source == 'websearch':
            web_data.append({'content': body.get('content', '') or ''})
    return twitter_data, reddit_data, web_data


# ==================== 小工具：输出条数、权重 ====================
def calc_output_num(word_count: int) -> int:
    if word_count <= 50:
        return 3
    elif word_count <= 80:
        return 4
    return 5


def calc_weight(platform: str, metric: int) -> float:
    weight = 1.0
    if platform == "twitter":
        if metric >= 150: weight += 1
        if metric >= 500: weight += 1
    elif platform == "reddit":
        if metric >= 100:  weight += 0.66
        if metric >= 500:  weight += 0.66
        if metric >= 1000: weight += 0.66
    return weight


# ==================== 一次生成（兜底重试由上层控制） ====================
def _generate_branch_once_logged(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    *,
    brand: str,
    index: int,
    attempt: int,
    stage: str,           # "try" 或 "empty_retry"
    platform: str,
    metric: int,
    log_dir: str = "datas/llm_logs"
) -> tuple[list[str], list[list[str]], str]:
    """调用 LLM + 解析 + 记录原始输出"""
    text = call_4o(system_prompt, user_prompt, model_name)
    #print(text)
    # 先落盘原始输出
    _log_llm_output(
        brand=brand, index=index, attempt=attempt, stage=stage,
        platform=platform, metric=metric, model_name=model_name,
        system_prompt=system_prompt, user_prompt=user_prompt, output_text=text,
        log_dir=log_dir
    )
    # 和原来一致的解析逻辑
    parts = re.split(r'[。\.]\s*', text)
    sentences = [s.strip() for s in parts if s.strip()]
    branches: list[list[str]] = []
    for s in sentences:
        items = re.findall(r'<([^>]+)>', s)
        if len(items) == 6:
            branches.append(items)
    return sentences, branches, text



# ==================== 对外唯一接口（支持“一个线程=一个文件” + 实时进度条） ====================
def run_discussion2branch(
    round_start: int,
    round_end: int,
    discussion_json_filename: str,
    brand:str,
    product_series:str,
    *,
    template_yaml_path: str = "prompts/discussion_generate_branch.yaml",
    save_dir: str = "datas/branch/branch_from_disc",
    verbose: bool = True,
    model_name: str = "",
    max_workers: int = 15,
    qps: Optional[float] = None,
    retries: int = 2,
    external_rate_limiter: Optional[_RateLimiter] = None,
    show_progress: bool = True,
    progress_position: Optional[int] = None
) -> dict:

    #提前检测是否已经生成过了
    os.makedirs(save_dir, exist_ok=True)
    out_name = f"{brand}^{product_series}^discussion_branch.pkl"
    out_path = os.path.join(save_dir, out_name)


    if os.path.exists(out_path):
        log.info(f"文件 {out_path} 已存在，跳过生成")
        return


    #brand = os.path.splitext(os.path.basename(discussion_json_filename))[0]
    json_path = discussion_json_filename if os.path.exists(discussion_json_filename) else os.path.join("datas", "discussion", discussion_json_filename)
    json_data = safe_read(json_path)
    if not json_data:
        raise RuntimeError("未读取到讨论数据。")

    twitter_data, reddit_data, web_data = extract_comments_data(json_data)
    total = len(twitter_data) + len(reddit_data) + len(web_data)
    if total == 0:
        raise RuntimeError("讨论数据为空。")

    log.info(f"共{total}条数据")

    if not os.path.exists(template_yaml_path):
        raise FileNotFoundError(f"模板文件不存在: {template_yaml_path}")
    with open(template_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sys_template: str = config['system_prompt']
    template_raw: str = config["template"]

    start = max(0, int(round_start))
    end = min(int(round_end), total)
    if start >= end:
        raise ValueError(f"无效的处理区间 start={start}, end={end}, 总数={total}")

    # 任务列表
    tasks: List[Tuple[int, str, str, int]] = []
    for i in range(start, end):
        if i < len(twitter_data):
            platform = "twitter"; idx = i
            data_txt = twitter_data[idx]["full_text"]; metric = int(twitter_data[idx]["views_count"])
        elif i < len(twitter_data) + len(reddit_data):
            platform = "reddit"; idx = i - len(twitter_data)
            data_txt = (reddit_data[idx]["title"] or "") + (reddit_data[idx]["content"] or "")
            metric = int(reddit_data[idx]["upvote"])
        else:
            platform = "web"; idx = i - len(twitter_data) - len(reddit_data)
            data_txt = web_data[idx]["content"]; metric = 0
        tasks.append((i, platform, data_txt, metric))

    limiter = external_rate_limiter if external_rate_limiter is not None else _RateLimiter(qps)

    logic_map: Dict[int, List[str]] = {}
    branch_map: Dict[int, List[List[Any]]] = {}
    cumulative_branches = 0
    lock = threading.Lock()

    # ==== 进度条：固定写到 stdout，完成后不保留行 ====
    pbar: Optional[tqdm] = None
    total_items = len(tasks)
    if show_progress:
        pbar = tqdm(
            total=total_items,
            desc=f"{brand}",
            position=progress_position if progress_position is not None else 0,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
            maxinterval=1.0,
            file=sys.stdout
        )
        pbar.set_postfix(branches=0)

    def _tick(n_new: int):
        if not pbar:
            return
        nonlocal cumulative_branches
        with lock:
            cumulative_branches += n_new
            pbar.update(1)
            pbar.set_postfix(branches=cumulative_branches)

    def _worker(i: int, platform: str, data_txt: str, metric: int) -> Tuple[int, List[str], List[List[Any]]]:
        output_num = calc_output_num(len(data_txt.split()))
        weight = calc_weight(platform, metric)
        system_prompt= build_user_prompt(
            sys_template,
            user_discussion=data_txt,
            product_brand=brand,
            output_num=output_num,
            product_series=product_series,
        )

        user_prompt = build_user_prompt(
            template_raw,
            user_discussion=data_txt,
            product_brand=brand,
            output_num=output_num,
            product_series=product_series,

        )

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                limiter.wait()
                # 第一次尝试：记录到 txt
                sentences, branches, _raw = _generate_branch_once_logged(
                    system_prompt, user_prompt, model_name,
                    brand=brand, index=i, attempt=attempt, stage="try",
                    platform=platform, metric=metric, log_dir="datas/llm_logs"
                )
                
                for branch_tmp in branches:
                    branch_tmp.insert(0,brand)
                    branch_tmp[1]=product_series
                    branch_tmp[4], branch_tmp[5] =branch_tmp[5], branch_tmp[4]
                log.info(f"初始生成结果：{branches}")
                # 为空兜底再打一遍：同样记录到 txt
                if not branches:
                    limiter.wait()
                    sentences, branches, _raw2 = _generate_branch_once_logged(
                        system_prompt, user_prompt, model_name,
                        brand=brand, index=i, attempt=attempt, stage="empty_retry",
                        platform=platform, metric=metric, log_dir="datas/llm_logs"
                    )
                    for branch_tmp in branches:
                        branch_tmp.insert(0,brand)
                        branch_tmp[1]=product_series
                        branch_tmp[4], branch_tmp[5] =branch_tmp[5], branch_tmp[4]
                    log.info(f"再次尝试结果：{branches}")

                enriched: List[List[Any]] = []
                for b in branches:
                    b = list(b)
                    b.extend([weight, data_txt, [platform, metric]])
                    enriched.append(b)
                return i, sentences, enriched

            except Exception as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 8))

        log.error(f"[task-{i}] 生成失败：{last_exc}")
        # 失败也可以（可选）写一条记录说明失败，这里先返回空结果
        return i, [], []

    # 用 logging_redirect_tqdm 保证日志不打乱进度条
    with logging_redirect_tqdm() if show_progress else nullcontext():
        try:
            if max_workers <= 1:
                for (i, platform, data_txt, metric) in tasks:
                    ii, ss, bb = _worker(i, platform, data_txt, metric)
                    logic_map[ii] = ss
                    branch_map[ii] = bb
                    _tick(len(bb))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_i = {
                        executor.submit(_worker, i, platform, data_txt, metric): i
                        for (i, platform, data_txt, metric) in tasks
                    }
                    for fut in as_completed(future_to_i):
                        i = future_to_i[fut]
                        try:
                            ii, ss, bb = fut.result()
                        except Exception as e:
                            log.error(f"[task-{i}] 未捕获异常：{e}")
                            ii, ss, bb = i, [], []
                        logic_map[ii] = ss
                        branch_map[ii] = bb
                        _tick(len(bb))
        finally:
            if pbar is not None:
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.close()

    logic_whole: List[str] = []
    branch_whole: List[List[Any]] = []
    for i in range(start, end):
        logic_whole.extend(logic_map.get(i, []))
        branch_whole.extend(branch_map.get(i, []))

    if verbose:
        log.info(f"[ok] {brand} 抽取完成。总枝数：{len(branch_whole)}")


    with open(out_path, "wb") as f:
        pickle.dump({'logics': logic_whole, 'branches': branch_whole}, f)

    if verbose:
        log.info(f"[ok] 数据已成功保存到 {out_path}")
        log.info(f"无后缀文件名：{os.path.splitext(out_name)[0]}")

    return {
        "brand": brand,
        "save_path": out_path,
        "num_branches": len(branch_whole),
        "num_total_items": total,
        "span": [start, end],
        "max_workers": max_workers
    }


if __name__ == "__main__":
    Discussion_path="./datas/discussion"
    json_files = get_json_filenames(Discussion_path)
    brand_product_list = []
    for filename in json_files:
        brand, product_series = extract_brand_and_product(filename)
        if not (brand and product_series):
            log.error("格式解析错误")
            break
        run_discussion2branch(
            round_start=1,
            round_end=30000,
            #round_end=10,
             discussion_json_filename= filename,
            brand=brand,
            product_series=product_series,

            template_yaml_path="prompts/discussion_generate_branch.yaml",
            save_dir="datas/branch/branch_from_disc",
            verbose=True,
            model_name=""
        )   
  
