import asyncio
import json
import os
import pickle
import concurrent.futures
import re
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass
from utils.logging_utils import log
from utils.cope_json import extract_brand_and_product_from_questionnaire
from utils.call_llm import get_prompt, call_4o

from dotenv import load_dotenv
load_dotenv(override=True)
import os
import shutil
#

@dataclass
class Config:
    #配置类
    model_name: str = "chatgpt-4o-latest"
    max_workers: int = 5
    target_questions: int = 4*int(os.getenv('QUE_NUM'))
    max_retries: int = 5
    questionnaire_dir: str = "./datas/questionnaire"
    output_dir: str = "./datas/branch/branch_from_ques"
    prompts_dir: str = "./prompts"


class QuestionnaireProcessor:
    """问卷处理器"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, filename: str) -> str:
        """生成输出路径"""
        brand, product_series, model, task_type = extract_brand_and_product_from_questionnaire(filename)
        output_path_1=f"{self.config.output_dir}/{brand}^{product_series}^{model}^{task_type}^sentence_list.pkl"
        output_path_2=f"{self.config.output_dir}/{brand}^{product_series}^{model}^{task_type}^sentence_list.txt"
        return output_path_1, output_path_2

    def _file_exists(self, filepath: str) -> bool:
        """检查文件是否存在"""
        exists = os.path.exists(filepath)
        if exists:
            log.info(f"文件已存在，跳过: {file·path}")
        return exists

    def generate_questions(self, file_path: str, repetition_path:str,  filename: str) -> Optional[List]:
        """生成问题列表"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                question_data = json.load(f)

            system_prompt, user_prompt = get_prompt(
                f"{self.config.prompts_dir}/ques_option_combine.yaml",
                {"questionnaire": str(question_data)}
            )

            # 重试生成问题
            for retry in range(self.config.max_retries):
                try:
                    response = call_4o(system_prompt + user_prompt, str(question_data), self.config.model_name)
                    question_list = json.loads(response)

                    if len(question_list) == self.config.target_questions:
                        log.info(f"生成 {len(question_list)} 个问题: {filename}")
                        return question_list
                    else:
                        log.error(f"{file_path}生成长度不符合要求:{len(question_list)}，正在重试{retry + 1}/{self.config.max_retries}")

                except json.JSONDecodeError:
                    log.warning(f"问题选项生成返回JSON解析失败，重试 {retry + 1}/{self.config.max_retries}")
                    continue

            
            log.error(f"生成问题失败: {filename}, 问卷具有repetition, 处理到错误问卷文件夹")
            
            #dest_path = repetition_path
            try:
                dest_file_name=repetition_path+"/"+filename
                shutil.move(file_path, dest_file_name)
            except:
                print("移动失败")
            
            return None

        except Exception as e:
            log.error(f"处理文件出错 {filename}: {e}")
            return None

    def process_single_question(self, item, brand: str, product_series: str) -> Optional[List]:
        try:
            context = {
                "product_brand": brand,
                "data": item,
                "product_series": product_series,
            }
            system_prompt, user_prompt = get_prompt(
                f"{self.config.prompts_dir}/generate_branch_questionnaire_group.yaml",
                context
            )
            response = call_4o(system_prompt, user_prompt, self.config.model_name)
            matches = re.findall(r'<(.*?)>', response)

            if len(matches) >= 6:
                matches.insert(0,brand)
                matches[1]=product_series
                matches[4],matches[5]=matches[5],matches[4]
                return matches 
            else:
                return None
            

        except Exception as e:
            log.error(f"处理问题项出错: {e}")
            return None

    """
    def process_questions_to_branches(self, filename: str, question_list: List) -> bool:
        #将问题转换为分支
        # 两个输出路径
        output_path ,output_path_txt= self._get_output_path(filename)
        

        if self._file_exists(output_path):
            return True

        brand, product_series, _, _ = extract_brand_and_product_from_questionnaire(filename)
        #100个问题
        items_to_process = question_list[:self.config.target_questions]

        if not items_to_process:
            return False

        process_func = partial(self.process_single_question, brand=brand, product_series=product_series)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(process_func, item) for item in items_to_process]
                results = []

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        log.error(f"任务 {i} 失败: {e}")

                #存成pkl文件
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

                #存成txt文件
                with open(output_path_txt, 'w', encoding='utf-8') as txt_file:
                    for item in results:
                        # 可以根据需要调整格式，这里使用逗号分隔每个字符串
                        line = ' , '.join(item)
                        txt_file.write(line + '\n')


                log.info(f"处理完成 {filename}: {len(results)} 个有效分支")
                return True

        except Exception as e:
            log.error(f"并发处理出错 {filename}: {e}")
            return False
    """
    def process_questions_to_branches(self, filename: str, question_list: List) -> bool:
        """将问题转换为分支"""
        # 两个输出路径
        output_path, output_path_txt = self._get_output_path(filename)
        #print(output_path)

        if self._file_exists(output_path):
            return True

        brand, product_series, _, _ = extract_brand_and_product_from_questionnaire(filename)

        items_to_process = question_list[:self.config.target_questions]

        if not items_to_process:
            return False
        log.info(f"{filename}->{self.config.target_questions}个自然问题")

        # 调整处理函数，使其返回原始索引和结果
        def process_func_with_index(index, item):
            return index, self.process_single_question(item, brand=brand, product_series=product_series)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 提交任务时带上索引
                futures = [executor.submit(process_func_with_index, i, item) 
                          for i, item in enumerate(items_to_process)]
                results_with_index = []

                for future in concurrent.futures.as_completed(futures):
                    try:
                        # 获取带索引的结果
                        index, result = future.result()
                        if result:
                            results_with_index.append((index, result))
                            log.info(f"完成{filename}第{index} 个问题")
                        else:
                            results_with_index.append((index,"None"))
                    except Exception as e:
                        log.error(f"任务处理失败: {e}")

                results_with_index.sort(key=lambda x: x[0])
                results = [item for _, item in results_with_index]

                # 存成pkl文件
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

                # 存成txt文件
                with open(output_path_txt, 'w', encoding='utf-8') as txt_file:
                    for item in results:
                        line = ' , '.join(item)
                        txt_file.write(line + '\n')

                log.info(f"处理完成 {filename}: {len(results)} 个有效分支")
                return True

        except Exception as e:
            log.error(f"并发处理出错 {filename}: {e}")
            return False


    def filter_valid_files(self, file_list: List[str]) -> List[str]:
        """过滤有效文件"""
        pattern = r'^[^/\\^]+(\^[^/\\^]+){4}\.json$'
        return [f for f in file_list if re.match(pattern, f)]

    async def process_all_files(self) -> None:
        """处理所有文件"""
        if not os.path.exists(self.config.questionnaire_dir):
            log.error(f"目录不存在: {self.config.questionnaire_dir}")
            return

        file_list = os.listdir(self.config.questionnaire_dir)
        valid_files = self.filter_valid_files(file_list)

        #for v_file in valid_files:
        #    if "AirPods" in v_file:
        #        print(v_file)

        if not valid_files:
            log.warning("没有找到有效文件")
            return

        log.info(f"开始处理 {len(valid_files)} 个文件")

        # 并发处理文件
        def process_single_file(filename):
            file_path = f"{self.config.questionnaire_dir}/{filename}"
            repetition_path = f"{self.config.questionnaire_dir}/repetition"
            #if "AirPods" in file_path:
            #    print(file_path)
            final_output,_ = self._get_output_path(filename)
            #print(f"final_output:{final_output}")
            if self._file_exists(final_output):
                return True

            question_list = self.generate_questions(file_path, repetition_path,filename)
            if not question_list:
                return False

            return self.process_questions_to_branches(filename, question_list)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            tasks = [loop.run_in_executor(executor, process_single_file, filename) for filename in valid_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if r is True)
            log.info(f"处理完成: {success_count}/{len(valid_files)} 个文件成功")


async def questionnaire_to_branch():
    """主函数"""
    # 这里设的太大会报错打开了太多的文件
    config = Config(max_workers=15, target_questions=4*int(os.getenv('QUE_NUM')), max_retries=4)
    processor = QuestionnaireProcessor(config)
    await processor.process_all_files()


if __name__ == "__main__":
    async def questionnaire_to_branches():
        await questionnaire_to_branch()
    asyncio.run(questionnaire_to_branches())
