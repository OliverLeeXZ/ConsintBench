from langchain.chat_models import init_chat_model

#---------------------------------配置系统环境--------------------------------------
from dotenv import load_dotenv
load_dotenv(override=True)
from utils.call_llm import get_prompt,call_4o
import asyncio

import os
from utils.cope_json import *

#读取原始讨论文本
import json



#重新整理数据，获得关键信息
#返回类型是一个json字符串
def aggregate_comments_data(data: list[dict[str, any]])->str:
    #Json数据可以是一个列表吗？
    aggregate_data=[]


    for item in data:
        if not isinstance(item, dict):
            continue

        
        source = item.get('source', '')
        body = item.get('body', {})

        # 处理Twitter数据
        if source == 'twitter':

            item_dict={"source":"twitter", "discussion":body.get('full_text',""), "views_count":body.get('views_count',0)}        

        # 处理reddit数据
        elif source == 'reddit':

            item_dict={"source":"reddit", "discussion":body.get('content',''), "upvotes_count":body.get('upvote',0)}

        # 处理web数据
        elif source=='websearch':
            item_dict={"source":"web", "discussion":body.get('content','')}

        aggregate_data.append(item_dict)    

    json_data = json.dumps(aggregate_data, ensure_ascii=False, indent=2)
    print(f"数据聚合完成，类型为：{type(json_data)}")


    #保留中间变量
    """
    json_path=f"./datas/discussion/Tesla_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_data)
    print(f"聚合数据已导出已导出：{json_path}, 长度为{len(json_data)}个字符")
    """

    return json_data




#抽取评论数据
def extract_comments_data(data: list[dict[str, any]]) -> (list[dict[str, any]], list[dict[str, any]], list[dict[str, any]]):

    #推特的数据
    twitter_data=[];     reddit_data=[];   web_data=[]; 
    
    #推特测试用数据, 用来打印查看取值情况
    quote_counts=[]; bookmark_counts=[]; retweet_counts=[]; is_pinneds=[]; 
    favorite_counts=[]; views_counts=[]; reply_counts=[]; full_texts=[];
    #reddit
    upvotes=[];

    
    for item in data:
        if not isinstance(item, dict):
            continue

        
        source = item.get('source', '')
        body = item.get('body', {})

        """
        #数据格式
        twitter_comment_detail = {
            'platform': source,
            'quote_count':None,             #引用量
            'bookmark_count': None,     #收藏数
            'retweet_count': None,         #转发量
            'is_pinned': None,                 #星标
            'favorite_count':None,          #喜爱量
            'views_count':None,              #浏览量
            'reply_count' : None,             #回复量
            'full_text':None,                     #回复全文
             }
        """
    


        # 处理Twitter数据
        if source == 'twitter':
            quote_counts.append(body.get('quote_count',0))
            bookmark_counts.append(body.get('bookmark_count',0))
            retweet_counts.append(body.get('retweet_count',0))
            is_pinneds.append(body.get('is_pinned',0))
            favorite_counts.append(body.get('favorite_count',0))
            views_counts.append(body.get('views_count',0))
            reply_counts.append(body.get('reply_count',0))
            full_texts.append(body.get('full_text',0))

            twitter_comment_detail = {
            #'platform': 'twitter',
            #'quote_count':body.get('quote_count',0),                   #引用量                   数目极低
            #'bookmark_count': body.get('bookmark_count',0),    #收藏数                数目极少
            'retweet_count': body.get('retweet_count',0),              #转发量
            #'is_pinned': body.get('is_pinned',0),                            #星标                      都是False
            'favorite_count':body.get('favorite_count',0),               #喜爱量
            'views_count':body.get('views_count',0),                      #浏览量                   有代表性
            'reply_count' : body.get('reply_count',0),                     #回复量
            'full_text':body.get('full_text',''),                                    #回复全文
            }

            twitter_data.append(twitter_comment_detail)            

        elif source == 'reddit':
            upvotes.append(body.get('upvote',0))

            reddit_comment_detail = {
            #'platform': 'reddit',
            'title':body.get('title',''),
            'upvote':body.get('upvote',0),                            #点赞
            'content':body.get('content',''),
            }

            reddit_data.append(reddit_comment_detail)    

        # 这来自多平台，不一定具有点赞量
        elif source=='websearch':
            web_comment_detail = {
            #'platform': 'web',
            'content':body.get('content',''),
            }
            web_data.append(web_comment_detail)

    """
    print("推特调试数据")

    #数值稀疏数据
    #print("quote_counts:",quote_counts); print("bookmark_counts:",bookmark_counts);
    #print("is_pinneds:",is_pinneds);

    views_counts: ['6', '121', '193', '29', '8', '29', '52', '26', '17', '19255', '31', '36', '13', '41', '10', '45', '14', '19', '64', '199', '14', '9', '13', '67', '19', '21', '38', '161', '679', '2', '48', '38', '19', '64', '26', '126', '4', '17', '25', '7', '24', '31', '14', '530', '9', '15', '32', '10', '776', '18', '49', '111', '13', '47', '107', '25', '48', '84', '20', '11', '158', '17', '24', '72', '12', '207', '31', '14', '295', '1013', '45', '12', '13', '11', '9', '148', '58', '41', '17', '54', '54', '11644', '42', '7', '290', '16', '18', '25', '178', '34', '79', '35', '24', '31', '41', '18', '16', '40', '43', '32', '106', '22', '76', '38', '10', '101', '36', '70', '125', '39', '8', '5', '3', '96', '11', '19', '129', '186', '15', '8', '12', '18', '40', '40', '9', '15', '16', '17', '22', '45', '7', '5', '31', '17', '14', '15', '9', '10', '94', '9']


    #数值较多数据
    print("retweet_counts:",retweet_counts) ; 
    print("favorite_counts:",favorite_counts); print("views_counts:",views_counts);
    print("reply_counts:",reply_counts); #print(full_texts);
    """
    #print("upvotes:",upvotes);
    #upvotes: ['176', '3485', '3382', '4405', '1586', '1985', '1189', '1534', '757', '1270', '880', '605', '389', '652', '495', '145', '515', '418', '478', '120', '132', '13950', '196', '629', '637', '411', '99', '51', '62', '81', '158', '81', '61', '8741', '29', '77', '73', '3846', '175', '62', '400', '276', '0', '120', '83', '6', '63', '386', '296', '106', '155', '164', '29', '2', '10', '218', '128', '8', '11', '1805', '0', '12', '0', '213', '153', '28', '7', '0', '6', '0', '20', '41', '38', '0', '1617', '14', '83', '33', '6', '128', '0', '0', '0', '31', '0', '104', '90', '49', '16', '1', '9', '63', '50', '51', '4', '258', '23', '19', '0', '1308', '2419', '950', '647', '650', '937', '401', '40', '123', '9709', '20870', '4470', '8725', '1600', '39254', '1928', '2581', '3167', '1943', '16730', '1401', '3726', '5581', '23761', '4858', '1642', '1540', '1901', '52528', '988', '1613', '4037', '1879', '1501', '692', '1015', '2069', '690', '1383', '1000', '1163', '881', '932', '1464', '3731', '2019', '3729', '7219', '1060', '1414', '3790', '4812', '997', '1303', '1530', '1919', '1002', '267', '1484', '2367', '2284', '880', '9253', '4796', '996', '4722', '695', '921', '577', '1231', '2358', '1367', '8200', '816', '592', '1116', '319', '818', '584', '3222', '352', '704', '3135', '417', '6457', '3137']

    print("twitter个数:", len(twitter_data))
    print("reddit个数:", len(reddit_data))
    print("web个数:", len(web_data))
    
    return twitter_data, reddit_data, web_data




#带回退的agent调用
"""
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)   )
async def safe_agent_invoke(agent, prompt,  timeout=60):
    #带重试和超时的安全调用函数
    try:
        result= await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
            timeout=timeout
            )
        return result
    except asyncio.TimeoutError:
        print(f"调用超时 ({timeout}秒), 正在重试")
        raise   #触发重试
    except Exception as e:
        print(f"调用发生错误:{e}, 正在重试...")
        raise
"""



#读取用于从discussion生成问卷的大模型提示词
import yaml
from jinja2 import Environment, StrictUndefined

jinja_env = Environment(undefined=StrictUndefined)


def extract_json_header(text):
    import re
    # 关键修正：正则匹配「```json」开头、「```」结尾的代码块
    # 解释正则规则：
    # - ^```json\s* ：匹配开头的「```json」，允许后续有空白字符（如换行）
    # - (.*?) ：非贪婪捕获中间所有内容（即JSON本体），避免多匹配
    # - \s*```$ ：匹配结尾的「```」，允许前面有空白字符（如换行）
    # - re.DOTALL ：让「.」能匹配换行符（JSON通常跨多行）
    pattern = r'^```json\s*(.*?)\s*```$'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        raise ValueError("未找到被「```json」和「```」包围的JSON内容（检查文本格式是否正确）")
    
    # 提取JSON字符串并去除前后空白（避免换行/空格导致解析失败）
    json_str = match.group(1).strip()
    try:
        # 解析JSON并返回Python字典
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误，解析失败：{str(e)}") from e



questionnaire=""
def LLM_ask_with_discussion(aggregative_data, brand="Tesla", product_series="Model Y",
                                     model_name="chatgpt-4o-latest",
                                  washer_model_name="chatgpt-4o-latest",
                                ):

    global questionnaire


    system_prompt,user_prompt = get_prompt("./prompts/discussion_2_questionnaire.yaml",
                                           {'Brand':brand, "Product_series":product_series,
                                            "document_json":aggregative_data , "q_num":os.getenv("QUE_NUM")})


    questionnaire=call_4o(system_prompt,user_prompt,model_name)

    print(f"返回结果为 {questionnaire}")


    try:
        questionnaire = extract_json_header(questionnaire)
        print("解析成功：")
        print(questionnaire)
        return questionnaire

    except Exception as e:
        print(e, "接下来执行常规清洗流程")
        #简单清理
        questionnaire=questionnaire.replace("\\n", "\n").replace("\\t", "\t")#简单清洗

        try:
            questionnaire=json.loads(questionnaire)

            return questionnaire
        
        except Exception as e:
            
            print(f"出现了{e}, 需要进行一次清洗")

            system_prompt,user_prompt = get_prompt("prompts/json_washer.yaml",{'json_str': questionnaire})
            
            washed_result=call_4o(system_prompt,user_prompt,washer_model_name)


            print("清洗后的结果")
            print(washed_result)

            questionnaire=safe_read_json_str(washed_result)
            print("json格式问卷清洗成功")
            print(questionnaire)      
            return questionnaire


    




if __name__=="__main__":
    #-------------------------首先读取讨论文本------------------------------
    model_names=["chatgpt-4o-latest","o3-2025-04-16", #"gpt-5",
                 #"o4-mini-2025-04-16",
                 #"gpt-4.1-2025-04-14",
                 #"GPT-4.5",    #没有这个
                 #"gpt-5-chat-latest"
                 #"claude-opus-4-1-20250805",
                 #"claude-sonnet-4-20250514", #"claude-opus-4-20250514",
                 #"claude-3-7-sonnet-20250219",
                 #"claude-3-5-haiku-20241022",
                 #"claude-3-5-sonnet-20241022",
                 #"deepseek-reasoner",
                 #"deepseek-chat",
                 #"qwen3-8b"
                 #"Qwen2.5-7B-Instruct",
                 #"Qwen2.5-3B-Instruct",
                 #"qwen3-32b"
                 #"Qwen3-8B",
                 "Llama-3.2-8B-Instruct",
                 ]
    #special_mode=["claude-sonnet-4-20250514"]

    Discussion_path="./datas/discussion"

    json_files = get_json_filenames(Discussion_path)


    brand_product_list = []
    
    # 逐个解析文件名
    for filename in json_files:
        brand, product_series = extract_brand_and_product(filename)
        if not (brand and product_series):
            print("格式解析错误")
            break
        
        discussion=safe_read(f"./datas/discussion/{filename}")      #读原始讨论文本->list

        print(f"{filename}讨论文本已读,长度为 {len(str(discussion))}个字")

        aggregative_data=aggregate_comments_data(discussion)    #获得一个json字符串

        #aggregative_data=aggregative_data[:25000]

        #先截取前10000个文本
        #aggregative_data=aggregative_data[:5000]; print("保留前5000字")
        print(f"数据聚合完成，长度为{len(aggregative_data)}")
        
        #------------------------问卷生成---------------------------------
        for model_name in model_names:
            washer_model_name="chatgpt-4o-latest"

            filename = f"./datas/questionnaire/{brand}^{product_series}^{model_name}^discuss^questionnaire.json"
            print(f"准备操作{filename}")

            # 分离目录和文件名
            dir_path, file_name = os.path.split(filename)


            files_in_dir = os.listdir(dir_path)
            # 检查目标文件名是否在目录文件列表中（精确匹配，包括大小写）
            if file_name in files_in_dir:
                print(f"文件 {filename} 已存在，跳过生成")
                continue  # 继续循环的下一次迭代



            #brand="Tesla"

            
            questionnaire=LLM_ask_with_discussion(aggregative_data,brand, product_series, model_name,  washer_model_name)
            
            with open(filename, 'w', encoding='utf-8') as f:
                # ensure_ascii=False确保中文等特殊字符正常显示
                json.dump(questionnaire, f, ensure_ascii=False, indent=4)

            print(f"问卷已成功保存到 {filename}")

    








