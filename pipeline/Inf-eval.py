"""
信息量评估- 核心指标计算函数

相关四个指标：
- calculate_sem_ent_batch(): 批量计算语义熵
- calculate_novelsum_batch(): 批量计算NovelSum多样性
- calculate_redundancy_batch(): 批量计算冗余度
- calculate_lexical_diversity_batch(): 批量计算词汇多样性

最终总指标- calculate_comprehensive_richness_batch(): 批量计算综合信息量评分

计算词汇多样性的部分要用到额外这些辅助函数：
calculate_distinct_n() - 计算Distinct-n指标
calculate_ttr() - 计算TTR指标
calculate_ttr_batch() - 批量计算TTR
calculate_distinct_n_batch() - 批量计算Distinct-n
"""

import torch
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




import os
from utils.cope_json import *


from dotenv import load_dotenv
load_dotenv(override=True)


def calculate_redundancy_batch(questionnaires_data, ques_name_list,similarity_threshold=0.85):
    """
    批量计算问卷的综合冗余度：包含问题间冗余度和问题内部选项冗余度
    公式：最终冗余度 = (问题间冗余度 + 选项间冗余度) / 2
    
    Args:
        questionnaires_data: 问卷数据列表，每个问卷是一个包含问题和选项的字典
        similarity_threshold: 相似度阈值，用于识别重复问题对（仅用于详细分析）
    
    Returns:
        redundancy_scores: 综合冗余度分数列表 (0-1)，0表示无冗余，1表示完全冗余
    """
    redundancy_scores = []
    
    for i in range(len(questionnaires_data)):
        questionnaire_data=questionnaires_data[i]
        if not questionnaire_data:
            redundancy_scores.append(0.0)
            continue
        
        # 这里需要传入一个dict，实际却传入了一个list
        print(f"calculate redundancy :{i}")
        #print(type(questionnaire_data))

        # 提取所有问题文本
        questions = []
        question_options = []
        
        for key, value in questionnaire_data.items():
            if 'question' in value:
                questions.append(value['question'])
                # 提取选项
                if 'options' in value and isinstance(value['options'], list):
                    options = value['options']
                    if len(options) >= 2:  # 至少需要2个选项才能计算冗余度
                        question_options.append(options)
        
        # 计算问题间冗余度
        question_redundancy = 0.0
        if len(questions) >= 2:
            # 获取所有问题的嵌入向量
            embeddings = model.encode(questions, convert_to_tensor=True)
            
            # 计算每个问题与其他问题的最大相似度的平均值
            max_similarities = []
            for i in range(len(questions)):
                max_sim = 0.0
                for j in range(len(questions)):
                    if i != j:  # 不与自己比较
                        cos_sim = util.pytorch_cos_sim(
                            embeddings[i].unsqueeze(0), 
                            embeddings[j].unsqueeze(0)
                        ).item()
                        if cos_sim > max_sim:
                            max_sim = cos_sim
                max_similarities.append(max_sim)
            
            question_redundancy = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
        
        # 计算各问题拼接文本间的相似度作为选项冗余度
        if len(question_options) >= 2:
            # 将所有问题的选项内容分别拼接成字符串
            combined_options_texts = []
            for options in question_options:
                if len(options) >= 2:
                    combined_text = " ".join(options)
                    combined_options_texts.append(combined_text)
            
            if len(combined_options_texts) >= 2:
                # 获取所有拼接文本的嵌入向量
                combined_embeddings = model.encode(combined_options_texts, convert_to_tensor=True)
                
                # 计算各拼接文本间的最大相似度的平均值
                max_similarities = []
                for i in range(len(combined_options_texts)):
                    max_sim = 0.0
                    for j in range(len(combined_options_texts)):
                        if i != j:  # 不与自己比较
                            cos_sim = util.pytorch_cos_sim(
                                combined_embeddings[i].unsqueeze(0), 
                                combined_embeddings[j].unsqueeze(0)
                            ).item()
                            if cos_sim > max_sim:
                                max_sim = cos_sim
                    max_similarities.append(max_sim)
                
                # 选项冗余度 = 各拼接文本间相似度的平均值
                avg_option_redundancy = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
            else:
                avg_option_redundancy = 0.0
        else:
            avg_option_redundancy = 0.0
        
        # 计算最终的综合冗余度
        final_redundancy = (question_redundancy + avg_option_redundancy) / 2.0
        
        redundancy_scores.append(final_redundancy)
    
    return redundancy_scores

def calculate_distinct_n(questions, n=1):
    """
    Calculate Distinct-n metric: ratio of unique n-grams to total n-grams
    Args:
        questions: list of question strings
        n: n-gram size (default 1 for unigrams)
    Returns: Distinct-n score (0-1)
    """
    all_ngrams = []
    
    for question in questions:
        # 简单的文本预处理：转小写，移除标点符号
        text = re.sub(r'[^\w\s]', '', question.lower())
        words = text.split()
        
        # 生成n-grams
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            all_ngrams.append(ngram)
    
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)

def calculate_ttr(questions):
    """
    Calculate Type-Token Ratio (TTR): ratio of unique words to total words
    Args:
        questions: list of question strings
    Returns: TTR score (0-1)
    """
    all_words = []
    
    for question in questions:
        # 简单的文本预处理：转小写，移除标点符号
        text = re.sub(r'[^\w\s]', '', question.lower())
        words = text.split()
        all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)

def calculate_lexical_diversity_batch(questionnaires_data, ques_name_list):
    """
    批量计算问卷的综合词汇多样性：包含问题间词汇多样性和问题内部选项词汇多样性
    公式：最终词汇多样性 = (问题间词汇多样性 + 平均问题内选项词汇多样性) / 2
    
    Args:
        questionnaires_data: 问卷数据列表，每个问卷是一个包含问题和选项的字典
    
    Returns:
        lexical_diversity_scores: 综合词汇多样性分数列表 (0-1)
    """
    lexical_diversity_scores = []
    
    for i in range(len(questionnaires_data)):
        questionnaire_data=questionnaires_data[i]
        print(f"{ques_name_list[i]}:lexical_diversity")
        
        if not questionnaire_data:
            lexical_diversity_scores.append(0.0)
            continue
        
        # 提取所有问题文本
        questions = []
        question_options = []
        
        for key, value in questionnaire_data.items():
            if 'question' in value:
                questions.append(value['question'])
                # 提取选项
                if 'options' in value and isinstance(value['options'], list):
                    options = value['options']
                    if len(options) >= 2:  # 至少需要2个选项才能计算词汇多样性
                        question_options.append(options)
        
        # 计算问题间词汇多样性
        question_lexical_diversity = 0.0
        if questions:
            # 计算TTR指标（词汇级别）
            ttr = calculate_ttr(questions)
            
            # 计算Distinct-2指标（短语级别）
            distinct_2 = calculate_distinct_n(questions, n=2)
            
            # 使用TTR和Distinct-2的平均值作为问题间词汇多样性指标
            question_lexical_diversity = (ttr + distinct_2) / 2
        
        # 计算每个问题内部选项的词汇多样性，然后取平均
        option_lexical_diversities = []
        for options in question_options:
            if len(options) >= 2:
                # 计算选项的TTR指标
                option_ttr = calculate_ttr(options)
                
                # 计算选项的Distinct-2指标
                option_distinct_2 = calculate_distinct_n(options, n=2)
                
                # 使用TTR和Distinct-2的平均值作为该问题选项的词汇多样性指标
                option_lexical_diversity = (option_ttr + option_distinct_2) / 2
                option_lexical_diversities.append(option_lexical_diversity)
        
        # 计算平均问题内部选项词汇多样性
        avg_option_lexical_diversity = np.mean(option_lexical_diversities) if option_lexical_diversities else 0.0
        
        # 计算最终的综合词汇多样性
        final_lexical_diversity = (question_lexical_diversity + avg_option_lexical_diversity) / 2.0
        
        lexical_diversity_scores.append(final_lexical_diversity)
    
    return lexical_diversity_scores

def calculate_information_richness_batch(questionnaires_data, ques_name_list):
    """
    批量计算信息量指标
    公式：信息量指标 = (1 - 综合冗余度 + 综合词汇多样性) / 2
    
    Args:
        questionnaires_data: 问卷数据列表，每个问卷是一个包含问题和选项的字典
    
    Returns:
        information_richness_scores: 信息量指标分数列表 (0-1)
    """
    # 批量计算冗余度和词汇多样性
    redundancy_scores = calculate_redundancy_batch(questionnaires_data, ques_name_list)
    lexical_diversity_scores = calculate_lexical_diversity_batch(questionnaires_data, ques_name_list)
    
    # 计算信息量指标
    information_richness_scores = []
    
    for i in range(len(questionnaires_data)):
        # 信息量指标 = 冗余度与词汇多样性的平均值
        # 注意：冗余度越低越好，所以用 (1 - redundancy) 来转换
        information_richness = (1 - redundancy_scores[i] + lexical_diversity_scores[i]) / 2
        information_richness_scores.append(information_richness)

        print(f"{ques_name_list[i]}:{information_richness}")
    
    return information_richness_scores


if __name__=="__main__":
    # 1) 收集文件
    ques_dir = "datas/questionnaire"
    file_list = os.listdir(ques_dir)
    
    file_list = [file for file in os.listdir(ques_dir) if file.endswith('.json')]

    # 2) 搜集所有的问卷，涵盖品牌+产品系列
    brand_list = []
    product_series_list=[]
    generate_questionnaire_model_list=[]
    task_type_list=[]
    for file in file_list:
        brand, product_series, generate_questionnaire_model, task_type = extract_brand_and_product_from_questionnaire(file)
        brand_list.append(brand)
        product_series_list.append(product_series)
        generate_questionnaire_model_list.append(generate_questionnaire_model)
        task_type_list.append(task_type)

    print(brand_list)
    print(product_series_list)


    #3) 遍历所有lighted tree

    ques_list_list=[]    #每个问卷问题列表 的列表
    ques_name_list=[]    #问卷名的列表
    brand_key_list=[]
    product_series_key_list=[]
    generate_questionnaire_model_key_list=[]
    task_type_key_list=[]
    
    for i in range(len(brand_list)):
        brand_key=brand_list[i]
        product_series_key=product_series_list[i]
        generate_questionnaire_model_key=generate_questionnaire_model_list[i]
        task_type_key=task_type_list[i]

        brand_key_list.append(brand_key)
        product_series_key_list.append(product_series_key)
        generate_questionnaire_model_key_list.append(generate_questionnaire_model_key)
        task_type_key_list.append(task_type_key)
        
        
        ques_path = f"datas/questionnaire/{brand_key}^{product_series_key}^{generate_questionnaire_model_key}^{task_type_key}^questionnaire.json"

        if not os.path.exists(ques_path):
            log.warning(f"[Skip ques] 未找到问卷：{ques_path}")

        else:
            with open(ques_path, "r", encoding="utf-8") as f:

                ques_json = json.load(f)
            ques_name_list.append(f"{brand_key}^{product_series_key}^{generate_questionnaire_model_key}^{task_type_key}")
            #print(type(ques_json))

            #dict_keys(['Question 1', 'Question 2', 'Question 3' ...

            # ques_json['Question 1'].keys()
            # dict_keys(['question', 'options', 'answer'])


            """
            question_list=[]
            for key in ques_json.keys():
                try:
                    ques_ques=ques_json[key]['question']
                except:
                    try:
                        ques_ques=ques_json[key]['questions']
                    except:
                        raise
                        
                ques_options=str(ques_json[key]['options'])
                question_list.append(ques_ques+ques_options)
            """


            #读取之后要把每个问卷解析成一个问题列表
            #ques_str=json.dumps(ques_json)
            #print(type(ques_str))

            ques_list_list.append(ques_json)

    print(f"问卷读取完成，共{len(ques_list_list)}份")


    #截取一部分用于实验
    """
    brand_key_list=brand_key_list[:20]
    product_series_key_list=product_series_key_list[:20]
    generate_questionnaire_model_key_list=generate_questionnaire_model_key_list[:20]
    task_type_key_list=task_type_key_list[:20]
    
    ques_list_list=ques_list_list[:20]
    """
    

    #exit()

    global model
    model = SentenceTransformer('all-MiniLM-L6-v2', token=os.getenv("HF_SENTENCE_TRANSFORMER_TOKEN"))
    print("模型加载完成")


    #richness= calculate_information_richness_batch(ques_list_list, ques_name_list)
    #diversity=calculate_lexical_diversity_batch(ques_list_list, ques_name_list)




    redundancy_scores= calculate_redundancy_batch(ques_list_list, ques_name_list)
    diversity_scores=calculate_lexical_diversity_batch(ques_list_list, ques_name_list)


    model_scores={}
    #为每个模型处理出一个最终总分
    for i in range(len(brand_key_list)):
        print(brand_key_list[i], product_series_key_list[i], generate_questionnaire_model_key_list[i], task_type_key_list[i])
        print(redundancy_scores[i], diversity_scores[i])

        key=(generate_questionnaire_model_key_list[i], task_type_key_list[i])
        if key not in model_scores.keys():
            model_scores[key]=[]
        
        model_scores[key].append([redundancy_scores[i], diversity_scores[i]])
        
    for key in model_scores.keys():
        key_redundancy_scores=[ i[0]   for i in model_scores[key]]
        key_diversity_scores=[ 1-i[1]   for i in model_scores[key]]     #做了修正
        import numpy as np
        #richness_sum=np.sum(key_richness_scores)
        #diversity_sum=np.sum(key_diversity_scores)
        redundancy_mean=np.sum(key_redundancy_scores) *100/len(key_redundancy_scores)
        
        
        diversity_mean=np.sum(key_diversity_scores)*len(key_diversity_scores)/100
        print(f"{key}:{redundancy_mean}:{diversity_mean}:{len(key_redundancy_scores)}")

      
    #把 brand_key_list, product_series_key_list, generate_questionnaire_model_key_list,\
    # task_type_key_list, diversity_scores,  sem_ent_scores,redundancy_scores ,\
    # lexical_diversity_scores, richness_scores  写入一个excel

    import pandas as pd

    # 创建数据字典
    data = {
        '品牌': brand_key_list,
        '产品系列': product_series_key_list,
        '问卷模型': generate_questionnaire_model_key_list,
        '任务类型': task_type_key_list,
        '信息丰富度': diversity_scores,
        '语义冗余度': redundancy_scores
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 导出到Excel文件
    output_file = '信息量评估结果.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')

    print(f"数据已成功写入Excel文件: {output_file}")

