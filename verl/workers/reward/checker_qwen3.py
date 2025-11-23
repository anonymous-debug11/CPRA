# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

# from mathruler.grader import extract_boxed_content, grade_answer


# def accuracy_reward(response: str, ground_truth: str) -> float:

#     return 1.0 if grade_answer(response, ground_truth) else -1.0


# def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
#     expected_len = max_response_length - overlong_buffer_length
#     if response_length <= expected_len:
#         return 0.0
#     elif response_length <= max_response_length:
#         return (expected_len - response_length) / overlong_buffer_length
#     else:
#         return -1.0


# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
# ) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for dapo reward function.")

#     scores = []
#     for reward_input in reward_inputs:
#         accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
#         scores.append(
#             {
#                 "overall": accuracy_score, #+ overlong_score * overlong_penalty_factor,
#                 "accuracy": accuracy_score,
#                 "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
#             }
#         )

#     return scores





import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# from sentence_transformers import SentenceTransformer, util
# import spacy
# import langid  # 用于语言检测
# import jieba  # 中文分词
# import jieba.posseg as pseg  # 中文词性标注和实体识别
from collections import defaultdict

from qwen_vl_utils import process_vision_info
# from transformers import AutoProcessor,AutoTokenizer
# from vllm import LLM, SamplingParams
import os
import re
from openai import OpenAI
# # 确保安装了必要的库:
# # pip install transformers sentence-transformers spacy langid jieba
# # python -m spacy download en_core_web_sm
# # python -m spacy download zh_core_web_sm
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"] = "1"
class Qwen3Checker:
    def __init__(self,min_threshold = 0.7,max_threshold = 0.8,steps = 100,tensor_parallel_size=4):
        print('tensor_parallel_size',tensor_parallel_size)

        # 加载多语言语义相似度模型
        # self.llm = llm
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.steps = steps
        openai_api_key = "Empty"
        openai_api_base = "http://127.0.0.1:8080/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        # self.llm = LLM('/apdcephfs_qy3/share_301069248/users/yominyan/qwen3/Qwen3-4B-Thinking-2507/',trust_remote_code=True,tensor_parallel_size=tensor_parallel_size,dtype = 'auto',gpu_memory_utilization=0.23,enable_expert_parallel=False,max_model_len=3000)
        # self.tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_301069248/users/yominyan/qwen3/Qwen3-4B-Thinking-2507/")

        # Configurae the sampling parameters (for thinking mode)
        # self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20,max_tokens=25000)
        print('*'*90)
        
    def extract_score(self, text):
        """从"总分：X分"格式中提取分数"""
        pattern = r'总分[：:]\s*(\d+(?:\.\d+)?)分?'
        match = re.search(pattern, text)
        print(text)
        print(match)
        return match.group(1) if match else None

    def is_similar(self, response, ground_truth,threshold=0.6):


        # Modify OpenAI's API key and API base to use vLLM's API server.
        
        # completion = self.client.completions.create(
        #     model="Qwen/Qwen2.5-1.5B-Instruct",
        #     prompt="San Francisco is a",
        # )
        print('dive into similarity')
        openai_api_key = "none"
        openai_api_base = "http://0.0.0.0:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        print('construct successful')
        pmpt = f"""请基于语义一致性对以下两段文本进行详细打分（总分0-100分），核心判断聚焦“语义本质统一、无逻辑矛盾、细节匹配度”，无需纠结文字表述形式的差异。具体要求如下：

            ### 打分维度与标准（总分100分）
            1. **核心语义相似度（40分）**
            - 35-40分：两段文本核心观点、核心信息完全一致，本质含义无差异；
            - 25-34分：核心语义基本一致，仅一方省略少量非关键信息，不影响整体理解；
            - 10-24分：核心语义部分重合，但存在明显的信息偏差或重点偏移；
            - 0-9分：核心语义完全不同，无本质关联。

            2. **逻辑一致性（40分）**
            - 35-40分：无任何逻辑矛盾，判断、因果、条件等逻辑关系完全统一（如一段说“A是B”，另一段明确支持或不否定；一段说“C导致D”，另一段完全呼应该因果）；
            - 25-34分：无直接逻辑矛盾，但逻辑表述不够严谨（如一段明确因果，另一段仅陈述现象，不冲突但未明确呼应）；
            - 0-24分：存在明显逻辑矛盾（如对立判断、反向因果、矛盾条件等，如“A是B”与“A不是B”）。

            3. **细节信息匹配度（20分）**
            - 16-20分：关键细节（数量、范围、对象、程度、时间等）完全一致或无实质性差异；
            - 10-15分：非核心细节存在差异（如表述风格、次要修饰词），但不影响核心语义与逻辑；
            - 0-9分：关键细节存在显著偏差（如数量错误、对象混淆、程度矛盾），可能误导对核心语义的理解。


            整体分析精简一些，不要过于冗余，在1000字以内完成分析和总结。

            ### 输出要求
            1. 先给出总分，再分维度列出得分；

            文本1：{response}
            文本2：{ground_truth}
            """
        completion = self.client.completions.create(
                model="/apdcephfs_qy3/share_301069248/users/yominyan/qwen3/Qwen3-4B-Thinking-2507/",
                prompt=pmpt,
                # messages=[
                #     # {"role": "system", "content": "You are a chatbot."},
                #     {"role": "user", "content": pmpt},
                # ],
                max_tokens=2500,
            )
        try:
            
            # print(completion)
            print(completion.choices[0].text)
            return float(self.extract_score(completion.choices[0].text))/100 >=threshold
        except:
            return False
    
    def is_similar_bk(self, response, ground_truth,threshold=0.6):
        # format_prompt = """
        #     评估要求：
        #     1. 判断模型生成的内容是否与真实的描述相符
        #     2. 评估回答是否准确、合理、符合事实
        #     3. 判断是否存在逻辑错误或事实错误
        #     4. 评估整体表述的准确性

        #     请按以下格式回复：
        #     【图片相关性】: [高/中/低] - 图片内容与问答对的相关程度
        #     【回答准确性】: [准确/基本准确/部分准确/不准确] - 回答的准确程度
        #     【逻辑合理性】: [合理/基本合理/部分合理/不合理] - 回答的逻辑合理性
        #     【事实正确性】: [正确/基本正确/部分正确/不正确] - 回答的事实正确性
        #     【总体评价】: [优秀/良好/一般/较差] - 整体评价
        #     【详细分析】: 提供具体的分析说明，指出存在的问题和改进建议
        #     """
        messages = [
                {"role":"user",
                "content":f"""请基于语义一致性对以下两段文本进行详细打分（总分0-100分），核心判断聚焦“语义本质统一、无逻辑矛盾、细节匹配度”，无需纠结文字表述形式的差异。具体要求如下：

            ### 打分维度与标准（总分100分）
            1. **核心语义相似度（40分）**
            - 35-40分：两段文本核心观点、核心信息完全一致，本质含义无差异；
            - 25-34分：核心语义基本一致，仅一方省略少量非关键信息，不影响整体理解；
            - 10-24分：核心语义部分重合，但存在明显的信息偏差或重点偏移；
            - 0-9分：核心语义完全不同，无本质关联。

            2. **逻辑一致性（40分）**
            - 35-40分：无任何逻辑矛盾，判断、因果、条件等逻辑关系完全统一（如一段说“A是B”，另一段明确支持或不否定；一段说“C导致D”，另一段完全呼应该因果）；
            - 25-34分：无直接逻辑矛盾，但逻辑表述不够严谨（如一段明确因果，另一段仅陈述现象，不冲突但未明确呼应）；
            - 0-24分：存在明显逻辑矛盾（如对立判断、反向因果、矛盾条件等，如“A是B”与“A不是B”）。

            3. **细节信息匹配度（20分）**
            - 16-20分：关键细节（数量、范围、对象、程度、时间等）完全一致或无实质性差异；
            - 10-15分：非核心细节存在差异（如表述风格、次要修饰词），但不影响核心语义与逻辑；
            - 0-9分：关键细节存在显著偏差（如数量错误、对象混淆、程度矛盾），可能误导对核心语义的理解。


            整体分析精简一些，不要过于冗余，在1000字以内完成分析和总结。

            ### 输出要求
            1. 先给出总分，再分维度列出得分；

            文本1：{response}
            文本2：{ground_truth}
            """}]#            2. 详细说明打分理由：重点分析逻辑是否矛盾、细节是否匹配，同时补充核心语义的匹配情况；3. 最终给出“语义一致”“语义基本一致”“语义部分一致”“语义不一致”的定性结论。
        text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Set to False to strictly disable thinking
            )

        # Generate outputs
        outputs = self.llm.generate([text], self.sampling_params)
        # try:
        re = outputs[0].outputs[0].text.split('</think>')[1]
        print(re)
        try:
            return float(self.extract_score(re))/100 >=threshold
        except:
            return False







def accuracy_reward(response: str, ground_truth: str,checker : Qwen3Checker) -> float:
    # checker = MultilingualTextSimilarityChecker()
    print('accuracy_reward')
    result = checker.is_similar(response, ground_truth)

    return 1.0 if result else -1.0


# def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
#     expected_len = max_response_length - overlong_buffer_length
#     if response_length <= expected_len:
#         return 0.0
#     elif response_length <= max_response_length:
#         return (expected_len - response_length) / overlong_buffer_length
#     else:
#         return -1.0


# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
#     checker: MultilingualTextSimilarityChecker,
# ) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for dapo reward function.")

#     scores = []
#     for reward_input in reward_inputs:
#         accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"],checker)
#         scores.append(
#             {
#                 "overall": accuracy_score, #+ overlong_score * overlong_penalty_factor,
#                 "accuracy": accuracy_score,
#                 "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
#             }
#         )

#     return scores

# # 使用示例
if __name__ == "__main__":
    
    checker = Qwen3Checker()
    
    # 示例1: 中英文互译相似文本
    text1 = "苹果公司昨天发布了新的iPhone 15系列手机"
    text2 = "Apple released the new iPhone 15 series of smartphones yesterday"
    # print(accuracy_reward(text1,text2))

    result = checker.is_similar(text1, text2)
    print(result)
    # print(f"文本1: {text1}")
    # print(f"文本2: {text2}")
    # print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
    # print(f"事实一致性: {result['factual_consistency']}")
    # print(f"实体一致性: {result['entity_consistency']}")
    # print(f"检测语言1: {result['detected_language_text1']}, 语言2: {result['detected_language_text2']}")
    # print("实体比较结果:", result['entity_comparison'])
    
    # 示例2: 存在逻辑错误的中英文混合文本
    text3 = "苹果公司宣布将在中国市场上停止销售iPhone"
    text4 = "Apple announced it will increase iPhone sales in the Chinese market"
    result = checker.is_similar(text3, text4)
    print(result)
    # print(f"\n文本3: {text3}")
    # print(f"文本4: {text4}")
    # print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
    # print(f"事实一致性: {result['factual_consistency']}")
    # print(f"实体一致性: {result['entity_consistency']}")
    # print(f"冲突实体数: {result['entity_comparison']['conflict_count']}")
    
    # 示例3: 中英文混合文本
    text5 = "华为的Mate 60 Pro 智能手机has received high praise for its innovative features"
    text6 = "Huawei's Mate 60 Pro smartphone因其创新特性而受到高度评价"
    result = checker.is_similar(text5, text6)
    print(result)
    # print(f"\n文本5: {text5}")
    # print(f"文本6: {text6}")
    # print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
    
    # 示例4: 存在实体冲突的文本
    text7 = "苹果公司与微软合作开发新操作系统"
    text8 = "Apple and Google are collaborating on a new operating system"
    result = checker.is_similar(text7, text8)
    print(result)
    # print(f"\n文本7: {text7}")
    # print(f"文本8: {text8}")
    # print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
    # print(f"冲突实体数: {result['entity_comparison']['conflict_count']}")
    # print("相似性判断结果:", checker.is_similar(text7, text8))
