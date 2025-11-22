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
from sentence_transformers import SentenceTransformer, util
import spacy
import langid  # 用于语言检测
import jieba  # 中文分词
import jieba.posseg as pseg  # 中文词性标注和实体识别
from collections import defaultdict

# 确保安装了必要的库:
# pip install transformers sentence-transformers spacy langid jieba
# python -m spacy download en_core_web_sm
# python -m spacy download zh_core_web_sm

# class MultilingualTextSimilarityChecker:
#     def __init__(self):
#         # 加载多语言语义相似度模型
#         self.semantic_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
#         # 加载多语言事实一致性验证模型
#         self.consistency_model = pipeline(
#             "text-classification", 
#             model="vicgalle/xlm-roberta-large-xnli-anli",
#             device= 0 if torch.cuda.is_available() else -1
#         )
        
#         # 加载中文实体识别工具
#         self.nlp_zh = spacy.load("zh_core_web_sm")
        
#         # 加载英文实体识别工具
#         self.nlp_en = spacy.load("en_core_web_sm")
    
#     def detect_language(self, text):
#         """检测文本中的主要语言"""
#         lang, conf = langid.classify(text)
#         return lang
    
#     def preprocess_text(self, text):
#         """文本预处理，保留中英文混合特性"""
#         return ' '.join(text.split()).strip()
    
#     def extract_multilingual_entities(self, text):
#         """提取中英文混合文本中的实体"""
#         # 检测文本语言
#         lang = self.detect_language(text)
        
#         # 英文为主的处理
#         if lang == 'en':
#             doc = self.nlp_en(text)
#             return {ent.text: ent.label_ for ent in doc.ents}
        
#         # 中文为主的文本处理
#         elif lang == 'zh':
#             # 使用spacy处理中文实体
#             doc = self.nlp_zh(text)
#             spacy_entities = {ent.text: ent.label_ for ent in doc.ents}
            
#             # 使用jieba补充提取中文实体
#             jieba_entities = {}
#             words = pseg.cut(text)
#             for word, flag in words:
#                 if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构名、其他专名
#                     jieba_entities[word] = flag
            
#             # 合并结果
#             return {**spacy_entities, **jieba_entities}
        
#         # 混合语言文本处理
#         else:
#             # 分割中英文部分
#             chinese_parts = []
#             english_parts = []
#             current = ""
#             is_english = True
            
#             for char in text:
#                 if '\u4e00' <= char <= '\u9fff':  # 中文字符
#                     if is_english and current:
#                         english_parts.append(current)
#                         current = ""
#                     is_english = False
#                     current += char
#                 else:
#                     if not is_english and current:
#                         chinese_parts.append(current)
#                         current = ""
#                     is_english = True
#                     current += char
            
#             if current:
#                 if is_english:
#                     english_parts.append(current)
#                 else:
#                     chinese_parts.append(current)
            
#             # 分别处理中英文部分
#             entities = {}
            
#             # 处理中文部分
#             for part in chinese_parts:
#                 doc = self.nlp_zh(part)
#                 for ent in doc.ents:
#                     entities[ent.text] = ent.label_
#                 words = pseg.cut(part)
#                 for word, flag in words:
#                     if flag in ['nr', 'ns', 'nt', 'nz']:
#                         entities[word] = flag
            
#             # 处理英文部分
#             for part in english_parts:
#                 doc = self.nlp_en(part)
#                 for ent in doc.ents:
#                     entities[ent.text] = ent.label_
            
#             return entities
    
#     def compute_semantic_similarity(self, text1, text2):
#         """计算多语言语义相似度得分"""
#         embeddings = self.semantic_model.encode([text1, text2], convert_to_tensor=True)
#         cos_sim = util.cos_sim(embeddings[0], embeddings[1])
#         return cos_sim.item()
    
#     def check_factual_consistency(self, text1, text2):
#         """检查多语言文本的事实一致性"""
#         # 对于多语言模型，直接输入文本对
#         result = self.consistency_model(f"{text1} [SEP] {text2}", max_length=512, truncation=True)[0]
        
#         # 解析结果
#         label_score = {
#             'contradiction': 0.0,   # 矛盾
#             'neutral': 0.5,          # 中性
#             'entailment': 1.0        # 蕴含
#         }
        
#         return label_score.get(result['label'].lower(), 0.0), result['score']
    
#     def normalize_entity(self, entity):
#         """实体归一化处理（简繁体转换、大小写统一等）"""
#         # 简体转繁体（可扩展）
#         simplified_to_traditional = {
#             '苹果': '蘋果',
#             '华为': '華為'
#         }
        
#         # 大小写处理
#         if entity.lower() in ['apple', 'google', 'microsoft']:
#             return entity.lower().capitalize()
        
#         # 简体转繁体
#         if entity in simplified_to_traditional:
#             return simplified_to_traditional[entity]
        
#         return entity
    
#     def compare_multilingual_entities(self, text1, text2):
#         """比较两段混合语言文本的实体一致性"""
#         entities1 = self.extract_multilingual_entities(text1)
#         entities2 = self.extract_multilingual_entities(text2)
        
#         # 实体归一化处理
#         normalized_entities1 = {self.normalize_entity(k): v for k, v in entities1.items()}
#         normalized_entities2 = {self.normalize_entity(k): v for k, v in entities2.items()}
        
#         # 创建实体映射字典
#         entity_mapping = defaultdict(set)
#         for ent, label in normalized_entities1.items():
#             entity_mapping[ent].add(label)
        
#         # 检查一致性和冲突
#         consistent_count = 0
#         conflict_count = 0
#         total_shared = 0
        
#         for ent, label in normalized_entities2.items():
#             if ent in entity_mapping:
#                 total_shared += 1
#                 if label in entity_mapping[ent]:
#                     consistent_count += 1
#                 else:
#                     conflict_count += 1
        
#         # 计算一致性得分
#         if total_shared > 0:
#             consistency_score = consistent_count / total_shared
#             conflict_score = conflict_count / total_shared
#         else:
#             consistency_score = 0.5
#             conflict_score = 0
        
#         return {
#             "entity_score": consistency_score,
#             "conflict_count": conflict_count,
#             "shared_count": total_shared
#         }
    
#     def translate_keywords(self, text):
#         """翻译文本中的关键术语（简化版）"""
#         # 这里使用一个简化的术语翻译表，实际应用可以使用专业翻译API
#         term_dict = {
#             '手机': 'phone',
#             '电脑': 'computer',
#             '苹果': 'Apple',
#             '华为': 'Huawei',
#             '小米': 'Xiaomi',
#             'iPhone': 'iPhone',
#             'iPad': 'iPad',
#             'MacBook': 'MacBook',
#             'Microsoft': '微软',
#             'Google': '谷歌'
#         }
        
#         for term, translation in term_dict.items():
#             text = text.replace(term, translation)
#         return text
    
#     def preprocess_for_similarity(self, text):
#         """为相似度计算准备文本"""
#         # 替换常见术语的统一表达
#         return self.translate_keywords(text)
    
#     def get_similarity_score(self, text1, text2, semantic_weight=0.6, logic_weight=0.4):
#         """综合计算文本相似度分数"""
#         text1 = self.preprocess_text(text1)
#         text2 = self.preprocess_text(text2)
        
#         # 准备用于相似度计算的文本（统一术语表达）
#         sim_text1 = self.preprocess_for_similarity(text1)
#         sim_text2 = self.preprocess_for_similarity(text2)
        
#         # 计算语义相似度
#         semantic_score = self.compute_semantic_similarity(sim_text1, sim_text2)
        
#         # 检查事实一致性
#         consistency_score, confidence = self.check_factual_consistency(text1, text2)
        
#         # 检查实体一致性
#         entity_result = self.compare_multilingual_entities(text1, text2)
#         entity_score = entity_result["entity_score"]
        
#         # 加权计算逻辑一致性
#         logic_score = (consistency_score * 0.7) + (entity_score * 0.3)
        
#         # 总评分
#         final_score = (semantic_score * semantic_weight) + (logic_score * logic_weight)
        
#         # 调整分数：如果存在实体冲突，降低分数
#         if entity_result["conflict_count"] > 0:
#             conflict_penalty = min(0.2, 0.05 * entity_result["conflict_count"])
#             final_score = max(0, final_score - conflict_penalty)
        
#         # 确定相似级别
#         if final_score >= 0.85:
#             similarity_level = "高度相似"
#         elif final_score >= 0.70:
#             similarity_level = "相似"
#         elif final_score >= 0.50:
#             similarity_level = "中等相似"
#         else:
#             similarity_level = "不相似"
        
#         return {
#             "final_score": round(final_score, 3),
#             "similarity_level": similarity_level,
#             "semantic_similarity": round(semantic_score, 3),
#             "factual_consistency": round(consistency_score, 3),
#             "entity_consistency": round(entity_score, 3),
#             "logic_score": round(logic_score, 3),
#             "entity_comparison": entity_result,
#             "detected_language_text1": self.detect_language(text1),
#             "detected_language_text2": self.detect_language(text2)
#         }
    
#     def is_similar(self, text1, text2, threshold=0.9):
#         """判断两段文本是否足够相似且没有事实逻辑错误"""
#         score_data = self.get_similarity_score(text1, text2)
#         return score_data['final_score'] >= threshold and score_data['entity_comparison']['conflict_count'] == 0





def accuracy_reward(response: str, ground_truth: str,checker,global_step = 0) -> float:
    # checker = MultilingualTextSimilarityChecker()
    threshold = checker.min_threshold+(checker.max_threshold-checker.min_threshold)*(global_step/checker.steps) if global_step<=checker.steps else checker.max_threshold
    result = checker.is_similar(response, ground_truth,threshold=threshold)
    # print(f"文本1: {response}")
    # print(f"文本2: {ground_truth}")
    # print(f"综合相似度: {score} )")
    # print(f"事实一致性: {result['factual_consistency']}")
    # print(f"实体一致性: {result['entity_consistency']}")
    # print(f"检测语言1: {result['detected_language_text1']}, 语言2: {result['detected_language_text2']}")
    # print("实体比较结果:", result['entity_comparison'])

    return 1.0 if result else -1.0


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    checker,
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for dapo reward function.")

    scores = []
    for reward_input in reward_inputs:
        accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"],checker,reward_input["global_step"])
        scores.append(
            {
                "overall": accuracy_score, #+ overlong_score * overlong_penalty_factor,
                "accuracy": accuracy_score,
                "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
            }
        )

    return scores

# # 使用示例
# if __name__ == "__main__":
    
#     checker = MultilingualTextSimilarityChecker()
    
#     # 示例1: 中英文互译相似文本
#     text1 = "苹果公司昨天发布了新的iPhone 15系列手机"
#     text2 = "Apple released the new iPhone 15 series of smartphones yesterday"
#     print(accuracy_reward(text1,text2))

#     result = checker.get_similarity_score(text1, text2)
#     print(f"文本1: {text1}")
#     print(f"文本2: {text2}")
#     print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
#     print(f"事实一致性: {result['factual_consistency']}")
#     print(f"实体一致性: {result['entity_consistency']}")
#     print(f"检测语言1: {result['detected_language_text1']}, 语言2: {result['detected_language_text2']}")
#     print("实体比较结果:", result['entity_comparison'])
    
#     # 示例2: 存在逻辑错误的中英文混合文本
#     text3 = "苹果公司宣布将在中国市场上停止销售iPhone"
#     text4 = "Apple announced it will increase iPhone sales in the Chinese market"
#     result = checker.get_similarity_score(text3, text4)
#     print(f"\n文本3: {text3}")
#     print(f"文本4: {text4}")
#     print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
#     print(f"事实一致性: {result['factual_consistency']}")
#     print(f"实体一致性: {result['entity_consistency']}")
#     print(f"冲突实体数: {result['entity_comparison']['conflict_count']}")
    
#     # 示例3: 中英文混合文本
#     text5 = "华为的Mate 60 Pro 智能手机has received high praise for its innovative features"
#     text6 = "Huawei's Mate 60 Pro smartphone因其创新特性而受到高度评价"
#     result = checker.get_similarity_score(text5, text6)
#     print(f"\n文本5: {text5}")
#     print(f"文本6: {text6}")
#     print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
    
#     # 示例4: 存在实体冲突的文本
#     text7 = "苹果公司与微软合作开发新操作系统"
#     text8 = "Apple and Google are collaborating on a new operating system"
#     result = checker.get_similarity_score(text7, text8)
#     print(f"\n文本7: {text7}")
#     print(f"文本8: {text8}")
#     print(f"综合相似度: {result['final_score']} ({result['similarity_level']})")
#     print(f"冲突实体数: {result['entity_comparison']['conflict_count']}")
#     print("相似性判断结果:", checker.is_similar(text7, text8))
