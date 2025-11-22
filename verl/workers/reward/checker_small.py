from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

class LightweightTextSimilarity:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu',min_threshold = 0.7,max_threshold = 0.8,steps = 100):
        """
        初始化轻量级相似度匹配模型
        :param model_name: 模型名称（默认小型多语言模型）
        :param device: 运行设备（默认CPU，可选 'cuda'）
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)  # 将模型移至指定设备
        print(f"Loaded model: {model_name} on {device}")
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.steps = steps

    def encode_texts(self, texts, batch_size=32):
        """
        将文本列表编码为向量
        :param texts: 文本列表（支持中英文混合）
        :param batch_size: 批处理大小（提升效率）
        :return: 文本向量数组 [n, 384]
        """
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

    def calculate_similarity(self, text1, text2):
        """
        计算两段文本的相似度（余弦值）
        :return: 相似度分数 [0-1]
        """
        embeddings = self.encode_texts([text1, text2])
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def find_most_similar(self, query, corpus, top_k=3):
        """
        在语料库中查找与查询最相似的文本
        :param query: 查询文本
        :param corpus: 语料库（文本列表）
        :param top_k: 返回最相似的数量
        :return: 匹配结果列表 [(文本, 相似度)]
        """
        query_embed = self.encode_texts([query])
        corpus_embeds = self.encode_texts(corpus)
        cos_scores = util.cos_sim(query_embed, corpus_embeds)[0]
        
        # 获取Top-K结果
        top_results = torch.topk(cos_scores, k=top_k)
        return [(corpus[idx], score.item()) for score, idx in zip(top_results.values, top_results.indices)]
    def is_similar(self, respnse, ground_truth,threshold):
        score = self.calculate_similarity(respnse, ground_truth)
        return score >= threshold

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 初始化模型（自动下载预训练权重）
    similarity_model = LightweightTextSimilarity(device='cpu')  # 使用GPU时改为 'cuda'

    # 示例1：计算两段混合文本的相似度
    text_a = "Natural language processing (NLP) 是人工智能的重要方向"
    text_b = "NLP 技术让计算机理解人类语言"
    similarity_score = similarity_model.calculate_similarity(text_a, text_b)
    print(f"相似度: {similarity_score:.4f}")  # 输出示例: 相似度: 0.7823

    # 示例2：在语料库中搜索相似文本
    corpus = [
        "机器学习需要大量数据和算力",
        "深度学习在图像识别中表现突出",
        "NLP enables computers to understand human languages",
        "人工智能的未来在于多模态学习",
        "自然语言处理是AI的核心技术之一"
    ]
    query = "NLP如何让机器理解语言？"
    results = similarity_model.find_most_similar(query, corpus, top_k=2)
    
    print("\n最相似结果:")
    for i, (text, score) in enumerate(results):
        print(f"{i+1}. {text} (Score: {score:.4f})")