import os
import re
import jieba
import numpy as np
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    from rank_bm25 import  BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("<router-warn> rank_bm25 未安装，混合检索不可用，请执行: pip install rank_bm25")

class Router:
    def __init__(self, llm=None, knowledge_path: str = None):
        """
        动态路由器：根据知识库实际内容判断路由

        Args:
            llm: 语言模型
            knowledge_path: 知识库文件夹路径
        """
        self.llm = llm

        # 设置知识库路径
        if knowledge_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_path = os.path.join(current_dir, "knowledge_base")
        self.knowledge_path = knowledge_path

        # 混合检索配置
        self.hybrid_alpha = 0.6  # BM25 权重（向量权重为 1-alpha）
        self.route_threshold = 0.35  # 混合检索阈值

        # 强制 RAG 关键词（用于前置匹配）
        self.force_rag_keywords = ["知识库", "文件", "rag", "RAG", "在知识库中", "在文件中", "查找", "文档", "资料"]

        # 初始化 ZhiPuZAIEmbedding（用于语义搜索）
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings
            from src.config import get_api_key
            self.embeddings = ZhipuAIEmbeddings(
                api_key=get_api_key(),
                model="embedding-2"
            )
            self.use_embedding = True
            print("<router-init> Embedding 已初始化")
        except Exception as e:
            print(f"<router-init> Embedding 初始化失败 {e}，将使用 LLM 判断")
            self.use_embedding = False


        # 动态加载知识库摘要
        self.kb_summary = self._load_kb_summary()

        # 缓存知识库向量和 BM25 索引
        self.kb_vectors: Optional[List] = None
        self.kb_chunks: Optional[List] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_chunks: Optional[List[List[str]]] = None

        if self.use_embedding:
            self._cache_kb_vectors()

    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 分词（支持中英文）"""
        if not text or not text.strip():
            return []
        # 去除标点符号，保留中文、英文、数字
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', text)
        # 使用 jieba 精确模式分词
        tokens = jieba.lcut(text)
        # 过滤掉单字符和空白，转为小写（英文）
        tokens = [t.lower().strip() for t in tokens if len(t.strip()) >= 2]
        return tokens

    def _load_kb_summary(self) -> str:
        """加载知识库内容摘要"""
        if not os.path.exists(self.knowledge_path):
            return "暂无知识库内容"

        all_content = []
        try:
            loader = DirectoryLoader(
                self.knowledge_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=True,
            )
            docs_summary = loader.load()

            for doc in docs_summary:
                # 提取文件名和内容预览
                filename = os.path.basename(doc.metadata.get('source', ''))
                content_preview = doc.page_content[:500]  # 只取前500字
                all_content.append(f"【文件：{filename}】\n{content_preview}")
        except Exception as e:
            print(f"<router-error> 加载知识库失败: {e}")

        if not all_content:
            return "暂无知识库内容"

        return "\n\n".join(all_content)

    def _cache_kb_vectors(self):
        """缓存知识库所有文档的向量和 BM25 索引"""
        if not self.use_embedding:
            return

        try:
            loader = DirectoryLoader(
                self.knowledge_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=True,
            )
            cache_docs = loader.load()

            if not cache_docs:
                print("<router-warn> 知识库为空")
                return

            # 分块处理
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.kb_chunks = splitter.split_documents(cache_docs)

            # 计算所有块的向量
            texts = [chunk.page_content for chunk in self.kb_chunks]
            self.kb_vectors = self.embeddings.embed_documents(texts)

            # 构建 BM25 索引
            if BM25_AVAILABLE:
                self.tokenized_chunks = [self._tokenize(chunk.page_content) for chunk in self.kb_chunks]
                self.bm25_index = BM25Okapi(self.tokenized_chunks)
                print(f"<router-init> 已缓存 {len(self.kb_vectors)} 个知识块向量，BM25 索引已构建")
            else:
                print(f"<router-init> 已缓存 {len(self.kb_vectors)} 个知识块向量（BM25 不可用）")

        except Exception as e:
            print(f"<router-error> 向量缓存失败 {e}")

    def reload(self):
        """重新加载知识库（上传新文件后调用）"""
        print("<router-reload> 重新加载知识库...")
        self.kb_summary = self._load_kb_summary()
        if self.use_embedding:
            self._cache_kb_vectors()
        print("<router-reload> 知识库已更新")


    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def hybrid_search(self, question: str) -> Tuple[float, Optional[str]]:
        """
        混合检索：BM25 + 向量相似度

        Returns:
            (综合得分, 匹配的内容片段)
        """
        if not self.use_embedding or self.kb_vectors is None or len(self.kb_vectors) == 0:
            return 0.0, None

        try:
            # 1. 向量检索得分
            question_vector = self.embeddings.embed_query(question)
            vector_scores = []
            for kb_vector in self.kb_vectors:
                sim = self._cosine_similarity(question_vector, kb_vector)
                vector_scores.append(sim)

            max_vector_score = max(vector_scores)
            best_vector_idx = vector_scores.index(max_vector_score)

            # 2. BM25 检索得分
            tokenized_question = self._tokenize(question)

            # 处理空查询
            if not tokenized_question:
                print("<router-hybrid> 查询为空，使用纯向量检索")
                return max_vector_score, self.kb_chunks[best_vector_idx].page_content[:200]

            bm25_scores = self.bm25_index.get_scores(tokenized_question)

            # 确保 BM25 分数非负（取 max(0, score)）
            bm25_scores = [max(0, score) for score in bm25_scores]
            max_bm25_score = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
            # ========== 新增：BM25无匹配时直接返回向量分 ==========
            if max_bm25_score == 0:
                print("<router-hybrid> BM25无匹配，使用纯向量分")
                return max_vector_score, self.kb_chunks[best_vector_idx].page_content[:200]
            best_bm25_idx = bm25_scores.index(max_bm25_score) if len(bm25_scores) > 0 else -1

            # 3. 归一化并加权融合
            # BM25 归一化：使用 tanh 或简单除以最大值（避免负数）
            if max_bm25_score > 0:
                normalized_bm25 = min(max_bm25_score / 10.0, 1.0)
            else:
                normalized_bm25 = 0.0

            hybrid_score = self.hybrid_alpha * normalized_bm25 + (1 - self.hybrid_alpha) * max_vector_score

            # 选择最佳匹配的块
            best_idx = best_vector_idx if max_vector_score >= normalized_bm25 else best_bm25_idx
            best_content = self.kb_chunks[best_idx].page_content[:200] if best_idx >= 0 else None

            print(
                f"<router-hybrid> 向量分: {max_vector_score:.3f}, BM25分: {normalized_bm25:.3f}, 混合分: {hybrid_score:.3f}")
            return hybrid_score, best_content

        except Exception as e:
            print(f"<router-error> 混合检索失败: {e}")
            return self._vector_search_only(question)

    def _vector_search_only(self, question: str) -> Tuple[float, Optional[str]]:
        """仅使用向量检索（降级方案）"""
        try:
            question_vector = self.embeddings.embed_query(question)
            similarities = []
            for kb_vector in self.kb_vectors:
                sim = self._cosine_similarity(question_vector, kb_vector)
                similarities.append(sim)

            max_sim = max(similarities)
            best_idx = similarities.index(max_sim)
            best_content = self.kb_chunks[best_idx].page_content[:200] if best_idx >= 0 else None

            print(f"<router-vector> 最高相似度: {max_sim:.3f}")
            return max_sim, best_content
        except Exception as e:
            print(f"<router-error> 向量检索失败: {e}")
            return 0.0, None

    def semantic_similarity_route(self, question: str, threshold: float = 0.5) -> Tuple[Optional[bool], float]:
        """
        使用语义相似度判断：问题是否与知识库内容相关

        Returns:
            (是否走RAG, 最高相似度)
        """
        if not self.use_embedding or self.kb_vectors is None or len(self.kb_vectors) == 0:
            return None, 0.0

        score, _ = self.hybrid_search(question)
        is_rag = score >= threshold

        print(f"<router-semantic> 混合得分: {score:.3f}, 阈值: {threshold}, 走RAG: {is_rag}")
        return is_rag, score

    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def llm_route(self, question: str) -> str:
        """使用 LLM 判断：问题是否能在知识库中找到答案"""
        prompt = f"""判断用户问题是否可以通过以下知识库回答。

## 知识库内容
{self.kb_summary[:2000]}

## 用户问题
{question}

## 判断规则
- 如果知识库中包含回答该问题所需的信息，输出：rag
- 如果知识库中没有相关信息，输出：agent

## 注意
- 只输出 rag 或 agent，不要有其他内容
- 如果知识库为空，输出 agent

输出："""

        try:
            response = self.llm.invoke(prompt)
            decision = response.content.strip().lower()
            print(f"<router-llm> 决策: {decision}")
            return "rag" if decision == "rag" else "agent"
        except Exception as e:
            print(f"<router-error> LLM路由失败: {e}")
            return "agent"  # 默认走 agent

    def force_rag_check(self, question: str) -> bool:
        """检查是否强制走 RAG（基于关键词匹配）"""
        question_lower = question.lower()
        for keyword in self.force_rag_keywords:
            if keyword.lower() in question_lower:
                print(f"<router-force> 命中强制RAG关键词: {keyword}")
                return True
        return False

    def route(self, question: str) -> str:
        """
        智能路由决策（优先混合检索）

        优先级：
        1. 强制 RAG 关键词（最高优先级）
        2. 混合检索（BM25 + 向量）
        3. LLM 判断（降级方案）
        """
        # 第一优先级：强制 RAG 关键词
        if self.force_rag_check(question):
            return "rag"

        # 第二优先级：混合检索
        if self.use_embedding and self.kb_vectors is not None:
            score, matched_content = self.hybrid_search(question)

            if score > self.route_threshold:
                print(f"<router-decision> 混合检索得分 {score:.3f} > {self.route_threshold}，走 RAG")
                if matched_content:
                    print(f"<router-match> 匹配内容预览: {matched_content[:100]}...")
                return "rag"
            elif score < 0.15:
                print(f"<router-decision> 混合检索得分 {score:.3f} < 0.15，走 Agent")
                return "agent"
            # 得分在中间范围，继续用 LLM 确认

        # 第三优先级：LLM 判断
        print("<router-decision> 使用 LLM 进行最终判断")
        return self.llm_route(question)