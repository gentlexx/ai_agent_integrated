import os
import numpy as np
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

        # 初始化 ZhiPuZAIEmbedding（用于语义搜索）
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings
            from src.config import get_api_key
            self.embeddings = ZhipuAIEmbeddings(
                api_key=get_api_key(),
                model="embedding-2"
            )
            self.use_embedding = True
            print("<路由器>：Embedding 已初始化")
        except Exception as e:
            print(f"<路由器>：Embedding 初始化失败 {e}，将使用 LLM 判断")
            self.use_embedding = False


        # 动态加载知识库摘要
        self.kb_summary = self._load_kb_summary()

        # 缓存知识库向量（用于快速相似度匹配）
        self.kb_vectors: Optional[List] = None  # 👈 使用 List 类型提示
        self.kb_chunks: Optional[List] = None
        if self.use_embedding:
            self._cache_kb_vectors()

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
            print(f"加载知识库失败: {e}")

        if not all_content:
            return "暂无知识库内容"

        return "\n\n".join(all_content)

    def _cache_kb_vectors(self):
        """缓存知识库所有文档的向量"""
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
                print("<!路由器>：知识库为空")
                return

            # 分块处理
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.kb_chunks = splitter.split_documents(cache_docs)

            # 计算所有块的向量
            texts = [chunk.page_content for chunk in self.kb_chunks]
            self.kb_vectors = self.embeddings.embed_documents(texts)

            print(f"<路由器>：已缓存 {len(self.kb_vectors)} 个知识块向量")
        except Exception as e:
            print(f"<!路由器>：向量缓存失败 {e}")

    def reload(self):
        """重新加载知识库（上传新文件后调用）"""
        print("🔄<路由器>：重新加载知识库...")
        self.kb_summary = self._load_kb_summary()
        if self.use_embedding:
            self._cache_kb_vectors()
        print("<路由器>：知识库已更新")

    def semantic_similarity_route(self, question: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        使用语义相似度判断：问题是否与知识库内容相关

        Returns:
            (是否走RAG, 最高相似度)
        """
        if not self.use_embedding or self.kb_vectors is None or len(self.kb_vectors) == 0:
            return None, 0.0

        try:
            # 计算问题的向量
            question_vector = self.embeddings.embed_query(question)

            # 计算与所有知识块的最大相似度
            similarities = []
            for kb_vector in self.kb_vectors:
                sim = self._cosine_similarity(question_vector, kb_vector)
                similarities.append(sim)

            max_sim = max(similarities)
            is_rag = max_sim >= threshold

            print(f"<语义路由> 最高相似度: {max_sim:.3f}, 阈值: {threshold}, 走RAG: {is_rag}")
            return is_rag, max_sim

        except Exception as e:
            print(f"<语义路由> 错误: {e}")
            return None, 0.0

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
            print(f"<LLM路由> 决策: {decision}")
            return "rag" if decision == "rag" else "agent"
        except Exception as e:
            print(f"<LLM路由> 错误: {e}")
            return "agent"  # 默认走 agent

    def route(self, question: str) -> str:
        """
        智能路由决策（优先使用语义相似度）

        优先级：
        1. 语义相似度（快但可能有误差）
        2. LLM 判断（准但稍慢）
        """
        # 先用语义相似度快速判断
        is_rag, similarity = self.semantic_similarity_route(question, threshold=0.25)

        if is_rag is not None:
            if similarity > 0.25:
                return "rag"
            elif similarity < 0.1:
                return "agent"
            # 相似度在中间范围，用 LLM 确认

        # 使用 LLM 判断
        return self.llm_route(question)