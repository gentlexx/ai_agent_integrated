import os
from typing import List, Optional, Union
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGEngine:
    def __init__(self, llm, knowledge_path=None):
        self.llm = llm
        if knowledge_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_path = os.path.join(current_dir, "knowledge_base")
        self.knowledge_path = knowledge_path
        self.retriever = None  # 保存检索器
        self.chain = self._build_chain()

    def _build_chain(self):
        loader = DirectoryLoader(
            self.knowledge_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs = loader.load()
        print(f"(rag_engine)-加载了 {len(docs)} 个文档")
        print(f"(rag_engine)-文档路径：{self.knowledge_path}")

        if not docs:
            print("ERROR 没有找到任何文档！")
            return None

        for doc in docs:
            print(f"(rag_engine)-文档内容预览：\n{doc.page_content[:100]}...")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 使用本地模型路径
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "models", "text2vec-base-chinese")

        if not os.path.exists(model_path):
            raise RuntimeError(f"(rag-error) 本地模型不存在: {model_path}")

        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': False}
        )

        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)

        vectorstore = Chroma.from_documents(chunks, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是知识库问答助手。

规则：
1. 只使用【上下文】中的信息
2. 答案不超过3句话
3. 不要添加解释或你的知识
4. 如果上下文没有答案，只说：未找到相关信息

上下文：
{context}

回答："""),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        print(f"(rag-init) 知识库构建完成，共 {len(chunks)} 个文本块")

        return (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, question: str) -> Optional[str]:
        """使用 LLM 生成回答（会改写原文）"""
        if self.chain is None:
            print("(rag-warn) 知识库为空，无法回答")
            return None

        print(f"(rag-ask) 问题: {question[:50]}...")
        result = self.chain.invoke(question)
        print(f"(rag-ask) 回答: {result[:100]}...")
        return result

    def ask_direct(self, question: str, k: int = 1) -> Optional[Union[str, List[str]]]:
        """
        直接返回检索到的原文（不经过 LLM）

        用于强制 RAG 场景：只返回知识库原文，不让模型改写

        Args:
            question: 查询问题
            k: 返回文档数量
               - k=1（默认）：返回单条原文（字符串）
               - k>1：返回多条原文（列表）

        Returns:
            k=1 时返回字符串，k>1 时返回字符串列表，无结果时返回 None
        """
        if self.retriever is None:
            print("(rag-warn) 检索器未初始化，无法直接检索")
            return None

        try:
            docs = self.retriever.invoke(question)
            if not docs:
                print("(rag-warn) 未找到相关文档")
                return None

            if k == 1:
                result = docs[0].page_content
                print(f"(rag-direct) 直接检索成功，返回原文 {len(result)} 字符")
                print(f"(rag-direct) 内容预览: {result[:100]}...")
                return result
            else:
                results = [doc.page_content for doc in docs[:k]]
                print(f"(rag-direct) 直接检索成功，返回 {len(results)} 条原文")
                return results
        except Exception as e:
            print(f"(rag-error) 直接检索失败: {e}")
            return None

    def is_empty(self) -> bool:
        """检查知识库是否为空"""
        return self.chain is None

    def reload(self):
        """重新加载知识库"""
        print("(rag-reload) 重新加载知识库...")
        self.chain = self._build_chain()
        if self.chain:
            print("(rag-reload) 知识库已更新")
        else:
            print("(rag-reload) 知识库为空")
        return self.chain

    def get_retriever(self):
        """获取检索器（用于调试）"""
        return self.retriever