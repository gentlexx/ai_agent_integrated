import os
from typing import List, Optional
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
            raise RuntimeError(f"!!!hf本地模型不存在: {model_path}")

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

        return (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, question: str) -> Optional[str]:
        """使用 LLM 生成回答"""
        if self.chain is None:
            return None
        return self.chain.invoke(question)

    def ask_direct(self, question: str) -> Optional[str]:
        """直接返回检索到的原文（不经过 LLM）"""
        if self.retriever is None:
            return None

        try:
            docs = self.retriever.invoke(question)
            if docs:
                return docs[0].page_content
            return None
        except Exception as e:
            print(f"[RAG直接检索] 错误: {e}")
            return self.ask(question)

    def is_empty(self) -> bool:
        """检查知识库是否为空"""
        return self.chain is None

    def reload(self):
        """重新加载知识库"""
        print("🔄 RAG: 重新加载知识库...")
        self.chain = self._build_chain()
        if self.chain:
            print("✅ RAG: 知识库已更新")
        else:
            print("❌ RAG: 知识库为空")
        return self.chain