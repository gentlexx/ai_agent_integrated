import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from src.config import get_model_config
from src.rag_engine import RAGEngine
from src.agent_manager import AgentManager


def main():
    llm = ChatOpenAI(**get_model_config())
    rag = RAGEngine(llm)
    agent = AgentManager(llm)

    print("系统就绪（RAG + Agent）\n")

    while True:
        user = input("你：")
        if user.lower() == "quit":
            break

        if any(k in user for k in ["政策", "年假", "加班", "培训", "餐补"]):
            answer = rag.ask(user)
            if answer:
                print(f"助手(RAG): {answer}\n")
                continue

        answer = agent.ask(user)
        print(f"助手(Agent): {answer}\n")


if __name__ == "__main__":
    main()