import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_agent import MultiAgentSystem


def main():
    print("🚀 启动多智能体协作系统...")
    mas = MultiAgentSystem()

    while True:
        user = input("\n你：")
        if user.lower() == "quit":
            print("再见！")
            break

        answer = mas.run(user)
        print(f"\n🤖 助手：{answer}\n")


if __name__ == "__main__":
    main()