"""
统一测试入口
用法：
    python test.py                    # 交互式选择测试模块
    python test.py router             # 直接测试 router
    python test.py rag                # 直接测试 rag_engine
    python test.py multi              # 直接测试 multi_agent
    python test.py agent              # 直接测试 agent_manager
    python test.py all                # 测试所有模块
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_router():
    """测试路由器"""
    print_separator("测试 Router")

    from src.config import get_model_config
    from langchain_openai import ChatOpenAI
    from src.router import Router

    # 初始化
    llm_config = get_model_config()
    llm = ChatOpenAI(**llm_config)
    router = Router(llm=llm)

    print("\n【测试1】强制RAG关键词匹配")
    test_questions = [
        "在知识库中查找项目管理制度",
        "文件中有哪些政策",
        "rag 查询产品信息",
        "今天天气怎么样",
        "帮我记个笔记"
    ]

    for q in test_questions:
        result = router.route(q)
        force_check = router.force_rag_check(q)
        print(f"  问题: {q}")
        print(f"    强制RAG: {force_check}, 路由结果: {result}")

    print("\n【测试2】混合检索得分")
    if router.kb_vectors:
        score, content = router.hybrid_search("项目管理")
        print(f"  查询: 项目管理")
        print(f"    混合得分: {score:.3f}")
        if content:
            print(f"    匹配内容: {content[:100]}...")
    else:
        print("  知识库为空，跳过混合检索测试")

    print("\n【测试3】路由决策")
    test_q = "公司加班政策是什么"
    result = router.route(test_q)
    print(f"  问题: {test_q}")
    print(f"  决策: {result}")


def test_rag():
    """测试RAG引擎"""
    print_separator("测试 RAGEngine")

    from src.config import get_model_config
    from langchain_openai import ChatOpenAI
    from src.rag_engine import RAGEngine

    # 初始化
    llm_config = get_model_config()
    llm = ChatOpenAI(**llm_config)
    rag = RAGEngine(llm)

    if rag.is_empty():
        print("⚠️ 知识库为空，请先在 knowledge_base 目录下添加文件")
        return

    print("\n【测试1】直接检索（ask_direct）")
    test_q = "项目管理制度"
    result = rag.ask_direct(test_q)
    if result:
        print(f"  问题: {test_q}")
        print(f"  返回类型: {type(result).__name__}")
        print(f"  内容预览: {result[:200]}...")
    else:
        print(f"  未找到相关内容")

    print("\n【测试2】返回多条原文（ask_direct k=3）")
    results = rag.ask_direct(test_q, k=3)
    if results:
        print(f"  返回 {len(results)} 条原文")
        for i, r in enumerate(results):
            print(f"    第{i + 1}条: {r[:80]}...")

    print("\n【测试3】LLM改写（ask）")
    result = rag.ask(test_q)
    if result:
        print(f"  问题: {test_q}")
        print(f"  LLM回答: {result}")

    print("\n【测试4】对比原文 vs 改写")
    original = rag.ask_direct(test_q)
    rewritten = rag.ask(test_q)
    print(f"  原文长度: {len(original) if original else 0} 字符")
    print(f"  改写长度: {len(rewritten) if rewritten else 0} 字符")


def test_multi():
    """测试多智能体系统"""
    print_separator("测试 MultiAgentSystem")

    from src.multi_agent import MultiAgentSystem

    system = MultiAgentSystem()

    print("\n【测试1】强制RAG触发")
    result = system.run("在知识库中查找项目管理制度")
    print(f"  最终答案: {result['final_answer'][:200]}...")
    print(f"\n  思考过程关键步骤:")
    for line in result['process'].split('\n'):
        if any(x in line for x in ['判断', '规划决策', '强制', '执行']):
            print(f"    {line}")

    print("\n【测试2】多轮对话记忆")
    system.run("我叫张三")
    result = system.run("我叫什么名字")
    print(f"  问题: 我叫什么名字")
    print(f"  回答: {result['final_answer']}")
    if "张三" in result['final_answer']:
        print("  ✅ 记忆功能正常")
    else:
        print("  ❌ 记忆功能异常")

    print("\n【测试3】历史关联判断")
    system.run("我喜欢吃西瓜")
    result = system.run("它甜吗")
    print(f"  问题: 它甜吗")
    print(f"  回答: {result['final_answer']}")
    # 检查思考过程是否包含关联判断
    if "判断" in result['process']:
        print("  ✅ 历史关联判断已执行")
    else:
        print("  ⚠️ 未检测到历史关联判断")

    print("\n【测试4】查看完整思考过程")
    result = system.run("你好")
    print(f"  过程:\n{result['process']}")


def test_agent():
    """测试Agent管理器"""
    print_separator("测试 AgentManager")

    from src.config import get_model_config
    from langchain_openai import ChatOpenAI
    from src.agent_manager import AgentManager

    # 初始化
    llm_config = get_model_config()
    llm = ChatOpenAI(**llm_config)
    agent = AgentManager(llm)

    print("\n【测试1】基础问答")
    test_q = "你好，介绍一下你自己"
    result = agent.ask(test_q)
    print(f"  问题: {test_q}")
    print(f"  回答: {result[:150]}...")

    print("\n【测试2】工具调用 - 天气")
    test_q = "北京今天天气怎么样"
    result = agent.ask(test_q)
    print(f"  问题: {test_q}")
    print(f"  回答: {result[:150]}...")

    print("\n【测试3】工具调用 - 时间")
    test_q = "现在几点了"
    result = agent.ask(test_q)
    print(f"  问题: {test_q}")
    print(f"  回答: {result}")


def test_all():
    """测试所有模块"""
    print_separator("测试所有模块")

    test_router()
    test_rag()
    test_agent()
    test_multi()


def interactive_mode():
    """交互式选择测试模块"""
    print_separator("统一测试入口")
    print("\n可选测试模块:")
    print("  1. router     - 路由器（混合检索、路由决策）")
    print("  2. rag        - RAG引擎（知识库检索）")
    print("  3. multi      - 多智能体系统（完整流程）")
    print("  4. agent      - Agent管理器（工具调用）")
    print("  5. all        - 测试所有模块")
    print("  6. quick      - 快速测试（预设用例）")
    print("  0. exit       - 退出")

    choice = input("\n请输入序号或模块名: ").strip().lower()

    if choice in ["1", "router"]:
        test_router()
    elif choice in ["2", "rag"]:
        test_rag()
    elif choice in ["3", "multi"]:
        test_multi()
    elif choice in ["4", "agent"]:
        test_agent()
    elif choice in ["5", "all"]:
        test_all()
    elif choice in ["6", "quick"]:
        quick_test()
    elif choice in ["0", "exit"]:
        print("退出测试")
        return
    else:
        print(f"未知选项: {choice}")
        interactive_mode()
        return

    # 测试完成后询问是否继续
    again = input("\n是否继续测试其他模块？(y/n): ").strip().lower()
    if again == "y":
        interactive_mode()


def quick_test():
    """快速测试（预设用例）"""
    print_separator("快速测试")

    from src.multi_agent import MultiAgentSystem

    system = MultiAgentSystem()

    print("\n【快速测试用例】")
    test_cases = [
        ("强制RAG", "在知识库中查找公司政策"),
        ("记忆测试", "我叫李四"),
        ("记忆回忆", "我叫什么名字"),
        ("历史关联", "它好吃吗"),
        ("简单问候", "你好"),
    ]

    for name, question in test_cases:
        print(f"\n--- {name} ---")
        print(f"用户: {question}")
        result = system.run(question)
        print(f"助手: {result['final_answer'][:100]}")
        # 打印关键过程
        for line in result['process'].split('\n'):
            if any(x in line for x in ['判断', '规划决策', '强制']):
                print(f"  {line}")


if __name__ == "__main__":
    # 命令行参数模式
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "router":
            test_router()
        elif arg == "rag":
            test_rag()
        elif arg == "multi":
            test_multi()
        elif arg == "agent":
            test_agent()
        elif arg == "all":
            test_all()
        else:
            print(f"未知参数: {arg}")
            print("用法: python test.py [router|rag|multi|agent|all]")
    else:
        # 交互式模式
        interactive_mode()