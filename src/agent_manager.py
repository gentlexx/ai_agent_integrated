# src/agent_manager.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.tools import (
    get_weather,
    get_current_time,
    calculate,
    search_web,
    send_email,
    add_calendar_event,
    list_events,
    read_note,
    write_note,
)


class AgentManager:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [
            get_weather,
            get_current_time,
            calculate,
            search_web,
            send_email,
            add_calendar_event,
            list_events,
            read_note,
            write_note,
        ]
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt="你是智能助手，可以使用工具帮助用户。"
        )

    def ask(self, question: str) -> str:
        """执行 Agent 问答"""
        try:
            print(f"[Agent] 收到问题: {question}")
            result = self.agent.invoke(
                {"messages": [("user", question)]}
            )
            # 打印所有消息，查看工具调用过程
            print(f"[Agent] 消息数量: {len(result['messages'])}")
            for i, msg in enumerate(result['messages']):
                print(f"[Agent] 消息{i}: {msg.__class__.__name__} : {str(msg.content)[:100]}")
            # 提取最后一条消息的内容
            return result["messages"][-1].content
        except Exception as e:
            print(f"[Agent] 错误: {e}")
            # 降级处理：直接用 LLM 回答
            try:
                return self.llm.invoke(question).content
            except:
                return f"抱歉，处理您的问题时出错：{str(e)}"