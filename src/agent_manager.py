from langchain.agents import create_agent
from langchain.tools import tool
from datetime import datetime


@tool
def get_weather(city: str) -> str:
    """获取天气"""
    db = {"北京": "晴15-25°C", "上海": "多云18-26°C", "深圳": "晴22-30°C"}
    return db.get(city, f"{city}：晴，20°C")


@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expr: str) -> str:
    """计算数学表达式"""
    try:
        return f"{expr} = {eval(expr)}"
    except:
        return f"计算错误"


class AgentManager:
    def __init__(self, llm):
        self.llm = llm
        self.agent = create_agent(
            llm,
            tools=[get_weather, get_current_time, calculate],
            system_prompt="你是智能助手，可使用工具。"
        )

    def ask(self, question):
        resp = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
        return resp["messages"][-1].content