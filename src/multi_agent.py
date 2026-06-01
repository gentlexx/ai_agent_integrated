import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from src.config import get_model_config
from src.rag_engine import RAGEngine
from src.agent_manager import AgentManager
from src.router import Router


class MultiAgentSystem:
    def __init__(self):
        print("正在初始化多智能体系统...")
        llm_config = get_model_config()
        self.llm = ChatOpenAI(**llm_config)

        # 初始化基础组件
        self.rag = RAGEngine(self.llm)
        self.agent = AgentManager(self.llm)
        self.router = Router(llm=self.llm)

        # 记忆功能
        self.conversation_history = []  # 存储对话历史
        self.max_history = 20  # 最多保留20轮对话

        print("✅ 多智能体系统已就绪\n")

    def _add_to_history(self, user_msg: str, assistant_msg: str):
        """添加对话到历史"""
        self.conversation_history.append({
            "user": user_msg,
            "assistant": assistant_msg
        })
        # 保留最近的消息
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        print(f"[multi-记忆] 已保存对话，当前共 {len(self.conversation_history)} 条记录")

    def _get_history_context(self) -> str:
        """获取历史上下文（用于提示词）"""
        if not self.conversation_history:
            return ""

        context = "## 对话历史（用于理解上下文）\n"
        for i, msg in enumerate(self.conversation_history[-8:], 1):  # 最近8轮
            context += f"{i}. 用户：{msg['user']}\n"
            context += f"   助手：{msg['assistant']}\n"
        context += "\n"
        return context

    def _clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("[multi-清空记忆] 对话历史已清空")

    def get_history_summary(self) -> str:
        """获取历史摘要（用于调试）"""
        if not self.conversation_history:
            return "暂无对话历史"

        summary = "最近对话：\n"
        for msg in self.conversation_history[-5:]:
            summary += f"用户: {msg['user'][:50]}\n"
            summary += f"助手: {msg['assistant'][:50]}\n"
        return summary

    def _planning(self, question: str) -> list:
        """规划师：拆解任务"""

        # 简单问候直接返回空
        simple_responses = ["你好", "嗨", "hi", "hello", "谢谢", "感谢", "好的", "ok"]
        if question.lower() in simple_responses or question in simple_responses:
            return []

        history_context = self._get_history_context()

        prompt = f"""你是任务规划师
        
{history_context}

## 当前用户问题
{question}

## 任务规划规则
1. 如果用户只是在聊天、问候、感谢，或者问题很简单不需要查询知识库/工具，输出：无任务
2. 如果需要查询rag知识库文件内容，输出：查询知识库
3. 如果需要查询天气、时间、计算、搜索网络，输出：使用工具
4. 如果需要发送邮件、添加日历、写笔记，输出：执行操作
5. 如果问题涉及之前的对话（如"我之前说了什么","我刚才说什么"），输出：从历史获取

## 输出格式（只输出以下之一，不要有其他内容）：
- 无任务
- 查询政策
- 使用工具
- 执行操作
- 从历史获取

输出："""

        response = self.llm.invoke(prompt)
        decision = response.content.strip()

        print(f"[multi-规划师] 决策: {decision}")

        if decision == "无任务":
            return []
        elif decision == "查询知识库":
            return ["查询相关内容"]
        elif decision == "使用工具":
            return [question]
        elif decision == "执行操作":
            return [question]
        elif decision == "从历史获取":
            return ["从对话历史中查找信息"]
        else:
            # 降级：直接把原问题作为任务
            return [question]

    def _execute_task(self, task: str) -> str:
        """执行单个任务 - 使用路由"""

        print(f"[multi-执行] 开始处理任务: {task[:80]}...")

        # 特殊处理1：从历史获取信息
        if "从对话历史中查找" in task or "历史" in task:
            return self._answer_from_history()

        # 特殊处理2：查询政策（强制走 RAG）
        if "知识库" in task or "文件" in task or "rag" in task :
            rag_result = self.rag.ask_direct(task)
            if rag_result:
                return rag_result
            return "知识库中未找到相关信息"

        # 正常路由
        try:
            route_type = self.router.route(task)
            print(f"[multi-执行] 路由决策: {route_type}")

            if route_type == "rag":
                # 走知识库
                rag_result = self.rag.ask_direct(task)
                if rag_result:
                    return rag_result
                else:
                    # 知识库无结果，尝试 Agent
                    print(f"[multi-执行] 知识库无结果，尝试 Agent")
                    return self.agent.ask(task)
            else:
                # 走 Agent 工具
                return self.agent.ask(task)

        except Exception as e:
            print(f"[multi-执行] 路由错误: {e}，降级到 Agent")
            return self.agent.ask(task)

    def _answer_from_history(self) -> str:
        """从历史中回答问题"""
        if not self.conversation_history:
            return "暂无历史对话记录"

        # 构建历史文本
        history_text = ""
        for msg in self.conversation_history:
            history_text += f"用户说：{msg['user']}\n助手说：{msg['assistant']}\n"

        prompt = f"""根据以下对话历史回答用户的问题。

对话历史：
{history_text}

用户刚才问的是关于之前对话内容的问题（比如"我叫什么名字"、"我之前说了什么"）。

请从历史中提取用户想要的信息，直接回答。如果历史中没有相关信息，请说"未找到相关信息"。

回答："""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _synthesize(self, question: str, task_results: list) -> str:
        """撰写员：整合结果"""
        context = "\n\n".join(task_results) if task_results else "无执行结果"
        history_context = self._get_history_context()

        prompt = f"""{history_context}

## 执行结果
{context}

## 用户问题
{question}

## 回答规则
1. 如果用户是问候或感谢，直接礼貌回应
2. 如果用户问关于自己的信息，从历史中查找
3. 回答要简洁，不要添加无关内容
4. 如果执行结果为空，根据历史回答

## 回答
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def run(self, question: str) -> str:
        """运行多智能体协作（带记忆）"""
        print(f"\n[multi-记忆轮数] 当前对话轮数: {len(self.conversation_history)}")
        print(f"[用户] {question}")

        # 简单问候直接返回
        if question in ["谢谢", "感谢", "好的", "ok"]:
            responses = {"谢谢": "不客气！", "感谢": "不客气！", "好的": "好的", "ok": "OK"}
            final_answer = responses.get(question, "好的")
            self._add_to_history(question, final_answer)
            return final_answer

        # 处理"我叫什么名字"类型的问题
        if "我叫什么" in question or "我的名字" in question or "我是谁" in question:
            for msg in reversed(self.conversation_history):
                if "我叫" in msg['user']:
                    match = re.search(r'我叫([^，,.。！？]+)', msg['user'])
                    if match:
                        name = match.group(1).strip()
                        final_answer = f"你叫{name}"
                        self._add_to_history(question, final_answer)
                        return final_answer
                    break
            # 没找到
            final_answer = "抱歉，我不记得你叫什么名字。请告诉我你的名字。"
            self._add_to_history(question, final_answer)
            return final_answer

        # 处理"我之前说了什么"类型的问题
        if "之前说了什么" in question or "刚才说了什么" in question:
            if self.conversation_history:
                last = self.conversation_history[-1]
                final_answer = f"你刚才说：{last['user']}"
            else:
                final_answer = "还没有历史对话记录。"
            self._add_to_history(question, final_answer)
            return final_answer

        # 正常处理
        tasks = self._planning(question)

        if not tasks:
            # 直接用 LLM 回答（带历史）
            history = self._get_history_context()
            prompt = f"{history}\n用户问题：{question}\n请回答："
            final_answer = self.llm.invoke(prompt).content
        else:
            # 执行任务
            results = []
            for task in tasks:
                result = self._execute_task(task)
                results.append(result)
            final_answer = self._synthesize(question, results)

        # 保存到历史
        self._add_to_history(question, final_answer)

        return final_answer


    def run_with_process(self, question: str) -> dict:
        """运行多智能体协作，返回结果和过程"""

    def reload_knowledge(self):
        """重新加载知识库"""
        print("🔄 多智能体系统：重新加载知识库...")
        self.rag.reload()
        if hasattr(self.router, 'reload'):
            self.router.reload()
        print("✅ 知识库已更新")