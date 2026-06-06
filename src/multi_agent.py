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
        print("[multi-init] 正在初始化多智能体系统...")
        llm_config = get_model_config()
        self.llm = ChatOpenAI(**llm_config)

        # 初始化基础组件
        self.rag = RAGEngine(self.llm)
        self.agent = AgentManager(self.llm)
        self.router = Router(llm=self.llm)

        # 记忆功能
        self.conversation_history = []  # 存储对话历史
        self.max_history = 20  # 最多保留20轮对话

        # 强制 RAG 关键词（与 router 保持一致）
        self.force_rag_keywords = ["知识库", "文件", "rag", "RAG", "在知识库中", "在文件中", "查找", "文档", "资料"]

        print("[multi-init] 多智能体系统已就绪\n")

    def _add_to_history(self, user_msg: str, assistant_msg: str):
        """添加对话到历史"""
        self.conversation_history.append({
            "user": user_msg,
            "assistant": assistant_msg
        })
        # 保留最近的消息
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        print(f"[multi-memory] 已保存对话，当前共 {len(self.conversation_history)} 条记录")

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

    def clear_history(self):
        """清空对话历史（公共接口）"""
        self.conversation_history = []
        print("[multi-memory] 对话历史已清空")

    def _clear_history(self):
        """清空对话历史（内部方法）"""
        self.clear_history()

    def get_history_count(self) -> int:
        """获取历史对话轮数"""
        return len(self.conversation_history)

    def get_history_summary(self) -> str:
        """获取历史摘要（用于调试）"""
        if not self.conversation_history:
            return "暂无对话历史"

        summary = "最近对话：\n"
        for msg in self.conversation_history[-5:]:
            summary += f"用户: {msg['user'][:50]}\n"
            summary += f"助手: {msg['assistant'][:50]}\n"
        return summary

    def _check_related_to_history(self, question: str) -> tuple:
        """
        判断当前问题是否与前文有关

        Returns:
            (是否相关, 相关说明)
        """
        if not self.conversation_history:
            return False, "无历史对话"

        # 关键词判断
        context_keywords = ["刚才", "之前", "上文", "刚才说的", "之前说的", "接着说", "继续", "还有呢", "然后呢"]
        for keyword in context_keywords:
            if keyword in question:
                return True, f"命中上下文关键词: {keyword}"

        # 代词判断（需要更精确的语义，这里用简单规则）
        pronoun_keywords = ["它", "他", "她", "这个", "那个", "那里", "这里", "那件事", "这件事"]
        for keyword in pronoun_keywords:
            if keyword in question:
                return True, f"命中代词关键词: {keyword}"

        # 如果问题很短（小于5个字），可能是在追问
        if len(question.strip()) < 5:
            return True, "问题较短，可能是追问"

        return False, "与前文无关"

    def _planning(self, question: str, process_log: list) -> tuple:
        """
        规划师：拆解任务

        Returns:
            (task_list, original_question)  # 返回任务列表和原始问题
        """

        # 记录思考过程
        is_related, reason = self._check_related_to_history(question)
        if is_related:
            process_log.append(f"🔗 判断：当前问题与前文有关（{reason}），将结合历史对话理解")
        else:
            process_log.append(f"📌 判断：当前问题与前文无关（{reason}），独立处理")

        # 简单问候直接返回空
        simple_responses = ["你好", "嗨", "hi", "hello", "谢谢", "感谢", "好的", "ok"]
        if question.lower() in simple_responses or question in simple_responses:
            process_log.append("💬 判断：检测到简单问候/感谢，无需任务规划")
            return [], question

        history_context = self._get_history_context()

        prompt = f"""你是任务规划师

{history_context}

## 当前用户问题
{question}

## 任务规划规则
1. 如果用户只是在聊天、问候、感谢，或者问题很简单不需要查询知识库/工具，输出：无任务
2. 如果需要查询知识库文件内容，输出：查询知识库
3. 如果需要查询天气、时间、计算、搜索网络，输出：使用工具
4. 如果需要发送邮件、添加日历、写笔记，输出：执行操作
5. 如果问题涉及之前的对话（如"我之前说了什么","我刚才说什么"），输出：从历史获取

## 输出格式（只输出以下之一，不要有其他内容）：
- 无任务
- 查询知识库
- 使用工具
- 执行操作
- 从历史获取

输出："""

        response = self.llm.invoke(prompt)
        decision = response.content.strip()

        print(f"[multi-planner] 决策: {decision}")
        process_log.append(f"📋 规划决策: {decision}")

        if decision == "无任务":
            return [], question
        elif decision == "查询知识库":
            process_log.append("📚 规划：需要查询知识库")
            return ["查询知识库"], question
        elif decision == "使用工具":
            process_log.append("🔧 规划：需要使用工具")
            return ["使用工具"], question
        elif decision == "执行操作":
            process_log.append("⚡ 规划：需要执行操作")
            return ["执行操作"], question
        elif decision == "从历史获取":
            process_log.append("📜 规划：从历史对话中获取信息")
            return ["从历史获取"], question
        else:
            # 降级：直接把原问题作为任务
            process_log.append(f"⚠️ 规划：未识别决策类型，降级为原问题处理")
            return [question], question

    def _force_rag_check(self, question: str) -> bool:
        """检查是否强制走 RAG"""
        question_lower = question.lower()
        for keyword in self.force_rag_keywords:
            if keyword.lower() in question_lower:
                print(f"[multi-force-rag] 命中强制RAG关键词: {keyword}")
                return True
        return False

    def _execute_task(self, task: str, original_question: str, process_log: list) -> str:
        """执行单个任务 - 使用路由"""

        print(f"[multi-executor] 开始处理任务: {task[:80]}...")
        process_log.append(f"⚙️ 执行任务: {task[:50]}...")

        # 特殊处理1：从历史获取信息
        if task == "从历史获取":
            process_log.append("📜 执行：从历史对话中查找信息")
            return self._answer_from_history(process_log)

        # 特殊处理2：强制 RAG（使用原始问题匹配关键词）
        if self._force_rag_check(original_question):
            process_log.append("🔍 执行：强制 RAG 模式（直接返回原文）")
            rag_result = self.rag.ask_direct(original_question)
            if rag_result:
                process_log.append(f"✅ 执行：从知识库检索到原文（{len(rag_result)} 字符）")
                return rag_result
            else:
                process_log.append("⚠️ 执行：知识库中未找到相关信息")
                return "知识库中未找到相关信息"

        # 特殊处理3：查询知识库（普通RAG）
        if task == "查询知识库":
            process_log.append("📚 执行：从知识库检索并生成回答")
            rag_result = self.rag.ask(original_question)
            if rag_result:
                process_log.append(f"✅ 执行：知识库回答完成")
                return rag_result
            else:
                process_log.append("⚠️ 执行：知识库无结果，降级到 Agent")
                return self.agent.ask(original_question)

        # 正常路由
        try:
            route_type = self.router.route(original_question)
            print(f"[multi-executor] 路由决策: {route_type}")
            process_log.append(f"🔄 执行：路由决策为 {route_type}")

            if route_type == "rag":
                process_log.append("📚 执行：从知识库检索并生成回答")
                rag_result = self.rag.ask(original_question)
                if rag_result:
                    process_log.append(f"✅ 执行：知识库回答完成")
                    return rag_result
                else:
                    process_log.append("⚠️ 执行：知识库无结果，降级到 Agent")
                    return self.agent.ask(original_question)
            else:
                process_log.append("🔧 执行：调用 Agent 工具")
                return self.agent.ask(original_question)

        except Exception as e:
            print(f"[multi-executor] 路由错误: {e}，降级到 Agent")
            process_log.append(f"❌ 执行：路由错误，降级到 Agent")
            return self.agent.ask(original_question)

    def _answer_from_history(self, process_log: list) -> str:
        """从历史中回答问题"""
        if not self.conversation_history:
            process_log.append("📜 历史为空，无信息可返回")
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
        result = response.content.strip()
        process_log.append(f"📜 从历史中提取答案: {result[:50]}...")
        return result

    def _synthesize(self, question: str, task_results: list, process_log: list) -> str:
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
        result = response.content.strip()
        process_log.append(f"✍️ 撰写完成，回答长度: {len(result)} 字符")
        return result

    def run(self, question: str) -> dict:
        """
        统一的多智能体协作入口（带记忆和思考过程）

        Returns:
            dict: {"final_answer": str, "process": str}
        """
        # 初始化过程日志
        process_log = []

        print(f"\n[multi-start] 开始处理用户问题")
        print(f"[multi-memory] 当前对话轮数: {len(self.conversation_history)}")
        print(f"[multi-user] {question}")

        process_log.append(f"📥 用户输入: {question}")
        process_log.append(f"📊 当前历史对话轮数: {len(self.conversation_history)}")

        # ========== 最高优先级：强制RAG ==========
        if self._force_rag_check(question):
            process_log.append("🔍 检测到强制RAG关键词，直接返回知识库原文")
            rag_result = self.rag.ask_direct(question)
            if rag_result:
                final_answer = rag_result
                process_log.append(f"✅ 返回原文（{len(final_answer)} 字符）")
                process_log.append(f"📄 原文内容: {final_answer[:200]}...")
                self._add_to_history(question, final_answer)
                return {"final_answer": final_answer, "process": "\n".join(process_log)}
            else:
                final_answer = "知识库中未找到相关信息"
                process_log.append("⚠️ 知识库中未找到相关信息")
                self._add_to_history(question, final_answer)
                return {"final_answer": final_answer, "process": "\n".join(process_log)}

        # ========== 简单问候直接返回 ==========
        if question in ["谢谢", "感谢", "好的", "ok"]:
            responses = {"谢谢": "不客气！", "感谢": "不客气！", "好的": "好的", "ok": "OK"}
            final_answer = responses.get(question, "好的")
            process_log.append(f"💬 检测到简单回应，直接返回: {final_answer}")
            self._add_to_history(question, final_answer)
            process_log.append(f"💾 已保存到对话历史")
            return {"final_answer": final_answer, "process": "\n".join(process_log)}

        # ========== 处理"我叫什么名字"类型的问题 ==========
        if "我叫什么" in question or "我的名字" in question or "我是谁" in question:
            process_log.append("🔍 检测到名字查询请求")
            for msg in reversed(self.conversation_history):
                if "我叫" in msg['user']:
                    match = re.search(r'我叫([^，,.。！？]+)', msg['user'])
                    if match:
                        name = match.group(1).strip()
                        final_answer = f"你叫{name}"
                        process_log.append(f"📜 从历史中找到名字: {name}")
                        self._add_to_history(question, final_answer)
                        return {"final_answer": final_answer, "process": "\n".join(process_log)}
                    break
            # 没找到
            final_answer = "抱歉，我不记得你叫什么名字。请告诉我你的名字。"
            process_log.append("❌ 历史中未找到名字信息")
            self._add_to_history(question, final_answer)
            return {"final_answer": final_answer, "process": "\n".join(process_log)}

        # ========== 处理"我之前说了什么"类型的问题 ==========
        if "之前说了什么" in question or "刚才说了什么" in question:
            process_log.append("🔍 检测到历史内容查询请求")
            if self.conversation_history:
                last = self.conversation_history[-1]
                final_answer = f"你刚才说：{last['user']}"
                process_log.append(f"📜 从历史中找到上一轮对话: {last['user'][:50]}...")
            else:
                final_answer = "还没有历史对话记录。"
                process_log.append("❌ 历史为空")
            self._add_to_history(question, final_answer)
            return {"final_answer": final_answer, "process": "\n".join(process_log)}

        # ========== 正常处理 ==========
        tasks, original_q = self._planning(question, process_log)

        if not tasks:
            # 直接用 LLM 回答（带历史）
            process_log.append("💬 无任务规划，直接使用 LLM 回答")
            history = self._get_history_context()
            prompt = f"{history}\n用户问题：{question}\n请回答："
            final_answer = self.llm.invoke(prompt).content
            process_log.append(f"✍️ LLM 直接回答完成")
        else:
            # 执行任务
            results = []
            for task in tasks:
                result = self._execute_task(task, original_q, process_log)
                results.append(result)
                process_log.append(f"📦 任务结果: {result[:100]}...")
            final_answer = self._synthesize(question, results, process_log)

        # 保存到历史
        self._add_to_history(question, final_answer)
        process_log.append(f"💾 对话已保存到历史，当前共 {len(self.conversation_history)} 条")

        print(f"[multi-final] 最终回答: {final_answer[:100]}...")

        return {"final_answer": final_answer, "process": "\n".join(process_log)}

    def reload_knowledge(self):
        """重新加载知识库"""
        print("[multi-reload] 重新加载知识库...")
        self.rag.reload()
        if hasattr(self.router, 'reload'):
            self.router.reload()
        print("[multi-reload] 知识库已更新")