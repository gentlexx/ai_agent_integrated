import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.multi_agent import MultiAgentSystem
from src.rag_engine import RAGEngine
import json

# 全局初始化
print("🚀 正在加载多智能体系统...")
mas = MultiAgentSystem()
print("✅ 系统已就绪\n")


def respond(message, chat_history):
    """响应函数"""
    if not message:
        return "", chat_history

    # 使用 run 方法（已合并 run_with_process）
    result = mas.run(message)
    final_answer = result["final_answer"]
    process = result["process"]

    # 格式化为可折叠的思考过程
    formatted = f"""<details>
<summary>查看思考过程</summary>
<div style="background:#f5f5f5; padding:10px; border-radius:5px; font-size:12px; margin-top:5px;">
{process.replace(chr(10), '<br>')}
</div>
</details>

---

{final_answer}"""

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": formatted})

    return "", chat_history


def upload_file(file):
    """上传知识库文件并自动更新 RAG"""
    if file is None:
        return "请选择文件", get_file_list()

    if isinstance(file, dict):
        file_path = file.get("path") or file.get("name")
        file_name = file.get("name", "unknown")
    elif isinstance(file, str):
        file_path = file
        file_name = os.path.basename(file)
    else:
        return "不支持的文件格式", get_file_list()

    if not file_path or not os.path.exists(file_path):
        return "文件不存在或路径无效", get_file_list()

    import shutil
    os.makedirs("./knowledge_base", exist_ok=True)

    dest_path = os.path.join("./knowledge_base", os.path.basename(file.name))
    shutil.copy(file.name, dest_path)

    mas.reload_knowledge()

    return f"✅ 文件已上传并索引：{os.path.basename(file.name)}", get_file_list()


def get_file_list():
    """获取知识库文件列表"""
    kb_dir = "./knowledge_base"
    if not os.path.exists(kb_dir):
        return []

    files = []
    for f in os.listdir(kb_dir):
        if f.endswith(".txt"):
            file_path = os.path.join(kb_dir, f)
            size = os.path.getsize(file_path)
            modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y/%m/%d %H:%M:%S")
            files.append({
                "name": f,
                "size": f"{size} B" if size < 1024 else f"{size/1024:.1f} KB",
                "modified": modified,
            })
    return files


def delete_file(filename):
    """删除知识库文件"""
    file_path = os.path.join("./knowledge_base", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        mas.reload_knowledge()
        return f"已删除：{filename}", get_file_list()
    return f"文件不存在：{filename}", get_file_list()


def view_file(filename):
    """查看文件内容"""
    file_path = os.path.join("./knowledge_base", filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read(2000)
            return f"{filename}\n\n{content}\n\n{'...' if len(content) > 2000 else ''}"
    return "文件不存在"


def rebuild_rag():
    """重建 RAG 引擎"""
    try:
        from langchain_openai import ChatOpenAI
        from src.config import get_model_config
        llm = ChatOpenAI(**get_model_config())
        new_rag = RAGEngine(llm)
        mas.rag = new_rag
        return "✅ 知识库已更新并重建"
    except Exception as e:
        return f"❌ 更新失败：{str(e)}"


def clear_all():
    """清空对话记忆"""
    mas.clear_history()
    return [], get_memory_status()


def get_memory_status():
    """获取记忆状态"""
    count = mas.get_history_count()
    return f"📝 已记住 {count} 轮对话"


def get_file_list_for_dropdown():
    """获取文件列表用于下拉菜单"""
    files = get_file_list()
    return [f["name"] for f in files]


# ========== 主题切换函数 ==========
def switch_theme():
    """切换主题"""
    if demo.theme == gr.themes.Soft():
        return gr.themes.Default()
    else:
        return gr.themes.Soft()


# ========== 更新记忆状态 ==========
def update_memory():
    return get_memory_status()


# 创建 Gradio 界面
with gr.Blocks(title="多智能体协作系统") as demo:
    # 顶部状态栏
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("# 🤖 有对话记忆的多智能体协作系统")
        with gr.Column(scale=1):
            memory_status = gr.Textbox(
                label="",
                value=get_memory_status(),
                interactive=False,
                show_label=False
            )

    gr.Markdown("""
    **规划师 → 执行器 → 撰写员**
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 对话", height=500)
            msg = gr.Textbox(
                label="你的问题",
                placeholder="例如：公司年假政策是什么？",
                lines=3,
            )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清空对话")
                clear_memory_btn = gr.Button("🧹 清空记忆", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 📁 知识库管理")
            file_input = gr.File(label="上传 txt 文件", file_types=[".txt"])
            upload_status = gr.Textbox(label="上传状态", interactive=False)

            gr.Markdown("### 📋 测试记忆功能")
            gr.Markdown("""
            1. 输入：**我叫张三**
            2. 输入：**我叫什么名字？**（应该回答张三）
            3. 输入：**我刚刚说了什么？**（应该回忆）
            4. 点击「清空记忆」重新开始
            """)

            gr.Markdown("### 🔧 其他功能")
            gr.Markdown("""
            - 查天气：北京天气怎么样？
            - 查政策：年假政策是什么？
            - 搜索：搜索 Python 教程
            - 笔记：帮我记一下明天开会
            """)

    # 绑定事件
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    clear.click(lambda: ([], get_memory_status()), None, [chatbot, memory_status])
    clear_memory_btn.click(clear_all, None, [chatbot, memory_status])
    file_input.change(upload_file, [file_input], [upload_status])


# 启动应用
if __name__ == "__main__":
    print("🔄 正在启动 Gradio 界面...")
    demo.launch(share=False, server_port=7860)