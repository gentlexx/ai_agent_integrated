# app_st.py
import streamlit as st
from src.multi_agent import MultiAgentSystem
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 初始化多智能体系统
@st.cache_resource
def load_system():
    return MultiAgentSystem()


mas = load_system()

st.set_page_config(page_title="多智能体协作系统", page_icon="🤖")
st.title("🤖 多智能体协作系统")
st.markdown("**规划师 → 研究员 → 撰写员**")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用多智能体
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = mas.run(prompt)
            final_answer = result["final_answer"]
            process = result["process"]

            # 显示可折叠的思考过程
            with st.expander("🤔 查看思考过程"):
                st.text(process)

            st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})