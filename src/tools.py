from langchain.tools import tool
from datetime import datetime
import requests
import os
import json
import re

from sqlalchemy.ext.asyncio import result


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    try:
        url = f"https://wttr.in/{city}?format=%C+%t+%w"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"{city}：{response.text.strip()}"
    except:
        pass
    # 降级模拟数据
    db = {"北京": "晴 15-25°C", "上海": "多云 18-26°C", "深圳": "晴 22-30°C"}
    return db.get(city, f"模拟{city}天气：晴，20°C左右")

@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"{expression} = {eval(expression)}"
    except:
        return f"计算错误"

@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    try:
        # 使用 duckduckgo-search 库
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if results:
            output =["搜索结果："]
            for r in results:
                title = r.get('title', '无标题')
                body = r.get('body', '')[:150]
                href = r.get('href', '')
                output.append(f"\n📌 {title}")
                output.append(f"   {body}")
                output.append(f"   🔗 {href[:80]}")
            return "\n".join(output)
        else:
            return f"关于 '{query}' 未找到搜索结果"
    except ImportError:
        try:
            import requests
            from urllib.parse import quote
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)

            results = re.findall(r'<a[^>]*href="//([^"]+)"[^>]*>([^<]+)</a>', response.text)

            if results:
                top = []
                for link, title in results[:3]:
                    if 'duckduckgo' not in link and len(title) > 5:
                        top.append(f"• {title[:50]}: {link[:80]}")
                if top:
                    return "搜索结果：\n" + "\n".join(top)
        except:
            pass
    except Exception as e:
        print(f"搜索错误: {e}")

    return f"关于 '{query}' 的搜索结果：暂无详细信息"

@tool
def send_email(recipient: str, subject: str, content: str) -> str:
    """发送邮件"""

    # 验证邮箱格式
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, recipient):
        # 尝试修复常见的邮箱格式错误
        if '.qq.com' in recipient and '@' not in recipient:
            recipient = recipient.replace('.qq.com', '@qq.com')
        elif recipient.isdigit() and len(recipient) > 5:
            recipient = f"{recipient}@qq.com"

        # 再次验证
        if not re.match(email_pattern, recipient):
            return f"❌ 邮箱格式错误：{recipient}。正确格式如：user@example.com"

    #获取配置
    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender:
        return "！！！邮件功能未配置"
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = subject
        server = smtplib.SMTP(os.getenv("SMTP_SERVER", "smtp.qq.com"), int(os.getenv("SMTP_PORT", "587")))
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return f"邮件已发送给 {recipient}"
    except Exception as e:
        return f"发送失败：{e}"

@tool
def add_calendar_event(event: str, time: str) -> str:
    """添加日历事件"""
    events_file = "calendar_events.json"
    events = []
    if os.path.exists(events_file):
        with open(events_file, 'r') as f:
            events = json.load(f)
    events.append({"event": event, "time": time, "created": datetime.now().isoformat()})
    with open(events_file, 'w') as f:
        json.dump(events[-50:], f, ensure_ascii=False, indent=2)
    return f"已添加：{event}，时间：{time}"


@tool
def list_events() -> str:
    """列出所有日历事件"""
    events_file = "calendar_events.json"
    if not os.path.exists(events_file):
        return "暂无事件"
    with open(events_file, 'r') as f:
        events = json.load(f)
    if not events:
        return "暂无事件"
    result = ["最近事件："]
    for e in events[-10:]:
        result.append(f"- {e['event']} ({e['time']})")
    return "\n".join(result)

@tool
def read_note(filename: str) -> str:
    """读取笔记文件（默认目录 ./notes）"""
    os.makedirs("./notes", exist_ok=True)
    path = os.path.join("./notes", filename)
    if not os.path.exists(path):
        return f"文件 {filename} 不存在"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read(1000)

@tool
def write_note(filename: str, content: str) -> str:
    """写入笔记"""
    os.makedirs("./notes", exist_ok=True)
    path = os.path.join("./notes", filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"已保存到 {filename}"