FROM python:3.11-slim

WORKDIR /app

#设置环境变量
ENV PYTHONDONTWORITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirrior.com

#安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

#复制依赖文件
COPY requirements.txt .

#安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && pip istall --no-cache-dir gradio langgraph ddgs

#复制项目文件
COPY scr/ ./scr/
COPY app.py .
COPY .env .

#暴露端口
EXPOSE 7860

#启动命令
CMD ["python", "app.py"]