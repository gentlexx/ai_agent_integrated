import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    return os.getenv("ZHIPU_API_KEY")

def get_model_config():
    return {
        "model": "glm-4-flash",
        "api_key": get_api_key(),
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "temperature": 0,
    }