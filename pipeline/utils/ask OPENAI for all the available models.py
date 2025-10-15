
from dotenv import load_dotenv
load_dotenv(override=True)


from openai import OpenAI
import os

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    if not api_key:
        raise EnvironmentError("未检测到 OPENAI_API_KEY（或 openai_api_key），请在 .env 配置。")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

client = get_client()
model_name = client.models.list()

model_name=[i.id for i in model_name]


#print(model_name)
for i in model_name:
    print(i)
