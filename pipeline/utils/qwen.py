import os
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI, APIConnectionError, RateLimitError, APIError

client = OpenAI(api_key=os.getenv("BY_API_KEY", "BY_API_KEY"),
                 base_url=os.getenv("BY_BASE_URL", "http://127.0.0.1:23333/v1"))
model_name = client.models.list().data[0].id      #本地端口每次只开放一个模型

print(model_name)