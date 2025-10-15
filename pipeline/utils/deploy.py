# export HF_ENDPOINT=https://hf-mirror.com
import os

# 设置环境变量
# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Use environment variable for HF_ENDPOINT or set default
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 验证是否设置成功
print(os.getenv("HF_ENDPOINT"))

from huggingface_hub import snapshot_download

# snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf',
#                   repo_type='model',
#                   local_dir='/home/lxz/code/LLaVA/LLama2-7b-chat-hf',
#                   token=os.getenv("HF_TOKEN"),
#                   resume_download=True)
# snapshot_download(repo_id='yahma/llama-7b-hf',
#                   repo_type='model',
#                   local_dir='/home/lxz/code/LLaVA/LLama-7b-hf',
#                   token=os.getenv("HF_TOKEN"),
#                   resume_download=True)

"""
snapshot_download(repo_id='Vision-CAIR/vicuna-7b',
                  repo_type='model',
                  local_dir='/home/data/lxz/MODEL/VICUNA/vicuna-7b',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)
"""
snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct',
                  repo_type='model',
                  local_dir='../../models/Qwen2.5-7B-Instruct',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)
              
snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct',
                  repo_type='model',
                  local_dir='../../models/Qwen2.5-3B-Instruct',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)

snapshot_download(repo_id='Qwen/Qwen2.5-14B-Instruct',
                  repo_type='model',
                  local_dir='../../models/Qwen2.5-14B-Instruct',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)
                  
snapshot_download(repo_id='Qwen/Qwen3-8B',
                  repo_type='model',
                  local_dir='../../models/Qwen3-8B',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)
                  
snapshot_download(repo_id='theo77186/Llama-3.2-8B-Instruct',
                  repo_type='model',
                  local_dir='../../models/Llama-3.2-8B-Instruct',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)
                  
snapshot_download(repo_id='internlm/internlm3-8b-instruct',
                  repo_type='model',
                  local_dir='../../models/internlm3-8b-instruct',
                  token=os.getenv("HF_TOKEN"),
                  resume_download=True)

print("start!")
from lmdeploy import pipeline, TurbomindEngineConfig
pipe = pipeline("../../models/Llama-3.2-8B-Instruct")
response = pipe(["你是谁", "你的模型参数量是多少"])
print(response)
