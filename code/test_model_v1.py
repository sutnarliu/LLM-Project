import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============ 1. 设置镜像源（防止任何联网请求）============
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 备用
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 🔥 强制离线模式！
os.environ['HF_HUB_OFFLINE'] = '1'        # 🔥 强制离线模式！

# ============ 2. 加载模型 ============
model_path = r"C:\Users\LJA\Desktop\LLM-Project\models\lora_distilgpt2"
base_model_name = "distilgpt2"

print(f"📂 加载模型: {model_path}")
print("=" * 60)

# 🔥 关键修复：先检查本地是否有缓存
try:
    # 尝试从本地缓存加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)
    print("✅ 从本地缓存加载成功！")
except:
    print("⚠️ 本地没有缓存，需要下载一次...")
    # 设置镜像源下载
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    print("✅ 下载完成，下次就可以离线使用了！")

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, model_path)

# 移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✅ 模型加载成功！运行设备: {device}")
print("=" * 60)

# ============ 3. 定义问答函数 ============
def ask_model(question):
    prompt = f"问题：{question}\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "答案：" in response:
        answer = response.split("答案：")[-1]
    else:
        answer = response.replace(prompt, "")
    
    return answer.strip()

# ============ 4. 测试效果 ============
test_questions = [
    "什么是时间复杂度？",
    "栈和队列的区别",
    "什么是死锁？",
]

print("\n🤖 模型测试结果:")
print("=" * 60)

for q in test_questions:
    answer = ask_model(q)
    print(f"📌 问题: {q}")
    print(f"💬 回答: {answer}")
    print("-" * 40)