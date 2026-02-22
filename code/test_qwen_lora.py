# -*- coding: utf-8 -*-
"""
test_qwen_lora.py - 测试训练好的 Qwen-LoRA 模型
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============ 设置路径 ============
# 基础模型路径（你移动后的位置，如果没有移动就用缓存）
base_model_path = r"C:\Users\LJA\Desktop\LLM-Project\models\base_models\Qwen_Qwen1.5-1.8B"
lora_path = r"C:\Users\LJA\Desktop\LLM-Project\models\qwen_lora_v1"

# 如果基础模型没移动，就从缓存加载
if not os.path.exists(base_model_path):
    base_model_path = "Qwen/Qwen1.5-1.8B"
    print("⚠️ 使用缓存中的基础模型")

print("="*60)
print("🧪 测试 Qwen-1.8B LoRA 微调模型")
print(f"📂 LoRA 路径: {lora_path}")
print("="*60)

# ============ 加载基础模型 ============
print("\n🔄 加载基础模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    print("✅ 基础模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 设置 padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============ 加载 LoRA 权重 ============
print("\n🔄 加载 LoRA 权重...")
try:
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    print("✅ LoRA 加载成功！")
except Exception as e:
    print(f"❌ LoRA 加载失败: {e}")
    exit(1)

# ============ 问答函数 ============
def ask(question, max_length=200):
    # Qwen 对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回答
    if "<|im_start|>assistant" in response:
        answer = response.split("<|im_start|>assistant")[-1].strip()
        return answer
    return response

# ============ 测试问题 ============
test_questions = [
    "什么是时间复杂度？",
    "栈和队列的区别",
    "什么是死锁？",
    "TCP和UDP的区别",
    "什么是虚拟内存？",
    "进程和线程的区别",
    "什么是二叉树？",
    "HTTP和HTTPS的区别"
]

print("\n📝 测试结果:")
print("="*60)

for q in test_questions:
    print(f"📌 问题: {q}")
    answer = ask(q)
    print(f"💬 回答: {answer}")
    print("-"*40)

# ============ 交互式对话 ============
print("\n💬 交互式对话模式（输入 exit 退出）")
print("="*60)

while True:
    user_input = input("\n你: ")
    if user_input.lower() in ['exit', 'quit', 'q']:
        break
    
    answer = ask(user_input)
    print(f"模型: {answer}")

print("\n✅ 测试完成！")