# -*- coding: utf-8 -*-
"""
web_demo_simple.py - 兼容旧版本Gradio
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# ============ 设置路径 ============
base_model_path = r"C:\Users\LJA\Desktop\LLM-Project\models\base_models\Qwen_Qwen1.5-1.8B"
lora_path = r"C:\Users\LJA\Desktop\LLM-Project\models\qwen_lora_v1"

print("🚀 启动Web界面...")
print("正在加载模型，请稍候...")

# ============ 加载模型 ============
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
print("✅ 模型加载成功！")

# ============ 问答函数 ============
def ask(message, history):
    """处理对话"""
    if not message.strip():
        return ""
    
    # Qwen对话格式
    prompt = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        answer = response.split("<|im_start|>assistant")[-1].strip()
        return answer
    return response

# ============ 创建最简单的聊天界面 ============
# 去掉theme参数
demo = gr.ChatInterface(
    fn=ask,
    title="🎓 计算机考研助手",
    description="基于Qwen-1.8B + LoRA微调",
    examples=[
        "什么是时间复杂度？",
        "栈和队列的区别",
        "什么是死锁？",
        "TCP和UDP的区别",
    ],
    # theme="soft"  ← 删掉这行
)

# ============ 启动 ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 启动Web服务器...")
    print("📱 访问地址: http://127.0.0.1:7860")
    print("🛑 按 Ctrl+C 停止")
    print("="*60)
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )