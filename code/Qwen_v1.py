# -*- coding: utf-8 -*-
"""
train_qwen.py - 用Qwen-1.8B训练中文模型（从缓存加载）
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============ 设置设备 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("🚀 开始训练 Qwen-1.8B 中文模型")
print("="*60)
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*60)

# ============ 加载数据 ============
data_path = r"C:\Users\LJA\Desktop\LLM-Project\data\train_data.json"
print(f"📂 加载数据: {data_path}")

if not os.path.exists(data_path):
    print(f"❌ 数据文件不存在: {data_path}")
    exit(1)

with open(data_path, 'r', encoding='utf-8') as f:
    train_examples = json.load(f)

print(f"✅ 加载了 {len(train_examples)} 条训练数据")
print("\n数据示例:")
for i in range(min(3, len(train_examples))):
    print(f"  {i+1}. 问题: {train_examples[i]['instruction']}")
    print(f"     答案: {train_examples[i]['output'][:50]}...")

# ============ 格式化数据 ============
print("\n📝 格式化数据...")
formatted_texts = []
for ex in train_examples:
    # Qwen的对话格式
    text = f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"
    formatted_texts.append(text)

# ============ 🔥 修改的地方：加载Qwen模型（从缓存）============
print("\n🔄 从缓存加载已下载的 Qwen-1.8B 模型...")

model_name = "Qwen/Qwen1.5-1.8B"

try:
    # 直接从缓存加载（local_files_only=True 强制只从本地加载）
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        local_files_only=True  # 🔥 关键修改！
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="float16",  # 用字符串形式避免警告
        device_map="auto",
        local_files_only=True  # 🔥 关键修改！
    )
    print("✅ 从缓存加载成功！")
except Exception as e:
    print(f"❌ 从缓存加载失败: {e}")
    print("请先确认模型已下载完成")
    exit(1)

# 设置padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============ Tokenize ============
print("\n🔄 正在tokenize数据...")

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    outputs["labels"] = outputs["input_ids"].clone()
    outputs["labels"][outputs["labels"] == tokenizer.pad_token_id] = -100
    return outputs

# 创建dataset
dataset = Dataset.from_dict({"text": formatted_texts})
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=["text"]
)

print(f"✅ Tokenize完成，数据集大小: {len(tokenized_dataset)}")

# ============ 配置LoRA ============
print("\n⚙️ 配置LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 配置训练参数 ============
print("\n🏋️ 配置训练参数...")

# 保存路径
save_path = r"C:\Users\LJA\Desktop\LLM-Project\models\qwen_lora_v1"
os.makedirs(save_path, exist_ok=True)

training_args = TrainingArguments(
    output_dir=save_path,
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

# 创建data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ============ 开始训练 ============
print("\n🚀 开始训练 Qwen-1.8B...")
print("训练过程可能需要20-30分钟，请耐心等待...")
print("-" * 40)

trainer.train()

print("\n✅ 训练完成！")
print("-" * 40)

# ============ 保存模型 ============
print("\n💾 保存模型...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 模型已保存到: {save_path}")

# ============ 验证保存 ============
print("\n🔍 验证保存结果...")
files = os.listdir(save_path)
print(f"📁 文件夹内容 ({len(files)} 个文件):")
for file in sorted(files):
    file_path = os.path.join(save_path, file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / 1024
        print(f"   - {file} ({size:.1f} KB)")

print("\n" + "="*60)
print("🎉 Qwen-1.8B 模型训练完成！")
print(f"📂 模型位置: {save_path}")
print("="*60)