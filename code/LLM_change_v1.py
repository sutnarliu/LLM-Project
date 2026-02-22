import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 🔥 关键！解决网络问题

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============ 1. 在桌面创建项目文件夹 ============
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
project_path = os.path.join(desktop, 'LLM-Project')
model_save_path = os.path.join(project_path, 'models', 'lora_distilgpt2')
os.makedirs(model_save_path, exist_ok=True)

print(f"📁 模型保存路径: {model_save_path}")
print("=" * 60)

# ============ 2. 加载模型（现在会从镜像下载）============
print("🔄 加载模型（从国内镜像）...")
model_name = "distilgpt2"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("💡 尝试使用本地缓存或更换模型...")
    raise

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✅ 运行设备: {device}")
print("=" * 60)

# ============ 3. 准备训练数据 ============
print("📊 准备训练数据...")

train_examples = [
    {"instruction": "什么是时间复杂度？", "output": "时间复杂度是算法执行时间随输入规模增长的量度。"},
    {"instruction": "解释一下栈和队列的区别", "output": "栈是后进先出（LIFO），队列是先进先出（FIFO）。"},
    {"instruction": "什么是死锁？", "output": "死锁是两个或多个进程互相等待资源，导致都无法继续执行的状态。"},
    {"instruction": "TCP和UDP有什么区别？", "output": "TCP面向连接、可靠；UDP无连接、速度快。"},
    {"instruction": "什么是虚拟内存？", "output": "虚拟内存是把磁盘空间当内存用，让程序拥有大于物理内存的地址空间。"},
]

formatted_texts = []
for ex in train_examples:
    text = f"问题：{ex['instruction']}\n答案：{ex['output']}"
    formatted_texts.append(text)

def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

dataset = Dataset.from_dict({"text": formatted_texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(f"✅ 数据集创建完成，共 {len(tokenized_dataset)} 条样本")
print("=" * 60)

# ============ 4. 配置LoRA ============
print("⚙️ 配置LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("=" * 60)

# ============ 5. 训练配置 ============
print("🏋️ 配置训练参数...")

training_args = TrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-4,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    #tokenizer=tokenizer,
)

# ============ 6. 开始训练 ============
print("🚀 开始训练...")
trainer.train()
print("✅ 训练完成！")
print("=" * 60)

# ============ 7. 强制保存模型 ============
print("💾 保存模型...")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✅ 模型已保存到: {model_save_path}")
print("=" * 60)

# ============ 8. 验证保存结果 ============
print("🔍 验证保存结果...")
files = os.listdir(model_save_path)
print(f"📁 文件夹内容 ({len(files)} 个文件):")

required_files = ['adapter_model.bin', 'adapter_config.json']
for file in sorted(files):
    file_path = os.path.join(model_save_path, file)
    size = os.path.getsize(file_path) / 1024
    print(f"   - {file} ({size:.1f} KB)")

if all(f in files for f in required_files):
    print("\n✨ 模型保存成功！完整可用！")
    print(f"📂 路径: {model_save_path}")
else:
    print("\n❌ 保存不完整")

print("\n" + "=" * 60)
print("🎉 完成！以后加载模型就用这个路径:")
print(f'model_path = r"{model_save_path}"')