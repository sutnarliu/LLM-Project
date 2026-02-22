import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============ 离线模式 ============
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ============ 你的实际路径（先用os.path处理）============
base_path = os.path.join("C:", os.sep, "Users", "LJA", "Desktop", "LLM-Project", "models", "base_models", "distilgpt2")
lora_path = r"C:\Users\LJA\Desktop\LLM-Project\models\v2_60data"

print("="*60)
print("🧪 测试 v2 模型 (60条数据)")
print(f"📂 基础模型路径: {base_path}")
print(f"📂 LoRA模型路径: {lora_path}")
print("="*60)

# ============ 检查路径是否存在 ============
if not os.path.exists(base_path):
    print(f"⚠️ 基础模型路径不存在: {base_path}")
    print("尝试从缓存加载 distilgpt2...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=True)
        print("✅ 从缓存加载成功！")
    except:
        print("❌ 缓存中也找不到，需要联网下载一次")
        # 临时允许联网下载
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        # 保存到你的base_models目录
        os.makedirs(base_path, exist_ok=True)
        tokenizer.save_pretrained(base_path)
        base_model.save_pretrained(base_path)
        print(f"✅ 已下载并保存到: {base_path}")
else:
    print("🔄 从本地路径加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============ 加载LoRA权重 ============
print("🔄 加载LoRA权重...")
if not os.path.exists(lora_path):
    print(f"❌ LoRA模型路径不存在: {lora_path}")
    exit(1)

model = PeftModel.from_pretrained(base_model, lora_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✅ 模型加载成功！设备: {device}")

# ============ 问答函数 ============
def ask(question):
    prompt = f"问题：{question}\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,           # 增加生成长度
            temperature=0.3,               # 降低温度，更确定性
            do_sample=False,                # 关闭采样，每次都选最可能的词
            repetition_penalty=1.2,         # 增加重复惩罚
            num_beams=3,                    # 使用beam search
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"完整输出: {response}")  # 调试用
    if "答案：" in response:
        return response.split("答案：")[-1].strip()
    return response.strip()

# ============ 测试问题 ============
test_questions = [
    "什么是时间复杂度？",
    "栈和队列的区别",
    "什么是死锁？",
    "TCP和UDP的区别",
    "什么是虚拟内存？",
    "进程和线程的区别"
]

print("\n📝 测试结果:")
print("="*60)

for q in test_questions:
    answer = ask(q)
    print(f"📌 问题: {q}")
    print(f"💬 回答: {answer}")
    print("-"*40)

print("\n✅ 测试完成！")