import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============ 离线模式 ============
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ============ 路径设置 ============
base_path = r"C:\Users\LJA\Desktop\LLM-Project\models\base_models\distilgpt2"
lora_path = r"C:\Users\LJA\Desktop\LLM-Project\models\v2_60data"

print("="*60)
print("🧪 测试 v2 模型 (60条数据)")
print("="*60)

# ============ 加载模型 ============
print("🔄 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

# ============ 对比v1模型（可选）============
try:
    v1_path = r"C:\Users\LJA\Desktop\LLM-Project\models\lora_models\v1_5data"
    if os.path.exists(v1_path):
        print("\n🔄 加载v1模型进行对比...")
        model_v1 = PeftModel.from_pretrained(base_model, v1_path).to(device)
        model_v1.eval()
        
        print("\n📊 v1 (5条) vs v2 (60条) 对比:")
        print("="*60)
        for q in test_questions[:2]:  # 只对比前两个
            answer_v1 = ask_with_model(model_v1, q)
            answer_v2 = ask_with_model(model, q)
            print(f"问题: {q}")
            print(f"v1: {answer_v1[:50]}...")
            print(f"v2: {answer_v2[:50]}...")
            print("-"*40)
except:
    pass

print("\n✅ 测试完成！")