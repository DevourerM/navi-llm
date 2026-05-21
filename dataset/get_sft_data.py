import os
import json
import random
from datasets import load_dataset

# ================= 强制使用国内镜像源 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "navi_sft_data.jsonl")
# 如果你自己提取了 Galgame 剧本，可以按格式放到这个文件里参与混合
LOCAL_RP_FILE = os.path.join(BASE_DIR, "dataset", "local_galgame_rp.jsonl")

def format_sharegpt(human_text, assistant_text):
    """标准化转换为 ShareGPT 格式"""
    return {
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "assistant", "value": assistant_text}
        ]
    }

def get_ruozhiba_data(target_count=8000):
    """
    获取弱智吧数据：提升模型的逻辑抗压能力和幽默感，打破机械回复
    """
    print("🚀 正在拉取 xtuner/ruozhiba 数据集...")
    try:
        # ruozhiba 数据集很小，直接拉取全量
        dataset = load_dataset("xtuner/ruozhiba", split="train")
        formatted_data = []
        for item in dataset:
            # ruozhiba 的原始格式是 prompt 和 output
            if "prompt" in item and "output" in item:
                formatted_data.append(format_sharegpt(item["prompt"], item["output"]))
        
        # 随机采样所需数量
        random.shuffle(formatted_data)
        result = formatted_data[:target_count]
        print(f"✅ 成功获取弱智吧数据: {len(result)} 条")
        return result
    except Exception as e:
        print(f"❌ 获取弱智吧数据失败: {e}")
        return []

def get_sharegpt_data(target_count=8000):
    """
    获取 ShareGPT 数据：兜底模型的基础智商，让它知道怎么正经回答问题
    """
    print("🚀 正在拉取 sharegpt_zh 数据集 (仅采样基础问答)...")
    try:
        # 使用 shibing624/sharegpt_zh 作为通用对话基准
        dataset = load_dataset("shibing624/sharegpt_zh", split="train[:5%]") 
        formatted_data = []
        for item in dataset:
            # 提取单轮对话 (为了 SFT 简单起见，我们暂时只取多轮对话的第一回合)
            if "conversations" in item and len(item["conversations"]) >= 2:
                human_text = item["conversations"][0]["value"]
                assistant_text = item["conversations"][1]["value"]
                formatted_data.append(format_sharegpt(human_text, assistant_text))
                
        random.shuffle(formatted_data)
        result = formatted_data[:target_count]
        print(f"✅ 成功获取 ShareGPT 数据: {len(result)} 条")
        return result
    except Exception as e:
        print(f"❌ 获取 ShareGPT 数据失败: {e}")
        return []

def get_local_roleplay_data():
    """
    加载你本地准备的高情商/Galgame角色扮演数据
    """
    if not os.path.exists(LOCAL_RP_FILE):
        print(f"⚠️ 未检测到本地剧本数据 ({LOCAL_RP_FILE})。")
        print("💡 建议：如果你有喜欢的 Galgame 角色，可以手动提取几千条对话保存为该文件。")
        return []
    
    print(f"🚀 正在加载本地角色扮演剧本: {LOCAL_RP_FILE}")
    try:
        with open(LOCAL_RP_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        print(f"✅ 成功加载本地剧本数据: {len(data)} 条")
        return data
    except Exception as e:
        print(f"❌ 加载本地数据失败: {e}")
        return []

def main():
    print("="*50)
    print("开始构建 Navi SFT 混合数据集...")
    print("="*50)
    
    # 1. 收集各路数据
    # 比例规划：8k 弱智吧 (幽默逻辑) + 8k ShareGPT (基础智商) + 你的本地 RP 数据
    ruozhiba_data = get_ruozhiba_data(target_count=8000)
    sharegpt_data = get_sharegpt_data(target_count=8000)
    local_rp_data = get_local_roleplay_data()
    
    # 2. 混合并打乱 (极其重要！如果不打乱，模型会先学成神经病，再学成书呆子)
    mixed_data = ruozhiba_data + sharegpt_data + local_rp_data
    random.seed(42)
    random.shuffle(mixed_data)
    
    if not mixed_data:
        print("❌ 没有收集到任何数据，请检查网络或报错信息。")
        return
        
    # 3. 写入最终的 JSONL 文件
    print(f"\n正在将 {len(mixed_data)} 条混合数据写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("\n🎉 SFT 数据集准备完成！")
    print(f"文件位置: {OUTPUT_FILE}")
    print("下一阶段：编写 sft_train.py，让 Navi 拥有真正的灵魂。")

if __name__ == "__main__":
    main()