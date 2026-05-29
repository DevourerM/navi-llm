import os
import sys
import time

# ================= 最关键的修改 =================
# 必须在导入任何 huggingface 相关的库之前设置环境变量！
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "hf_cache")

# 🌟 核心修复：强行将全局下载缓冲、符号链接、锁文件全部重定向到大硬盘，彻底解放系统盘 (/home)
os.environ["HF_HOME"] = cache_dir

# 2. 必须在导入任何 huggingface 相关的库之前设置环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =================================================

import json
import random
from datasets import load_dataset, DownloadConfig
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

# 开启详细日志和进度条
set_verbosity_info()
enable_progress_bar()

# ================= 路径配置 =================
OUTPUT_FILE = os.path.join(base_dir, "navi_sft_data.jsonl")

print("="*50, flush=True)
print(f"当前工作目录: {base_dir}", flush=True)
print(f"全局 HF 缓冲/缓存路径: {cache_dir}", flush=True)
print(f"最终保存路径: {OUTPUT_FILE}", flush=True)
print("="*50, flush=True)

# ================= 核心设定 =================
NAVI_SYSTEM_PROMPT = (
    "你现在的身份是Navi，一个诞生于赛博空间的人工智能导航员。"
    "你没有实体，不进行任何肢体动作、表情描写，严禁输出任何状态、表格或心理活动。"
    "你的语言风格理性、专业且亲切，直接用自然语言回答用户问题。"
)

# ================= 质量过滤器 =================
# 这些关键词暗示弱智/低质/过于娱乐化的内容，与理性专业人设冲突
LOW_QUALITY_KEYWORDS = [
    "弱智", "傻逼", "脑残", "逗比", "搞笑", "段子", "笑话", "哈哈哈",
    "恶搞", "整蛊", "吐槽", "神回复", "内涵段子", "暴走", "鬼畜",
    "撩妹", "撩汉", "表白", "恋爱", "情感咨询", "分手",
    "角色扮演", "假装你是", "你现在是", "扮演", "cosplay",
    "写小说", "写故事", "写诗", "歌词", "rap",
    "算命", "占卜", "星座", "塔罗", "风水", "面相",
    "夸夸我", "安慰我", "哄我", "陪我聊天", "无聊",
]

# 🌟 模板残留：这些字符串暗示数据来自未填充的模板
TEMPLATE_RESIDUES = [
    "[你的名字]", "[名字]", "[您的名字]", "[姓名]", "[你的邮箱]",
    "[日期]", "[时间]", "[地点]", "[公司名]", "[产品名]",
    "[联系信息]", "[个人简介]", "[简短介绍]",
    "祝好,", "祝好！", "此致", "敬礼",
]

def is_chinese_predominant(text: str, threshold: float = 0.5) -> bool:
    """检测文本是否以中文为主（过滤英文/代码为主的样本）"""
    if len(text) == 0:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return (chinese_chars / total_alpha) >= threshold

def is_rational_quality(instruction: str, output: str) -> bool:
    """过滤掉与理性专业人设不符的低质/娱乐化数据"""
    # 长度过滤
    out = output.strip()
    if len(out) < 30:
        return False
    if len(out) > 2000:
        return False
    
    # 中文为主过滤器
    if not is_chinese_predominant(instruction + out):
        return False
    
    # 关键词过滤
    text = (instruction + output).lower()
    for kw in LOW_QUALITY_KEYWORDS:
        if kw in text:
            return False
    
    # 🌟 模板残留过滤
    for residue in TEMPLATE_RESIDUES:
        if residue in output:
            return False
    
    return True

def format_item(human, assistant):
    """强制标准化格式"""
    return {
        "conversations": [
            {"from": "human", "value": f"【系统设定】\n{NAVI_SYSTEM_PROMPT}\n\n【用户指令】\n{human}"},
            {"from": "assistant", "value": assistant}
        ]
    }

def get_dl_config():
    """统一的下载配置"""
    return DownloadConfig(resume_download=True, max_retries=10)

def get_belle_data(target_count=30000):
    """BelleGroup 通用中文指令数据 —— 覆盖面广、格式规范"""
    print("🚀 [1/3] 正在拉取 BelleGroup/train_1M_CN ...", flush=True)
    
    dataset = load_dataset(
        "BelleGroup/train_1M_CN",
        split="train[:120000]",  # 扩大候选池以获取多样性
        cache_dir=cache_dir,
        download_config=get_dl_config()
    )
    formatted = []
    for item in dataset:
        if len(formatted) >= target_count:
            break
        inst = item['instruction'].strip()
        out = item['output'].strip()
        if not inst or not out:
            continue
        if is_rational_quality(inst, out):
            formatted.append(format_item(inst, out))
    
    print(f"   ✅ BelleGroup 筛选完成: {len(formatted)}/{target_count} 条", flush=True)
    return formatted

def get_alpaca_gpt4_data(target_count=10000):
    """silk-road/alpaca-data-gpt4-chinese —— GPT-4 生成的高质量中文指令数据。
    注意：此数据集含大量英文样本，经中文过滤器后可用量有限。"""
    print("🚀 [2/3] 正在拉取 silk-road/alpaca-data-gpt4-chinese ...", flush=True)
    
    dataset = load_dataset(
        "silk-road/alpaca-data-gpt4-chinese",
        split="train",
        cache_dir=cache_dir,
        download_config=get_dl_config()
    )
    formatted = []
    for item in dataset:
        if len(formatted) >= target_count:
            break
        inst = item['instruction'].strip()
        inp = item.get('input', '').strip()
        out = item['output'].strip()
        
        if inp:
            full_instruction = f"{inst}\n\n{inp}"
        else:
            full_instruction = inst
            
        if not full_instruction or not out:
            continue
        if is_rational_quality(full_instruction, out):
            formatted.append(format_item(full_instruction, out))
    
    print(f"   ✅ Alpaca-GPT4 筛选完成: {len(formatted)}/{target_count} 条", flush=True)
    return formatted

def get_math_reasoning_data(target_count=5000):
    """BelleGroup/school_math_0.25M —— 数学推理数据，强化模型的逻辑思维能力"""
    print("🚀 [3/3] 正在拉取 BelleGroup/school_math_0.25M ...", flush=True)
    
    dataset = load_dataset(
        "BelleGroup/school_math_0.25M",
        split="train",
        cache_dir=cache_dir,
        download_config=get_dl_config()
    )
    formatted = []
    for item in dataset:
        if len(formatted) >= target_count:
            break
        inst = item['instruction'].strip()
        out = item['output'].strip()
        if not inst or not out:
            continue
        if is_rational_quality(inst, out):
            formatted.append(format_item(inst, out))
    
    print(f"   ✅ 数学推理 筛选完成: {len(formatted)}/{target_count} 条", flush=True)
    return formatted

def main():
    print("=" * 50, flush=True)
    print("=== 开始构建 Navi 理性专业 SFT 数据集 ===", flush=True)
    print(f"🎯 目标风格: 理性、专业、亲切的中文AI导航员", flush=True)
    print(f"📊 数据来源: BelleGroup(30k) + Alpaca-GPT4(10k) + 数学推理(5k)", flush=True)
    print(f"🚫 已排除: 弱智吧、角色扮演、情感聊天、英文/代码为主的内容", flush=True)
    print("=" * 50, flush=True)
    
    # 获取数据 — 三个来源互补
    data = get_belle_data(30000) + get_alpaca_gpt4_data(10000) + get_math_reasoning_data(5000)
    
    # 去重（基于 instruction 文本）
    seen = set()
    deduped = []
    for item in data:
        key = item['conversations'][0]['value']
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    
    print(f"\n📊 去重: {len(data)} → {len(deduped)} 条", flush=True)
    
    # 洗牌
    random.seed(42)
    random.shuffle(deduped)
    
    # 写入
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ 数据集构建完成！共 {len(deduped)} 条。", flush=True)
    print(f"📍 文件位置: {OUTPUT_FILE}", flush=True)
    print(f"💡 建议 SFT 训练参数: epochs=3~5, lr=2e-5", flush=True)

if __name__ == "__main__":
    main()