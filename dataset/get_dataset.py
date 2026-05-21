import os
import sys

# ================= 最关键的修改 =================
# 必须在导入任何 huggingface/datasets 相关的库之前设置环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =================================================

from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

# 开启详细日志和进度条
set_verbosity_info()
enable_progress_bar()

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "hf_cache")
save_path = os.path.join(base_dir, "skypile_15b")

print("="*50, flush=True)
print(f"当前工作目录: {base_dir}", flush=True)
print(f"数据缓存路径: {cache_dir}", flush=True)
print(f"最终保存路径: {save_path}", flush=True)
print("="*50, flush=True)

def main():
    print("🚀 正在通过 HF 镜像源直连拉取 SkyPile-150B ...", flush=True)
    
    try:
        # 加载数据集
        dataset = load_dataset(
            "SkyWork/SkyPile-150B", 
            split="train[:10%]", 
            cache_dir=cache_dir
        )
        
        print(f"\n✅ 成功获取数据集，共包含 {len(dataset)} 篇文章/文档。", flush=True)
        
        print(f"正在整理并保存至: {save_path} ...", flush=True)
        dataset.save_to_disk(save_path)
        
        print("\n🎉 保存完成！可以开始训练你的 Tokenizer 了。", flush=True)

    except Exception as e:
        print(f"\n❌ 下载或处理过程中出现错误:\n{e}", flush=True)

if __name__ == "__main__":
    main()