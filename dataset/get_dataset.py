import os
import sys
import time

# ================= 最关键的修改 =================
# 必须在导入任何 huggingface 相关的库之前设置环境变量！
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "hf_cache")
save_path = os.path.join(base_dir, "skypile_100b")  # ~100B tokens, 预计占用 ~430GB

# 🌟 核心修复：强行将全局下载缓冲、符号链接、锁文件全部重定向到大硬盘，彻底解放系统盘 (/home)
os.environ["HF_HOME"] = cache_dir

# 2. 必须在导入任何 huggingface 相关的库之前设置环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =================================================

from datasets import load_dataset, DownloadConfig
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

# 开启详细日志和进度条
set_verbosity_info()
enable_progress_bar()

print("="*50, flush=True)
print(f"当前工作目录: {base_dir}", flush=True)
print(f"全局 HF 缓冲/缓存路径: {cache_dir}", flush=True)
print(f"最终保存路径: {save_path}", flush=True)
print("="*50, flush=True)

def main():
    print("🚀 正在从本地 HF 缓存加载 SkyPile-150B (前 67%, ~100B tokens) ...", flush=True)
    print("⏳ 这需要遍历数据并保存为高效格式，预计 10-30 分钟...", flush=True)
    
    # 工业级防断连配置
    dl_config = DownloadConfig(
        resume_download=True,
        max_retries=10
    )
    
    retry_count = 1
    while True:
        try:
            print(f"\n▶️ [发起第 {retry_count} 次加载请求] 正在从缓存读取...", flush=True)
            
            # 加载数据集 — 全量已在 hf_cache 中，不会重新下载
            dataset = load_dataset(
                "SkyWork/SkyPile-150B", 
                split="train[:67%]",   # ~100B tokens, Chinchilla 最优 ~83 tokens/param
                cache_dir=cache_dir,
                download_config=dl_config
            )
            
            print(f"\n✅ 成功获取数据集，共包含 {len(dataset):,} 篇文章/文档。", flush=True)
            print(f"📊 预估 token 量: ~100B (tokens/param ≈ 83)", flush=True)
            
            print(f"正在整理并保存至: {save_path} ...", flush=True)
            dataset.save_to_disk(save_path)
            
            print("\n🎉 保存完成！", flush=True)
            print(f"💡 提示: 旧数据集 skypile_15b/ (64GB) 现在可以手动删除以释放空间", flush=True)
            break

        except Exception as e:
            print(f"\n❌ [第 {retry_count} 次尝试失败] 错误:\n{e}", flush=True)
            print("⏳ 5秒后将自动重新发起连接并断点续传...", flush=True)
            time.sleep(5)
            retry_count += 1

if __name__ == "__main__":
    main()