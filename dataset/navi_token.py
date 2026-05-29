import os
import sentencepiece as spm
from datasets import load_from_disk

# 获取当前路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "skypile_100b")
MODEL_PREFIX = os.path.join(BASE_DIR, "dataset", "navi_tokenizer")

def train_tokenizer(vocab_size=65024, sample_size=500000):
    """
    从本地下载好的 HF 数据集中抽取 sample_size 篇文章，训练 SentencePiece 分词器。
    """
    print(f"正在从 {DATASET_PATH} 加载数据集...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"数据集未准备好，请等待 get_dataset.py 下载完成。报错: {e}")
        return

    # 1. 将抽样数据导出为纯文本文件供 SentencePiece 训练
    txt_path = os.path.join(BASE_DIR, "dataset", "train_vocab_sample.txt")
    print(f"抽取 {sample_size} 条数据生成训练文本...")
    with open(txt_path, "w", encoding="utf-8") as f:
        # 假设 SkyPile 的正文字段是 "text"
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            # 去除首尾空白并写入
            text = item.get("text", "").strip()
            if text:
                f.write(text + "\n")
    
    # 2. 训练 SentencePiece 模型
    print("开始训练 SentencePiece 词表 (可能需要十几分钟)...")
    spm.SentencePieceTrainer.train(
        input=txt_path,
        model_prefix=MODEL_PREFIX,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe", # 使用 BPE 算法 (与 LLaMA 相同)
        pad_id=0,
        unk_id=1,
        bos_id=2, # Begin of Sentence
        eos_id=3, # End of Sentence
    )
    print(f"🎉 分词器训练完成！模型已保存为 {MODEL_PREFIX}.model 和 .vocab")
    
    # 清理临时文本文件
    if os.path.exists(txt_path):
        os.remove(txt_path)

class NaviTokenizer:
    """提供给模型训练时调用的分词器包装类"""
    def __init__(self):
        model_path = f"{MODEL_PREFIX}.model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到分词器模型 {model_path}，请先运行 train_tokenizer()！")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text, add_bos=True, add_eos=True):
        """将文本转换为 Token ID 序列"""
        ids = self.sp.encode_as_ids(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        """将 Token ID 序列还原为文本"""
        return self.sp.decode_ids(ids)

if __name__ == "__main__":
    # 当直接运行此脚本时，执行训练流程
    train_tokenizer()