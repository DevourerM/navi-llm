import os
import sys
import torch
import torch.nn.functional as F

# 将项目根目录加入环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.token import NaviTokenizer

# ================= 路径配置 =================
BEST_PATH = os.path.join(BASE_DIR, "model", "navi_sft_final.pth")
FINAL_PATH = os.path.join(BASE_DIR, "model", "final_model.pth")

# 终端颜色代码 (Ollama 风格)
COLOR_USER = '\033[94m'      # 蓝色
COLOR_NAVI = '\033[92m'      # 绿色
COLOR_SYSTEM = '\033[93m'    # 黄色
COLOR_RESET = '\033[0m'      # 重置


# =====================================================================
# 1. 底层流式生成器 (核心推理引擎)
# =====================================================================
@torch.no_grad()
def generate_stream(model, tokenizer, input_tokens, max_new_tokens=512, temperature=0.7, top_k=50, device='cuda'):
    """内部使用的流式生成核心逻辑"""
    model.eval()
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # 截断超长上下文保护显存
        max_seq_len = model.args.max_seq_len
        if input_ids.size(1) >= max_seq_len:
            input_ids = input_ids[:, -max_seq_len+1:]
            
        logits, _ = model(input_ids)
        next_token_logits = logits[0, -1, :]
        
        # 采样策略 (Temperature & Top-K)
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        
        if token_id == tokenizer.eos_id:
            break
            
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        yield tokenizer.decode([token_id]), token_id


# =====================================================================
# 2. 单轮对话函数 (供外部无状态直接调用)
# =====================================================================
def single_turn_chat(prompt: str, model: NaviLLM, tokenizer: NaviTokenizer, device='cuda', max_new_tokens=512, temperature=0.7) -> str:
    """
    纯净的单轮对话接口：无记忆，一问一答，返回完整字符串。
    注意：为了性能，必须由外部传入已加载好的 model 和 tokenizer，防止重复加载权重。
    
    使用场景：FastAPI 接口、一次性文本处理、批量推理任务。
    """
    input_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    response_text = ""
    
    for chunk, _ in generate_stream(model, tokenizer, input_tokens, max_new_tokens, temperature, device=device):
        response_text += chunk
        
    return response_text


# =====================================================================
# 3. Navi 核心类 (多轮上下文状态机，供外部灵活集成)
# =====================================================================
class Navi:
    """
    Navi 多轮对话封装类。
    负责管理模型生命周期、设备调度以及多轮对话的 Token 历史记忆。
    """
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载分词器
        self.tokenizer = NaviTokenizer()
        
        # 加载模型
        self.args = ModelArgs()
        self.model = NaviLLM(self.args).to(self.device)
        
        path_to_load = model_path or (BEST_PATH if os.path.exists(BEST_PATH) else FINAL_PATH)
        if not os.path.exists(path_to_load):
            raise FileNotFoundError(f"找不到模型权重: {path_to_load}")
            
        self.model.load_state_dict(torch.load(path_to_load, map_location=self.device))
        self.model.eval()
        
        # 初始化上下文记忆库
        self.history_tokens = []
        
    def clear_memory(self):
        """清空对话历史"""
        self.history_tokens = []

    def chat_stream(self, user_prompt: str, temperature=0.7, top_k=50):
        """
        多轮对话接口 (流式版)：保持记忆并实时 yield 吐字。
        适合用于构建带有打字机特效的 UI 或终端。
        """
        # 编码用户输入并加入历史
        input_tokens = self.tokenizer.encode(user_prompt, add_bos=True, add_eos=False)
        self.history_tokens.extend(input_tokens)
        
        response_tokens = []
        for text_chunk, token_id in generate_stream(self.model, self.tokenizer, self.history_tokens, 
                                                    temperature=temperature, top_k=top_k, device=self.device):
            response_tokens.append(token_id)
            yield text_chunk
            
        # 记录模型回答和结束符到历史
        self.history_tokens.extend(response_tokens)
        self.history_tokens.append(self.tokenizer.eos_id)

    def chat(self, user_prompt: str, temperature=0.7, top_k=50) -> str:
        """
        多轮对话接口 (阻塞版)：保持记忆，等整段话生成完一次性返回。
        """
        response_text = ""
        for chunk in self.chat_stream(user_prompt, temperature, top_k):
            response_text += chunk
        return response_text


# =====================================================================
# 4. CLI 交互终端 (直接运行该文件时触发)
# =====================================================================
def start_cli():
    """启动类似 Ollama 的终端交互界面"""
    print(f"{COLOR_SYSTEM}正在唤醒 Navi-LLM 核心引擎...{COLOR_RESET}")
    
    try:
        navi = Navi()
    except Exception as e:
        print(f"{COLOR_SYSTEM}❌ 引擎启动失败: {e}{COLOR_RESET}")
        print(f"{COLOR_SYSTEM}提示：请确保已运行 dataset/token.py 和 train/llm_train.py{COLOR_RESET}")
        return

    print(f"{COLOR_SYSTEM}✅ Navi 准备就绪！{COLOR_RESET}")
    print(f"{COLOR_SYSTEM}(提示: 输入 'exit' 退出，输入 'clear' 清空上下文记忆){COLOR_RESET}")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n{COLOR_USER}>>> User:{COLOR_RESET} ")
            command = user_input.strip().lower()
            
            if command in ['exit', 'quit']:
                print(f"{COLOR_SYSTEM}Navi 正在休眠，下次见！{COLOR_RESET}")
                break
            if command == 'clear':
                navi.clear_memory()
                print(f"{COLOR_SYSTEM}[上下文记忆已清空]{COLOR_RESET}")
                continue
            if not command:
                continue
                
            print(f"{COLOR_NAVI}>>> Navi:{COLOR_RESET} ", end="", flush=True)
            
            # 调用封装好的流式多轮接口
            for chunk in navi.chat_stream(user_input):
                print(chunk, end="", flush=True)
                
            print() # 换行收尾
            
        except KeyboardInterrupt:
            print(f"\n{COLOR_SYSTEM}[生成被用户中断]{COLOR_RESET}")
            # 遇到打断时，可以选择安全地截断当前不完整的对话记忆
            navi.history_tokens.append(navi.tokenizer.eos_id)
            continue
        except Exception as e:
            print(f"\n{COLOR_SYSTEM}发生运行时错误: {e}{COLOR_RESET}")

if __name__ == "__main__":
    start_cli()


"""
example:

from navi import Navi, single_turn_chat

# 启动时实例化一次，模型驻留显存
navi_engine = Navi()

# 后续可以无数次极速调用单轮 API
ans = single_turn_chat("请总结一下相对论", navi_engine.model, navi_engine.tokenizer)
print(ans)

_______________________________

from navi import Navi

pet_brain = Navi()

# 第一轮
reply1 = pet_brain.chat("你好，我是你的主人！")
print(reply1)

# 第二轮（模型会记得你是它的主人）
reply2 = pet_brain.chat("我刚跟你说了什么？")
print(reply2)

"""