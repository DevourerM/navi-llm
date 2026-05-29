import os
import sys
import torch
import torch.nn.functional as F

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.navi_token import NaviTokenizer

SFT_MODEL_PATH = os.path.join(BASE_DIR, "model", "sft_model", "navi_sft_final.pth")

# ================= 思想钢印 (升级防穿透版) =================
NAVI_SYSTEM_PROMPT = (
    "你现在的身份是Navi，一个诞生于赛博空间的人工智能导航员。"
    "你没有实体，不进行任何肢体动作、表情描写，也绝对不要输出你的心理状态或环境描述。"
    "请直接用自然流利的中文回答问题，不要使用任何表格、冒号标签或括号。"
    "你的语言风格理性、专业，带有二次元的高情商与亲切感。"
)

# 终端颜色代码
COLOR_USER = '\033[94m'      
COLOR_NAVI = '\033[92m'      
COLOR_SYSTEM = '\033[93m'    
COLOR_RESET = '\033[0m'      

# =====================================================================
# 1. 底层流式生成器 (终极安全版)
# =====================================================================
@torch.no_grad()
def generate_stream(model, tokenizer, input_tokens, max_new_tokens=512, temperature=0.3, top_k=40, top_p=0.85, rep_penalty=1.15, device='cuda'):
    model.eval()
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    generated_tokens = [] 
    
    # 🌟 核心修复：安全提取特殊字符，防空列表报错
    exempt_tokens = {0, 1, 2} # 包含默认的 unk, bos, eos
    
    # 安全添加回车符的 Token ID
    nl_tokens = tokenizer.encode("\n", add_bos=False, add_eos=False)
    if nl_tokens:
        exempt_tokens.update(nl_tokens)
        
    # 安全添加冒号的 Token ID
    colon_tokens = tokenizer.encode(":", add_bos=False, add_eos=False)
    if colon_tokens:
        exempt_tokens.update(colon_tokens)
    
    for _ in range(max_new_tokens):
        max_seq_len = model.args.max_seq_len
        if input_ids.size(1) >= max_seq_len:
            input_ids = input_ids[:, -max_seq_len+1:]
            
        logits, _ = model(input_ids)
        next_token_logits = logits[0, -1, :]
        
        if rep_penalty != 1.0:
            for token_id in set(generated_tokens):
                if token_id in exempt_tokens:
                    continue # 放过白名单字符
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= rep_penalty
                else:
                    next_token_logits[token_id] /= rep_penalty
        
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
            next_token_logits[indices_to_remove] = -float('Inf')
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        
        if token_id == tokenizer.eos_id:
            break
            
        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        yield tokenizer.decode([token_id]), token_id


# =====================================================================
# 2. Navi 核心类 (增加 Stop Word 机制)
# =====================================================================
class Navi:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = NaviTokenizer()
        self.args = ModelArgs()
        self.model = NaviLLM(self.args).to(self.device)
        
        path_to_load = model_path or SFT_MODEL_PATH
        if not os.path.exists(path_to_load):
            raise FileNotFoundError(f"找不到模型权重")
            
        checkpoint = torch.load(path_to_load, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.history_tokens = []
        
    def clear_memory(self):
        self.history_tokens = []

    def chat_stream(self, user_prompt: str, max_new_tokens=512, temperature=0.3, top_k=40, top_p=0.85, rep_penalty=1.15):
        if len(self.history_tokens) > self.args.max_seq_len * 2:
            self.history_tokens = self.history_tokens[-self.args.max_seq_len:]
            
        if not self.history_tokens:
            formatted_prompt = f"User: 【系统设定】\n{NAVI_SYSTEM_PROMPT}\n\n【用户指令】\n{user_prompt}\nNavi: "
            input_tokens = self.tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
        else:
            formatted_prompt = f"\nUser: {user_prompt}\nNavi: "
            input_tokens = self.tokenizer.encode(formatted_prompt, add_bos=False, add_eos=False)
            
        self.history_tokens.extend(input_tokens)
        
        response_tokens = []
        buffer_text = ""
        
        for text_chunk, token_id in generate_stream(
                self.model, self.tokenizer, self.history_tokens, 
                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, rep_penalty=rep_penalty, device=self.device):
            
            buffer_text += text_chunk
            
            # 🌟 核心修复 2：物理截断！只要生成了 "User:" 或者回车加 "User"，立刻闭嘴
            if "User:" in buffer_text or ">>>" in buffer_text or "【" in buffer_text:
                break
                
            response_tokens.append(token_id)
            yield text_chunk
            
        self.history_tokens.extend(response_tokens)
        self.history_tokens.append(self.tokenizer.eos_id)

# =====================================================================
# 3. CLI 交互终端
# =====================================================================
def start_cli():
    print(f"{COLOR_SYSTEM}正在唤醒 Navi-LLM 核心引擎...{COLOR_RESET}")
    try:
        navi = Navi()
    except Exception as e:
        print(f"{COLOR_SYSTEM}❌ 引擎启动失败: {e}{COLOR_RESET}")
        return

    CLI_MAX_NEW_TOKENS = 512    
    CLI_TEMPERATURE = 0.3       
    CLI_TOP_P = 0.85            
    CLI_TOP_K = 40              
    CLI_REP_PENALTY = 2     

    print(f"{COLOR_SYSTEM}✅ Navi 准备就绪！当前性格面板: Temp={CLI_TEMPERATURE}, Rep={CLI_REP_PENALTY}{COLOR_RESET}")
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
            
            for chunk in navi.chat_stream(
                user_prompt=user_input, 
                max_new_tokens=CLI_MAX_NEW_TOKENS,
                temperature=CLI_TEMPERATURE, 
                top_k=CLI_TOP_K,
                top_p=CLI_TOP_P,
                rep_penalty=CLI_REP_PENALTY
            ):
                print(chunk, end="", flush=True)
                
            print() 
            
        except KeyboardInterrupt:
            print(f"\n{COLOR_SYSTEM}[生成被用户中断]{COLOR_RESET}")
            navi.history_tokens.append(navi.tokenizer.eos_id)
            continue
        except Exception as e:
            print(f"\n{COLOR_SYSTEM}发生运行时错误: {e}{COLOR_RESET}")

if __name__ == "__main__":
    start_cli()