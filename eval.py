import argparse
import random
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from model.model import SanzeninMindConfig, SanzeninMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def print_model_params(model):
    # 统计所有可训练参数，方便确认当前加载的是哪个规模的模型
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable params: {total_params / 1e6:.3f}M")


def init_model(args):
    # tokenizer 负责文本 <-> token id 的转换；默认从本地 model/ 目录加载
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    # 官方逻辑：load_from 包含 "model" 时，认为加载的是本项目的原生 PyTorch 模型
    if "model" in args.load_from:
        # 先按命令行参数创建同结构模型；权重 shape 必须和这些参数一致
        model = SanzeninMindForCausalLM(
            SanzeninMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )

        # MoE 权重保存时通常会带 _moe 后缀，用来和 dense 模型区分
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"

        # strict=True 要求权重文件和当前模型结构完全匹配，适合学习时尽早暴露不一致问题
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        # 如果 load_from 不是本地 model/，就按 Hugging Face transformers 格式加载
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            trust_remote_code=True,
        )

    print_model_params(model)

    # 推理阶段切到 eval 模式，关闭 dropout 等训练行为
    model = model.eval().to(args.device)
    # GPU 推理时转半精度节省显存；CPU 上保持 float32 更稳
    if "cuda" in args.device:
        model = model.half()
    return model, tokenizer


def build_inputs(args, tokenizer, conversation, prompt):
    # 预训练权重没有学聊天模板，直接用 BOS + 用户文本做续写
    if "pretrain" in args.weight:
        return tokenizer.bos_token + prompt

    # SFT/Chat 权重通常需要 chat template，把多轮对话渲染成模型训练时见过的格式
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(args.open_thinking),
    )


def main():
    parser = argparse.ArgumentParser(description="SanzeninMind 模型推理与对话")
    parser.add_argument("--load_from", default="model", type=str, help="模型加载路径；model=本项目原生权重，其他路径=transformers格式")
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument("--weight", default="full_sft", type=str, help="权重名称前缀，例如 pretrain 或 full_sft")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度，必须和训练时一致")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量，必须和训练时一致")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构")
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true", help="启用 RoPE 位置编码外推")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="最大新生成 token 数")
    parser.add_argument("--temperature", default=0.85, type=float, help="采样温度，越大越随机")
    parser.add_argument("--top_p", default=0.95, type=float, help="nucleus sampling 阈值")
    parser.add_argument("--open_thinking", default=0, type=int, choices=[0, 1], help="是否开启 thinking 模板参数")
    parser.add_argument("--historys", default=0, type=int, help="携带历史对话轮数；0 表示不携带")
    parser.add_argument("--show_speed", default=1, type=int, choices=[0, 1], help="是否显示生成速度")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="运行设备")
    args = parser.parse_args()

    # 自动测试模式使用的示例问题，方便快速检查模型是否能正常生成
    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        "解释一下机器学习是什么",
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))

    # TextStreamer 会边生成边打印 token，不必等整段生成完成
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 自动模式遍历 prompts；手动模式不断读取用户输入，输入空行结束
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("User: "), "")
    for prompt in prompt_iter:
        # 每轮设置不同随机种子，让采样生成有一定变化
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f"User: {prompt}")

        # 根据参数保留最近若干轮历史；不保留历史时每轮都是单轮对话
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 把 prompt 或对话模板渲染成字符串，再 tokenizer 成张量
        input_text = build_inputs(args, tokenizer, conversation, prompt)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)

        print("Assistant: ", end="")
        start_time = time.time()

        # generate 是 transformers 的生成接口，会循环调用模型 forward 预测下一个 token
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1,
        )

        # 只截取新生成的部分，去掉 prompt 本身
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        conversation.append({"role": "assistant", "content": response})

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f"\n[Speed]: {gen_tokens / (time.time() - start_time):.2f} tokens/s\n")
        else:
            print("\n")


if __name__ == "__main__":
    main()
