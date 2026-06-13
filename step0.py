# 引入依赖
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    # 确定模型
    MODEL_NAME = "Qwen/Qwen3-0.6B"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(type(tokenizer))

    device = get_device()
    print(f"Using device: {device}")
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    print(type(model))

    # 打印模型大小
    num_params = sum(p.numel() for p in model.parameters())

    print(num_params)

    # 把模型切换到推理评估模式
    model.eval()

    print(next(model.parameters()).device)

    # 增加输出
    prompt = "介绍一下人工智能"
    token_ids = tokenizer.encode(prompt)
    print(token_ids)
    print(len(token_ids))

    for token_id in token_ids:
        print(token_id, tokenizer.decode(token_id))

    input_ids = torch.tensor([token_ids]).to(
        device
    )  # 因为 transformer 的输入约定是[batch_size, sequence_length] 所以多加了一个 batch 维度
    print(input_ids)
    print(input_ids.shape)

    outputs = model(input_ids)
    logits = outputs.logits

    print("input_ids.shape", input_ids.shape)
    print("logits.shape", logits.shape)  # logits [batch, seq, vocab_size]
    print(logits)
    print(logits[0, 0, :], logits[0, 0, :].shape)  #

    last_logits = logits[:, -1, :]  # 只有最后一个 token 的预测真正用于生成下一token
    next_token = torch.argmax(last_logits, dim=-1)
    print("next_token:", next_token)
    print("decoded:", tokenizer.decode(next_token.tolist()))

    prompt = "请介绍一下上海"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"input_ids: {input_ids}")
    max_new_token = 10
    generated_token_ids = []
    with torch.no_grad():
        # prefill
        outputs = model(input_ids=input_ids, use_cache=True)
        print(f"outputs: {outputs}")
        print(f"type(outputs): {type(outputs)}")
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        print(f"logits: {logits}")
        print(f"logits.shape: {logits.shape}")  # [batch, seq_len, vocab_size]
        print(f"past_key_values: {past_key_values}")
        print(f"type(past_key_values): {type(past_key_values)}")

        # 只有最后一个位置可以用来预测下一个 TOKEN
        # logits 对应的是模型对词表中每个位置的打分，不是概率，经过 softmax 后可以转换为概率
        print(f"logits[0, -1]: {logits[0, -1]}")

        # greedy 从最后一个位置取 argmax
        next_token_id = int(
            torch.argmax(
                logits[0, -1],
                dim=-1,
            ).item()
        )
        print(f"next_token_id: {next_token_id}")
        generated_token_ids.append(next_token_id)
        print(f"generated_token_ids: {generated_token_ids}")

        # decode
        for step in range(max_new_token - 1):
            next_input_id = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=device
            )  # 使用 dtype 显示声明是整数
            print(f"next_input_id: {next_input_id}")  # 实际就是 prefill 的第一个 TOKEN

            output = model(
                input_ids=next_input_id, past_key_values=past_key_values, use_cache=True
            )
            logits = output.logits
            past_key_values = output.past_key_values

            # greedy 从最后一个位置取 argmax
            next_token_id = int(
                torch.argmax(
                    logits[0, -1],
                    dim=-1,
                ).item()
            )
            print(f"next_token_id: {next_token_id}")
            generated_token_ids.append(next_token_id)
            print(f"generated_token_ids: {generated_token_ids}")

            # 判断是否截止了
            if (
                tokenizer.eos_token_id is not None
                and next_token_id == tokenizer.eos_token_id
            ):
                print(f"Hit EOS at step {step + 1}")
                break
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        print("\n==== PROMOTE ====")
        print(prompt)

        print("\n==== GENERATED ====")
        print(generated_text)


if __name__ == "__main__":
    main()
