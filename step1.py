from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"


# ============================================================
# Device
# ============================================================


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# ============================================================
# Sampler
# ============================================================


def greedy_sample(logits: torch.Tensor) -> int:
    """
    logits shape:
        [vocab_size]

    return:
        next token id
    """
    return int(torch.argmax(logits, dim=-1).item())


# ============================================================
# Model Execute
# ============================================================


@torch.no_grad()
def prefill(
    model,
    input_ids: torch.Tensor,
):
    """
    Prefill Stage

    input:
        [batch_size, seq_len]

    return:
        logits
        past_key_values
    """

    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    return outputs.logits, outputs.past_key_values


@torch.no_grad()
def decode_one_token(
    model,
    token_id: int,
    past_key_values,
    device: str,
):
    """
    Decode Stage

    每次只输入一个 token

    input shape:
        [1, 1]
    """

    input_ids = torch.tensor(
        [[token_id]],
        dtype=torch.long,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    return outputs.logits, outputs.past_key_values


# ============================================================
# Generate Loop
# ============================================================


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 50,
):
    generated_token_ids = []

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(device)

    # --------------------------------------------------------
    # Prefill
    # --------------------------------------------------------

    logits, past_key_values = prefill(
        model=model,
        input_ids=input_ids,
    )

    # 只取最后一个位置
    next_token_id = greedy_sample(logits[0, -1])

    generated_token_ids.append(next_token_id)

    # --------------------------------------------------------
    # Decode Loop
    # --------------------------------------------------------

    for step in range(max_new_tokens - 1):

        logits, past_key_values = decode_one_token(
            model=model,
            token_id=next_token_id,
            past_key_values=past_key_values,
            device=device,
        )

        next_token_id = greedy_sample(logits[0, -1])

        generated_token_ids.append(next_token_id)

        if (
            tokenizer.eos_token_id is not None
            and next_token_id == tokenizer.eos_token_id
        ):
            print(f"Hit EOS at step {step + 1}")
            break

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    return generated_text


# ============================================================
# Main
# ============================================================


def main():

    device = get_device()

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model Params: {num_params:,}")

    prompt = "请介绍一下上海"

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=50,
    )

    print("\n==== PROMPT ====")
    print(prompt)

    print("\n==== GENERATED ====")
    print(generated_text)


if __name__ == "__main__":
    main()
