from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
from typing import Optional, List

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


@dataclass
class SamplingParams:
    strategy: str = "greedy"  # greedy / temperature
    temperature: float = 1.0

    max_new_tokens: int = 50
    stop_strings: Optional[List[str]] = None


# ============================================================
# Sampler (Unified Entry)
# ============================================================


def sample(
    logits: torch.Tensor,
    strategy: str = "greedy",
    temperature: float = 1.0,
) -> int:
    """
    Unified Sampling Entry

    logits shape:
        [vocab_size]

    return:
        next token id
    """

    if strategy == "greedy":
        return int(torch.argmax(logits, dim=-1).item())

    elif strategy == "temperature":
        temperature = max(temperature, 1e-6)

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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
# Stop Policy
# ============================================================


def should_stop(
    generated_token_ids,
    generated_text,
    tokenizer,
    max_new_tokens,
    stop_strings=None,
):
    """
    Stop Policy

    当前支持：
    1. max_new_tokens
    2. eos_token
    3. stop_strings
    """

    # Stop Condition 1:
    # 达到最大生成长度

    if len(generated_token_ids) >= max_new_tokens:
        return True

    # Stop Condition 2:
    # EOS Token

    if (
        tokenizer.eos_token_id is not None
        and len(generated_token_ids) > 0
        and generated_token_ids[-1] == tokenizer.eos_token_id
    ):
        print("Hit EOS Token")
        return True

    # Stop Condition 3:
    # Stop String

    if stop_strings:

        for stop_str in stop_strings:

            if generated_text.endswith(stop_str):
                print(f"Hit Stop String: {repr(stop_str)}")
                return True

    return False


# ============================================================
# Generate Loop
# ============================================================


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    sampling_params: SamplingParams,
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

    # --------------------------------------------------------
    # First token
    # --------------------------------------------------------

    next_token_id = sample(
        logits[0, -1],
        strategy=sampling_params.strategy,
        temperature=sampling_params.temperature,
    )

    generated_token_ids.append(next_token_id)

    # --------------------------------------------------------
    # Decode Loop
    # --------------------------------------------------------

    for step in range(sampling_params.max_new_tokens - 1):

        logits, past_key_values = decode_one_token(
            model=model,
            token_id=next_token_id,
            past_key_values=past_key_values,
            device=device,
        )

        next_token_id = sample(
            logits[0, -1],
            strategy=sampling_params.strategy,
            temperature=sampling_params.temperature,
        )

        generated_token_ids.append(next_token_id)

        # current text (needed for stop string)
        current_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        if should_stop(
            generated_token_ids=generated_token_ids,
            generated_text=current_text,
            tokenizer=tokenizer,
            max_new_tokens=sampling_params.max_new_tokens,
            stop_strings=sampling_params.stop_strings,
        ):
            break

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    return generated_text


@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt: str,
    device: str,
    sampling_params: SamplingParams,
):
    generated_token_ids = []

    # -------------------------
    # Encode
    # -------------------------

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(device)

    # -------------------------
    # Prefill
    # -------------------------

    logits, past_key_values = prefill(
        model=model,
        input_ids=input_ids,
    )

    next_token_id = sample(
        logits[0, -1],
        strategy=sampling_params.strategy,
        temperature=sampling_params.temperature,
    )

    generated_token_ids.append(next_token_id)

    prev_text = ""

    # -------------------------
    # Decode Loop
    # -------------------------

    for step in range(sampling_params.max_new_tokens - 1):

        logits, past_key_values = decode_one_token(
            model=model,
            token_id=next_token_id,
            past_key_values=past_key_values,
            device=device,
        )

        next_token_id = sample(
            logits[0, -1],
            strategy=sampling_params.strategy,
            temperature=sampling_params.temperature,
        )

        generated_token_ids.append(next_token_id)

        # full decode (only for diffing, NOT output)
        current_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        # ===== delta logic =====
        if current_text.startswith(prev_text):
            delta = current_text[len(prev_text) :]
        else:
            # fallback (rare tokenizer boundary cases)
            delta = current_text

        prev_text = current_text

        yield delta

        # stop check
        if should_stop(
            generated_token_ids,
            current_text,
            tokenizer,
            sampling_params.max_new_tokens,
            sampling_params.stop_strings,
        ):
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
    params = SamplingParams(
        strategy="temperature",
        temperature=0.8,
        max_new_tokens=200,
        stop_strings=["。"],
    )

    # generated_text = generate_stream(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     device=device,
    #     max_new_tokens=200,
    #     stop_strings=["。"],  # 修复：必须是 list
    #     sampling_strategy="temperature",
    #     temperature=0.8,
    # )

    final_text = ""

    for chunk in generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        sampling_params=params,
    ):
        if chunk:
            print(chunk, end="", flush=True)
            final_text += chunk
    generated_text = final_text

    print("\n==== PROMPT ====")
    print(prompt)

    print("\n==== GENERATED ====")
    print(generated_text)


if __name__ == "__main__":
    main()
