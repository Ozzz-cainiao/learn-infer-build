from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

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


class FinishReason(Enum):
    EOS = "eos"
    LENGTH = "length"
    STOP_STRING = "stop_string"


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
# Runtime State
# ============================================================


class RuntimeState:
    """
    Runtime state of one generation request.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: torch.Tensor,
        sampling_params: SamplingParams,
    ):
        # ---------- Request ----------
        self.request_id = request_id

        # ---------- Input ----------
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids

        # ---------- Config ----------
        self.sampling_params = sampling_params

        # ---------- Runtime ----------
        self.generated_token_ids: List[int] = []
        self.generated_text = ""
        self.current_token_id: int
        self.logits = None
        self.past_key_values = None

        # ---------- Status ----------
        self.finished = False
        self.finish_reason: FinishReason | None

    def append_token(self, token_id: int):
        self.generated_token_ids.append(token_id)

    def get_text(self, tokenizer):
        return tokenizer.decode(
            self.generated_token_ids,
            skip_special_tokens=True,
        )

    def finish(self, reason: FinishReason):
        self.finished = True
        self.finish_reason = reason


# ============================================================
# Generation Output
# ============================================================


@dataclass
class GenerationOutput:
    """
    One generation event.
    """

    token_id: int | None

    delta_text: str | None

    finished: bool = False

    finish_reason: Optional[FinishReason] = None


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
) -> FinishReason | None:
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
        return FinishReason.LENGTH

    # Stop Condition 2:
    # EOS Token

    if (
        tokenizer.eos_token_id is not None
        and len(generated_token_ids) > 0
        and generated_token_ids[-1] == tokenizer.eos_token_id
    ):
        print("Hit EOS Token")
        return FinishReason.EOS

    # Stop Condition 3:
    # Stop String

    if stop_strings:

        for stop_str in stop_strings:

            if generated_text.endswith(stop_str):
                print(f"Hit Stop String: {repr(stop_str)}")
                return FinishReason.STOP_STRING

    return None


# ============================================================
# Generate Loop
# ============================================================


class Generator:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_tokens(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ):
        # --------------------------------------------------------
        # Encode
        # --------------------------------------------------------

        runtime = RuntimeState(
            request_id="0",
            prompt=prompt,
            prompt_token_ids=self.tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self.device),
            sampling_params=sampling_params,
        )

        # --------------------------------------------------------
        # Prefill
        # --------------------------------------------------------

        runtime.logits, runtime.past_key_values = prefill(
            model=self.model,
            input_ids=runtime.prompt_token_ids,
        )

        # --------------------------------------------------------
        # First token
        # --------------------------------------------------------

        runtime.current_token_id = sample(
            runtime.logits[0, -1],
            strategy=sampling_params.strategy,
            temperature=sampling_params.temperature,
        )

        runtime.append_token(runtime.current_token_id)

        # --------------------------------------------------------
        # Decode Loop
        # --------------------------------------------------------

        for step in range(sampling_params.max_new_tokens - 1):

            runtime.logits, runtime.past_key_values = decode_one_token(
                model=self.model,
                token_id=runtime.current_token_id,
                past_key_values=runtime.past_key_values,
                device=self.device,
            )

            runtime.current_token_id = sample(
                runtime.logits[0, -1],
                strategy=sampling_params.strategy,
                temperature=sampling_params.temperature,
            )

            runtime.append_token(runtime.current_token_id)
            prev_text = runtime.generated_text

            current_text = runtime.get_text(self.tokenizer)

            if current_text.startswith(prev_text):
                delta = current_text[len(prev_text) :]
            else:
                delta = current_text

            runtime.generated_text = current_text
            yield GenerationOutput(
                token_id=runtime.current_token_id,
                delta_text=delta,
                finished=runtime.finished,
                finish_reason=None,
            )

            finish_reason = should_stop(
                generated_token_ids=runtime.generated_token_ids,
                generated_text=runtime.generated_text,
                tokenizer=self.tokenizer,
                max_new_tokens=sampling_params.max_new_tokens,
                stop_strings=sampling_params.stop_strings,
            )
            if finish_reason:
                runtime.finish(finish_reason)
                print(f"runtime.finish_reason: {runtime.finish_reason}")

                break
        yield GenerationOutput(
            token_id=None,
            delta_text="",
            finished=runtime.finished,
            finish_reason=runtime.finish_reason,
        )

        return runtime.generated_text

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ):

        pieces = []

        for output in self.generate_tokens(
            prompt,
            sampling_params,
        ):
            pieces.append(output.delta_text)

        return "".join(pieces)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ):

        for output in self.generate_tokens(
            prompt,
            sampling_params,
        ):
            yield output.delta_text


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

    generator = Generator(model, tokenizer, device)

    final_text = ""

    for chunk in generator.generate_stream(
        prompt=prompt,
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
