import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

# Mac GPU (Metal)
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
model.eval()

prompt = "介绍一下人工智能"

tokens = tokenizer.encode(prompt)

input_ids = torch.tensor([tokens]).to(device)

generated = input_ids

for step in range(50):

    with torch.no_grad():
        outputs = model(generated)

    logits = outputs.logits

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1
    )

    generated = torch.cat(
        [generated, next_token.unsqueeze(0)],
        dim=1
    )

    text = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    print(f"Step {step}:")
    print(text)
    print("-" * 50)