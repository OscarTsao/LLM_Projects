import torch
from transformers import AutoModel

print("Testing BERT on RTX 5090...")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").cuda()
x = {
    "input_ids": torch.randint(0, 30000, (2, 32)).cuda(),
    "attention_mask": torch.ones(2, 32).cuda()
}
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    out = model(**x)
print("✓ BERT forward pass successful on GPU")
print(f"Output shape: {out.last_hidden_state.shape}")
print(f"Device: {out.last_hidden_state.device}")
