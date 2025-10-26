import os

import tiktoken
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig

# Set device
device = 'cuda'
torch.set_float32_matmul_precision('high')

# Load the latest checkpoint
checkpoint_dir = "checkpoints"
latest_checkpoint = max(
    [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
    key=os.path.getctime
)
print(f"Loading checkpoint: {latest_checkpoint}")

# Initialize and load the model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
checkpoint = torch.load(latest_checkpoint, weights_only=False)
model.load_state_dict(checkpoint['model_state'])

# Prepare input tokens
model.eval()
num_return_sequences = 1
max_length = 100
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Thou art")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# Generate text
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=-1)

# Decode and print the output
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("> ", decoded)