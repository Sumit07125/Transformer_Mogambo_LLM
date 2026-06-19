import os
import json
import time
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from model import GPT

# -------------------------
# FROM-SCRATCH SETTINGS
# -------------------------
BATCH_SIZE = 16       
BLOCK_SIZE = 128      # Shorter block size for simple Q&A
MAX_ITERS = 1500      # 1500 steps to learn language from total zero
EVAL_INTERVAL = 100
LEARNING_RATE = 5e-4  # Faster learning rate to build connections quickly
CHECKPOINT_PATH = "checkpoints/mogambo_pure.pt"
DEVICE = (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
pad_id = tokenizer.token_to_id("<pad>")

with open("chat_training_dataset_320.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed_sequences = []
print("Tokenizing and isolating each example...")
for item in raw_data:
    formatted_text = f"User: {item['user'].strip()}\nAssistant: {item['bot'].strip()}\n\n"
    ids = tokenizer.encode(formatted_text).ids
    
    if len(ids) > BLOCK_SIZE:
        ids = ids[:BLOCK_SIZE]
    else:
        ids = ids + [pad_id] * (BLOCK_SIZE - len(ids))
        
    processed_sequences.append(ids)

all_data = torch.tensor(processed_sequences, dtype=torch.long)

def get_batch():
    ix = torch.randint(0, len(all_data), (BATCH_SIZE,))
    batch_data = all_data[ix] 
    x = batch_data[:, :-1]  
    y = batch_data[:, 1:]   
    return x.to(DEVICE), y.to(DEVICE)

# -------------------------
# BUILD A BRAND NEW BRAIN
# -------------------------
vocab_size = len(tokenizer.get_vocab())

# Tiny Brain (5M params) - Perfect for memorizing 320 lines instantly
model = GPT(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=256, n_head=8, n_layer=6).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n🚀 Training BRAND NEW Brain from Scratch...")
start_time = time.time()

for step in range(MAX_ITERS):
    xb, yb = get_batch()
    logits, _ = model(xb, targets=None)
    
    # Masked Loss: Ignore the padding cheat code!
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1), ignore_index=pad_id)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        print(f"Step {step:4d}/{MAX_ITERS} | Loss: {loss.item():.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(
    {"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "block_size": BLOCK_SIZE},
    CHECKPOINT_PATH
)
print(f"\nPure Mogambo brain saved to {CHECKPOINT_PATH}!")