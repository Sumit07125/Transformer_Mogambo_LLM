import torch
from tokenizers import ByteLevelBPETokenizer
from model import GPT

CHECKPOINT_PATH = "checkpoints/mogambo_pure.pt"
DEVICE = (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Match the architecture of the brain we just built
model = GPT(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    n_embd=256,   
    n_head=8,    
    n_layer=6    
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens=40, temperature=0.1, top_k=1):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -checkpoint["block_size"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

print("\n🤖 Mogambo (Pure From-Scratch Build) is online. Type 'exit' to stop.\n")

while True:
    user_text = input("You: ").strip()
    if user_text.lower() == "exit":
        break

    # THE FIX: Removed the trailing space after the colon!
    # The tokenizer will now naturally predict the space attached to the next word.
    prompt = f"User: {user_text}\nAssistant:"
    
    ids = tokenizer.encode(prompt).ids
    
    if len(ids) == 0:
        continue
        
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    out = generate(model, x, max_new_tokens=40, temperature=0.1, top_k=1)
    
    # Decode the newly generated tokens
    reply = tokenizer.decode(out[0].tolist()[len(ids):])

    # Clean text boundary cutoff
    for stop_word in ["User:", "Assistant:", "\n\n", "<pad>"]:
        if stop_word in reply:
            reply = reply.split(stop_word)[0]
            
    print(f"Mogambo: {reply.strip()}\n")