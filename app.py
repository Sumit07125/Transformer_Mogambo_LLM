from flask import Flask, render_template, request, jsonify
import torch
from tokenizers import ByteLevelBPETokenizer
from src.model import GPT

app = Flask(__name__)

# -------------------------
# LOAD MOGAMBO AT STARTUP
# -------------------------
CHECKPOINT_PATH = "checkpoints/mogambo_pure.pt"
DEVICE = (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Loading Mogambo onto {DEVICE}...")
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

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
def get_mogambo_reply(user_text):
    prompt = f"User: {user_text}\nAssistant:"
    ids = tokenizer.encode(prompt).ids
    
    if len(ids) == 0:
        return "..."
        
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    # Generation loop
    for _ in range(40):
        idx_cond = x[:, -checkpoint["block_size"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / 0.1  # temperature=0.1
        
        v, _ = torch.topk(logits, 1)     # top_k=1
        logits[logits < v[:, [-1]]] = -float("inf")
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
        
    # Decode only the newly generated tokens
    reply = tokenizer.decode(x[0].tolist()[len(ids):])

    # Clean the output cutoff
    for stop_word in ["User:", "Assistant:", "\n\n", "<pad>"]:
        if stop_word in reply:
            reply = reply.split(stop_word)[0]
            
    return reply.strip()

# -------------------------
# WEB ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    reply = get_mogambo_reply(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("\n🌐 Web Interface Running! Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)