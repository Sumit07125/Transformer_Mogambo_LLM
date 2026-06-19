from tokenizers import ByteLevelBPETokenizer
import os

os.makedirs("tokenizer", exist_ok=True)
tokenizer = ByteLevelBPETokenizer()

# Train ONLY on your custom text
tokenizer.train(
    files=["data/chat_data.txt"],
    vocab_size=2000,   # Tiny vocabulary for absolute purity
    min_frequency=1,   # Forces it to learn every single word
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

tokenizer.save_model("tokenizer")
print("Pure Mogambo Tokenizer Created!")