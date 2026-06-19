import json
import os

os.makedirs("data", exist_ok=True)

# FIX: Matched the exact name of your file (removed the _2)
input_file = "chat_training_dataset_320.json" 
output_file = "data/chat_data.txt"

print(f"Loading custom dataset: {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} custom persona/chat pairs. Formatting...")

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        user_text = item['user'].strip()
        bot_text = item['bot'].strip()
        f.write(f"User: {user_text}\nAssistant: {bot_text}\n\n")

print(f"Success! Saved fine-tuning text to {output_file}")