import json
import os
import csv
import random
import time
from collections import defaultdict
import requests
from datasets import load_dataset

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# =========================================================================
# 📂 PART 1: DailyDialog & EmpatheticDialogues Processing
# =========================================================================
print("=== PART 1: Starting online dataset download and preparation ===")
output_file_1 = "data/chat_pairs.jsonl"
pairs_count_1 = 0

with open(output_file_1, "w", encoding="utf-8") as out:
    
    print("\n[1/2] Fetching DailyDialog from Hugging Face...")
    # OpenRL provides a modern Parquet version of DailyDialog
    daily_dialog = load_dataset("OpenRL/daily_dialog")
    
    for split in ['train', 'validation', 'test']:
        for item in daily_dialog[split]:
            dialog = item['dialog']
            # Extract consecutive alternating message pairs
            for i in range(len(dialog) - 1):
                pair = {
                    "user": dialog[i].strip(),
                    "assistant": dialog[i+1].strip()
                }
                out.write(json.dumps(pair) + "\n")
                pairs_count_1 += 1

    print("\n[2/2] Fetching EmpatheticDialogues from Hugging Face...")
    empathetic = load_dataset("Ahren09/empathetic_dialogues")
    
    for split in ['train', 'validation', 'test']:
        # Group utterances by their conversation ID
        convs = defaultdict(list)
        for item in empathetic[split]:
            convs[item['conv_id']].append((item['utterance_idx'], item['utterance']))
        
        for conv_id, utterances in convs.items():
            # Sort chronologically by utterance index
            utterances.sort(key=lambda x: x[0])
            for i in range(len(utterances) - 1):
                pair = {
                    "user": utterances[i][1].strip(),
                    "assistant": utterances[i+1][1].strip()
                }
                out.write(json.dumps(pair) + "\n")
                pairs_count_1 += 1

print(f"\n✅ Success! Compiled {pairs_count_1} high-quality chat pairs into {output_file_1}\n")
print("-" * 70)


# =========================================================================
# 📂 PART 2: Bitext Customer Support CSV Dataset Download
# =========================================================================
print("=== PART 2: Downloading the Customer Support CSV Safely ===")

csv_url = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
csv_path = "data/raw_dataset.csv"
output_file_2 = "data/chat_data.txt"

print("\nDownloading the raw CSV safely...")

# Robust download with retries and a progress indicator
max_retries = 3
for attempt in range(max_retries):
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status() # Check if the request was actually successful
        
        with open(csv_path, "wb") as f:
            f.write(response.content)
            
        print("Download complete!")
        break # Exit the loop if successful
    except requests.exceptions.RequestException as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Failed to download after 3 attempts. Please check your internet connection.")
            exit()

print("Formatting data for Mogambo...")
count_2 = 0
with open(csv_path, "r", encoding="utf-8") as infile, open(output_file_2, "w", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    for row in reader:
        instruction = row.get("instruction", "").strip()
        response = row.get("response", "").strip()
        
        if instruction and response:
            formatted_turn = f"User: {instruction}\nAssistant: {response}\n\n"
            outfile.write(formatted_turn)
            count_2 += 1

print(f"\n✅ Success! Processed {count_2} ultra-clean QA pairs into {output_file_2}\n")
print("-" * 70)


# =========================================================================
# 📂 PART 3: Full Intelligence Dataset (Alpaca + DailyDialog Mixed)
# =========================================================================
print("=== PART 3: Downloading the FULL Intelligence Dataset ===")
output_file_3 = "data/chat_pairs_mixed.jsonl" # Saved uniquely to prevent file collisions

print("\nFetching Stanford Alpaca...")
all_pairs = []
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
for item in alpaca:
    user_text = item['instruction']
    if item['input']:
        user_text += f"\nContext: {item['input']}"
    all_pairs.append({
        "user": user_text.strip(),
        "assistant": item['output'].strip()
    })

print("Fetching Conversational Data...")
daily_dialog_train = load_dataset("OpenRL/daily_dialog", split="train")
for item in daily_dialog_train:
    dialog = item['dialog']
    for i in range(len(dialog) - 1):
        all_pairs.append({
            "user": dialog[i].strip(),
            "assistant": dialog[i+1].strip()
        })

print("Shuffling and mixing 100% of the data...")
random.shuffle(all_pairs)

with open(output_file_3, "w", encoding="utf-8") as out:
    for pair in all_pairs:
        out.write(json.dumps(pair) + "\n")

print(f"\n✅ Success! Saved all {len(all_pairs)} mixed logic/chat pairs to {output_file_3}\n")
print("=" * 70)
print("🎉 ALL PIPELINES COMPLETED SUCCESSFULY!")