import os
import json
import csv
import warnings
import torch
import re
from tqdm import tqdm
from bert_score import BERTScorer
from transformers import AutoTokenizer
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    STAGE5_END_DIR, DATA_HUMAN_DATASET,
    BGE_M3_PATH, BERT_SCORE_BATCH_SIZE
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Directories to evaluate — add or replace with other model output dirs as needed
TARGET_DIRS = [
    STAGE5_END_DIR,
]

HUM_DIR = DATA_HUMAN_DATASET
MODEL_PATH = BGE_M3_PATH
MAX_TOKENS = 8192
BATCH_SIZE = BERT_SCORE_BATCH_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"[*] GPU detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[*] TF32 acceleration enabled.\n")
    torch.cuda.empty_cache()


def truncate_to_max_tokens(text, tokenizer, max_tokens):
    if not text:
        return ""
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    print("[*] Loading BGE-M3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.model_max_length = MAX_TOKENS

    print("[*] Initializing BERTScorer (loading BAAI/bge-m3 into GPU memory)...")
    scorer = BERTScorer(
        model_type=MODEL_PATH,
        num_layers=24,
        device=device,
        nthreads=8,
        use_fast_tokenizer=False
    )
    print("[*] Model loaded.\n")
    print("=" * 60)

    print("[*] Scanning and indexing human reference dataset...")
    hum_dict = {}
    if os.path.exists(HUM_DIR):
        for h_file in os.listdir(HUM_DIR):
            if h_file.endswith(".json") and os.path.isfile(os.path.join(HUM_DIR, h_file)):
                h_id = re.sub(r'\D', '', h_file)
                if h_id:
                    hum_dict[h_id] = os.path.join(HUM_DIR, h_file)
        print(f"[*] Indexed {len(hum_dict)} human reference files.\n")
    else:
        print(f"[!] Human dataset path not found: {HUM_DIR}")
        return

    total_dirs = len(TARGET_DIRS)
    for idx, gen_dir in enumerate(TARGET_DIRS, 1):
        print(f"\n[{idx}/{total_dirs}] Processing directory: {gen_dir}")

        if not os.path.exists(gen_dir):
            print("[!] Warning: directory not found, skipping.")
            continue

        out_csv = os.path.join(gen_dir, "bge_m3_bertscore_results.csv")

        app_ids = []
        gen_texts = []
        hum_texts = []

        gen_files = [
            f for f in os.listdir(gen_dir)
            if f.endswith(".txt") and os.path.isfile(os.path.join(gen_dir, f))
        ]

        if not gen_files:
            print("[!] No .txt files found in this directory, skipping.")
            continue

        for filename in gen_files:
            gen_id = re.sub(r'\D', '', filename)
            if not gen_id or gen_id not in hum_dict:
                continue

            with open(os.path.join(gen_dir, filename), 'r', encoding='utf-8') as f:
                gen_text = f.read().strip()

            with open(hum_dict[gen_id], 'r', encoding='utf-8') as f:
                try:
                    hum_data = json.load(f)
                    hum_claim = hum_data.get("claims", "")
                    if isinstance(hum_claim, list):
                        hum_claim = " ".join(hum_claim)
                    hum_claim = hum_claim.strip()
                except Exception:
                    continue

            if gen_text and hum_claim:
                app_ids.append(gen_id)
                gen_texts.append(gen_text)
                hum_texts.append(hum_claim)

        num_pairs = len(app_ids)
        if num_pairs == 0:
            print("[!] No paired data found, skipping directory.")
            continue

        print(f"[*] Paired {num_pairs} samples. Truncating long texts...")
        safe_gen_texts = [truncate_to_max_tokens(t, tokenizer, MAX_TOKENS) for t in gen_texts]
        safe_hum_texts = [truncate_to_max_tokens(t, tokenizer, MAX_TOKENS) for t in hum_texts]

        print("[*] Running semantic similarity inference...")
        P, R, F1 = scorer.score(safe_gen_texts, safe_hum_texts, verbose=False, batch_size=BATCH_SIZE)

        bert_p = P.tolist()
        bert_r = R.tolist()
        bert_f1 = F1.tolist()

        print(f"[*] Saving results to: {out_csv}")
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["App_ID", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1"])
            for i in range(num_pairs):
                writer.writerow([
                    app_ids[i],
                    round(bert_p[i], 4),
                    round(bert_r[i], 4),
                    round(bert_f1[i], 4)
                ])

        avg_f1 = sum(bert_f1) / len(bert_f1)
        print(f"[*] Average BERTScore F1: {avg_f1:.4f}")

        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[*] BERTScore evaluation complete for all directories.")
    print("=" * 60)


if __name__ == "__main__":
    main()
