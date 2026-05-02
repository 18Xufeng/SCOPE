import os
import json
import csv
import warnings
import re
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STAGE5_END_DIR, DATA_HUMAN_DATASET

warnings.filterwarnings("ignore")

GEN_DIR = STAGE5_END_DIR
HUM_DIR = DATA_HUMAN_DATASET
OUT_CSV = os.path.join(GEN_DIR, "evaluation_ngram_results.csv")


def main():
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
        print(f"[!] Error: human dataset directory not found: {HUM_DIR}")
        return

    if not os.path.exists(GEN_DIR):
        print(f"[!] Error: generated files directory not found: {GEN_DIR}")
        return

    gen_files = [
        f for f in os.listdir(GEN_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(GEN_DIR, f))
    ]

    app_ids = []
    gen_texts = []
    hum_texts = []

    print("[*] Pairing files...")
    for filename in tqdm(gen_files, desc="Reading and pairing", unit="file"):
        gen_id = re.sub(r'\D', '', filename)
        if not gen_id or gen_id not in hum_dict:
            continue

        with open(os.path.join(GEN_DIR, filename), 'r', encoding='utf-8') as f:
            gen_text = f.read().strip()

        with open(hum_dict[gen_id], 'r', encoding='utf-8') as f:
            try:
                hum_data = json.load(f)
                hum_claim = hum_data.get("claims", "")
                if isinstance(hum_claim, list):
                    hum_claim = " ".join(hum_claim)
                hum_claim = hum_claim.strip()
            except Exception as e:
                tqdm.write(f"[!] Failed to parse {hum_dict[gen_id]}: {e}")
                continue

        if gen_text and hum_claim:
            app_ids.append(gen_id)
            gen_texts.append(gen_text)
            hum_texts.append(hum_claim)

    num_pairs = len(app_ids)
    print(f"\n[*] Successfully paired {num_pairs} claim sets. Computing scores...\n")

    if num_pairs == 0:
        print("[!] No valid data pairs found. Exiting.")
        return

    bleu_scores = [0.0] * num_pairs
    rouge1_p, rouge1_r, rouge1_f1 = [0.0] * num_pairs, [0.0] * num_pairs, [0.0] * num_pairs
    rougeL_p, rougeL_r, rougeL_f1 = [0.0] * num_pairs, [0.0] * num_pairs, [0.0] * num_pairs

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    print("[*] Computing BLEU, ROUGE-1, and ROUGE-L...")

    for i, (gen, hum) in enumerate(tqdm(zip(gen_texts, hum_texts), total=num_pairs, desc="Computing metrics", unit="pair")):
        bleu_scores[i] = sacrebleu.sentence_bleu(gen, [hum]).score
        rouge_res = rouge.score(hum, gen)
        r1 = rouge_res['rouge1']
        rouge1_p[i], rouge1_r[i], rouge1_f1[i] = r1.precision, r1.recall, r1.fmeasure
        rL = rouge_res['rougeL']
        rougeL_p[i], rougeL_r[i], rougeL_f1[i] = rL.precision, rL.recall, rL.fmeasure

    print(f"\n[*] Saving results to {OUT_CSV}...")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "App_ID",
            "BLEU",
            "ROUGE_1_Precision", "ROUGE_1_Recall", "ROUGE_1_F1",
            "ROUGE_L_Precision", "ROUGE_L_Recall", "ROUGE_L_F1"
        ])
        for i in tqdm(range(num_pairs), desc="Writing CSV", unit="row"):
            writer.writerow([
                app_ids[i],
                round(bleu_scores[i], 4),
                round(rouge1_p[i], 4), round(rouge1_r[i], 4), round(rouge1_f1[i], 4),
                round(rougeL_p[i], 4), round(rougeL_r[i], 4), round(rougeL_f1[i], 4)
            ])

    avg_bleu = sum(bleu_scores) / num_pairs
    avg_r1_f1 = sum(rouge1_f1) / num_pairs
    avg_rL_f1 = sum(rougeL_f1) / num_pairs
    print(f"\n[*] Avg BLEU: {avg_bleu:.4f} | Avg ROUGE-1 F1: {avg_r1_f1:.4f} | Avg ROUGE-L F1: {avg_rL_f1:.4f}")
    print("[*] Evaluation complete. All metrics exported.")


if __name__ == "__main__":
    main()
