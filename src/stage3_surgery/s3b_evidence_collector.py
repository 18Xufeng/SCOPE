import os
import json
import glob
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    BGE_RERANKER_PATH, CHUNK_DIR,
    STAGE3A_DIR, STAGE3B_DIR,
    BGE_BATCH_SIZE
)

os.makedirs(STAGE3B_DIR, exist_ok=True)

REGISTRY_PATH = os.path.join(STAGE3A_DIR, "global_suspects_registry.json")

print("[*] Loading BGE-Reranker-v2-m3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BGE_RERANKER_PATH)

print("[*] Loading model onto GPU (fp16)...")
model = AutoModelForSequenceClassification.from_pretrained(
    BGE_RERANKER_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("[*] Model loaded.")


def get_dynamic_top_k(total_chunks):
    if total_chunks <= 4:
        return 2
    elif total_chunks <= 8:
        return 3
    else:
        return 4


def run_stage3b():
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    tasks = [(p_id, sus) for p_id, suspects in registry.items() for sus in suspects]
    print(f"[*] {len(tasks)} suspect pairs locked. Starting BGE reranker inference...")
    global_evidence_registry = {}

    for patent_id, suspect_item in tqdm(tasks, desc="Evidence retrieval", colour="green"):
        app_num_match = re.search(r'\d+', patent_id)
        if not app_num_match:
            continue
        clean_app_num = app_num_match.group(0)

        patent_chunk_dir = os.path.join(CHUNK_DIR, clean_app_num)
        if not os.path.exists(patent_chunk_dir):
            continue

        chunk_files = sorted(glob.glob(os.path.join(patent_chunk_dir, "*.json")))
        if not chunk_files:
            continue

        chunk_texts, chunk_ids = [], []
        for cf in chunk_files:
            with open(cf, 'r', encoding='utf-8') as f:
                c_data = json.load(f)
                chunk_texts.append(c_data["text"])
                chunk_ids.append(c_data["chunk_id"])

        query = " ".join(suspect_item.get("focus_area", []))
        if not query.strip():
            continue

        scores = []
        pairs = [[query, doc] for doc in chunk_texts]

        for i in range(0, len(pairs), BGE_BATCH_SIZE):
            batch_pairs = pairs[i:i + BGE_BATCH_SIZE]
            inputs = tokenizer(
                batch_pairs, padding=True, truncation=True,
                max_length=1024, return_tensors='pt'
            ).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                batch_scores = logits.squeeze(-1).float().cpu().numpy().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

        scored_results = sorted(
            [{"index": idx, "relevance_score": s} for idx, s in enumerate(scores)],
            key=lambda x: x["relevance_score"], reverse=True
        )

        actual_k = min(get_dynamic_top_k(len(chunk_texts)), len(scored_results))
        evidence_snippets = [{
            "chunk_id": chunk_ids[res["index"]],
            "text": chunk_texts[res["index"]],
            "rerank_score": res["relevance_score"],
            "rank": i + 1
        } for i, res in enumerate(scored_results[:actual_k])]

        result_obj = {
            "target_claim_id": suspect_item["target_claim_id"],
            "cites_base_claim_id": suspect_item["cites_base_claim_id"],
            "query_used": query,
            "total_chunks_scanned": len(chunk_texts),
            "evidence_retained": actual_k,
            "evidence": evidence_snippets
        }

        if patent_id not in global_evidence_registry:
            global_evidence_registry[patent_id] = []
        global_evidence_registry[patent_id].append(result_obj)

    output_file = os.path.join(STAGE3B_DIR, "global_evidence_registry.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(global_evidence_registry, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Evidence registry saved to: {output_file}")


if __name__ == "__main__":
    run_stage3b()
