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
    STAGE4A_DIR, STAGE4B_DIR,
    BGE_BATCH_SIZE
)

os.makedirs(STAGE4B_DIR, exist_ok=True)

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


def build_query_from_claim(claim):
    preamble = claim.get("preamble", "").strip()
    transition = claim.get("transition", "").strip()
    elements_text = " ".join([elem.strip() for elem in claim.get("elements", [])])
    query = f"{preamble} {transition} {elements_text}"
    return re.sub(r'\s+', ' ', query).strip()


def run_stage4b():
    claim_files = glob.glob(os.path.join(STAGE4A_DIR, "*.json"))
    print(f"[*] Found {len(claim_files)} patent JSON files in input directory.")

    tasks = []
    for filepath in claim_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                app_num = data.get("application_number", "")
                claims = data.get("structured_claims", [])
                app_num_match = re.search(r'\d+', str(app_num))
                clean_app_num = app_num_match.group(0) if app_num_match else str(app_num)
                for claim in claims:
                    tasks.append({
                        "application_number": clean_app_num,
                        "original_app_num": app_num,
                        "claim_data": claim
                    })
            except Exception as e:
                print(f"[Warning] Failed to parse {filepath}: {e}")

    print(f"[*] Extracted {len(tasks)} claims. Starting BGE inference...")
    global_evidence_registry = {}

    for task in tqdm(tasks, desc="Evidence retrieval", colour="green"):
        app_num = task["application_number"]
        claim = task["claim_data"]
        claim_id = claim.get("claim_id")

        patent_chunk_dir = os.path.join(CHUNK_DIR, app_num)
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

        query = build_query_from_claim(claim)
        if not query:
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
            "claim_id": claim_id,
            "claim_type": claim.get("claim_type"),
            "subject_matter_category": claim.get("subject_matter_category"),
            "query_used": query,
            "total_chunks_scanned": len(chunk_texts),
            "evidence_retained": actual_k,
            "evidence": evidence_snippets
        }

        original_app_num = task["original_app_num"]
        if original_app_num not in global_evidence_registry:
            global_evidence_registry[original_app_num] = []
        global_evidence_registry[original_app_num].append(result_obj)

    output_file = os.path.join(STAGE4B_DIR, "global_evidence_registry.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(global_evidence_registry, f, ensure_ascii=False, indent=2)

    print(f"\n[*] Evidence retrieval complete. Results saved to: {output_file}")


if __name__ == "__main__":
    run_stage4b()
