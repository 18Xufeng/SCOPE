import os
import json
import glob
import tiktoken
import spacy
from tqdm import tqdm
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DATA_INPUT_DIR, CHUNK_DIR,
    CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_SENTENCES
)

os.makedirs(CHUNK_DIR, exist_ok=True)

print("[*] Initializing spaCy sentencizer...")
nlp = spacy.blank("xx")
nlp.add_pipe("sentencizer")
nlp.max_length = 40000000
print("[*] spaCy sentencizer ready.")

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(enc.encode(text))


def chunk_patent_description(text, target_tokens=CHUNK_TARGET_TOKENS, overlap_sentences=CHUNK_OVERLAP_SENTENCES):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk_sents = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if sent_tokens > target_tokens:
            if current_chunk_sents:
                chunks.append(" ".join(current_chunk_sents))
            chunks.append(sent)
            current_chunk_sents = [sent]
            current_tokens = sent_tokens
            continue

        if current_tokens + sent_tokens > target_tokens and current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            overlap = current_chunk_sents[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk_sents = overlap + [sent]
            current_tokens = sum(count_tokens(s) for s in overlap) + sent_tokens
        else:
            current_chunk_sents.append(sent)
            current_tokens += sent_tokens

    if current_chunk_sents:
        final_chunk_text = " ".join(current_chunk_sents)
        if not chunks or final_chunk_text != chunks[-1]:
            chunks.append(final_chunk_text)

    return chunks


def main():
    input_files = glob.glob(os.path.join(DATA_INPUT_DIR, "*.json"))
    if not input_files:
        print(f"[Error] No JSON files found in {DATA_INPUT_DIR}")
        return

    print(f"[*] Found {len(input_files)} patent files. Starting chunking...")

    stats_data = []
    total_chunks = 0
    failed_files = 0

    for filepath in tqdm(input_files, desc="Chunking progress", colour="green"):
        filename = os.path.basename(filepath)
        patent_id = os.path.splitext(filename)[0]

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            full_description = data.get("full_description", "")
            if not full_description.strip():
                failed_files += 1
                stats_data.append({"patent_id": patent_id, "chunk_count": 0, "status": "No Description"})
                continue

            chunks = chunk_patent_description(full_description)
            chunk_count = len(chunks)
            total_chunks += chunk_count

            patent_output_dir = os.path.join(CHUNK_DIR, patent_id)
            os.makedirs(patent_output_dir, exist_ok=True)

            for idx, chunk_text in enumerate(chunks):
                chunk_data = {
                    "patent_id": patent_id,
                    "chunk_id": idx,
                    "text": chunk_text,
                    "token_estimate": count_tokens(chunk_text)
                }
                chunk_filepath = os.path.join(patent_output_dir, f"chunk_{idx}.json")
                with open(chunk_filepath, 'w', encoding='utf-8') as cf:
                    json.dump(chunk_data, cf, ensure_ascii=False, indent=2)

            stats_data.append({"patent_id": patent_id, "chunk_count": chunk_count, "status": "Success"})

        except Exception as e:
            failed_files += 1
            stats_data.append({"patent_id": patent_id, "chunk_count": 0, "status": f"Error: {str(e)}"})

    stats_csv_path = os.path.join(CHUNK_DIR, "chunk_statistics_report.csv")
    with open(stats_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["patent_id", "chunk_count", "status"])
        writer.writeheader()
        writer.writerows(stats_data)

    print("\n" + "=" * 40)
    print("[*] Chunking complete.")
    print(f"    Processed successfully : {len(input_files) - failed_files}")
    print(f"    Failed / no description: {failed_files}")
    print(f"    Total chunks generated : {total_chunks}")
    print(f"    Chunk data saved to    : {CHUNK_DIR}")
    print(f"    Stats report saved to  : {stats_csv_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
