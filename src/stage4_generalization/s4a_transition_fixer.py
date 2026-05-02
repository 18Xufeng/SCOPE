import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL,
    STAGE3C_DIR, STAGE4A_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE4A_DIR, exist_ok=True)

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

# Only update the transition word; do not re-classify the category
SYSTEM_PROMPT = """
You are an expert Patent Attorney. Your task is to select the most appropriate transition word for a patent claim, based on its subject matter category and preamble content.

### Transition Word Rules:

1. Apparatus/Device
- Preferred Transition: "comprising:"
- Logic: Open-ended transitions prevent competitors from avoiding infringement by simply adding an irrelevant component.

2. Method/Process
- Preferred Transition: "comprising the steps of:" or "comprising:"
- Secondary Transition (Rare): "consisting of:"
- Logic: Open-ended transitions prevent competitors from inserting an irrelevant step to avoid infringement.

3. System/Computer-Readable Medium
- Preferred Transition: "comprising:"
- Logic: Closed-ended transitions must NOT be used in software/system claims.

4. Composition of Matter
- Scenario A (Mixtures/Formulations like cosmetics or detergents): "comprising:"
- Scenario B (High purity materials / API / Alloys): "consisting of:" or "consisting essentially of:"

### [OUTPUT FORMAT (STRICT JSON ONLY)]
You must output ONLY a valid JSON object. Do not include any natural language explanation, markdown formatting outside the JSON, or conversational filler.
{
  "transition_word": "Fill in the selected transition word here (e.g., comprising:)"
}
"""

progress_lock = threading.Lock()
processed_claims_count = 0
total_claims_count = 0


def extract_json_from_response(response_text):
    text_without_think = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    json_match = re.search(r'\{.*\}', text_without_think, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def process_single_claim_in_place(claim_dict):
    global processed_claims_count, total_claims_count

    # Read the pre-determined category; do not re-classify
    category = claim_dict.get("subject_matter_category", "Apparatus")

    claim_subset = {
        "claim_id": claim_dict.get("claim_id"),
        "subject_matter_category": category,
        "preamble": claim_dict.get("preamble"),
        "transition": claim_dict.get("transition"),
        "elements": claim_dict.get("elements")
    }

    claim_json_str = json.dumps(claim_subset, indent=2, ensure_ascii=False)
    user_prompt = (
        f"The claim has already been classified as category: '{category}'.\n"
        f"Please select the most appropriate transition word for this claim based on the category and preamble.\n\n"
        f"Claim Data:\n{claim_json_str}"
    )

    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_PATH,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=256,
                timeout=LLM_REQUEST_TIMEOUT
            )

            raw_output = response.choices[0].message.content
            result = extract_json_from_response(raw_output)

            if result:
                # Only update transition, never overwrite subject_matter_category
                claim_dict["transition"] = result.get("transition_word", claim_dict.get("transition"))
            break

        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            print(f"\n[Warning] Claim ID {claim_dict.get('claim_id')} timed out (attempt {attempt}). Retrying in 5s...\nError: {e}")
            time.sleep(5)

        except Exception as e:
            print(f"\n[Error] Claim ID {claim_dict.get('claim_id')} fatal error: {e}")
            break

    with progress_lock:
        processed_claims_count += 1
        if processed_claims_count % 20 == 0 or processed_claims_count == total_claims_count:
            print(f"-> Progress: {processed_claims_count} / {total_claims_count} claims processed...")


def main():
    global total_claims_count

    json_files = [f for f in os.listdir(STAGE3C_DIR) if f.endswith('.json')]
    print(f"[*] Found {len(json_files)} JSON files. Reading data...")

    file_data_map = {}
    all_claim_tasks = []

    for filename in json_files:
        filepath = os.path.join(STAGE3C_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                patent_data = json.load(f)
                file_data_map[filename] = patent_data
                for claim in patent_data.get("structured_claims", []):
                    all_claim_tasks.append(claim)
            except Exception as e:
                print(f"[Warning] Failed to read {filename}: {e}")

    total_claims_count = len(all_claim_tasks)
    print(f"[*] Done. Extracted {total_claims_count} claims.")
    print(f"[*] Starting thread pool with MAX_WORKERS = {MAX_WORKERS}...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_claim_in_place, claim) for claim in all_claim_tasks]
        for _ in as_completed(futures):
            pass

    print("\n[*] All concurrent requests complete. Writing updated data to disk...")

    for filename, patent_data in file_data_map.items():
        out_filepath = os.path.join(STAGE4A_DIR, filename)
        with open(out_filepath, 'w', encoding='utf-8') as f:
            json.dump(patent_data, f, indent=2, ensure_ascii=False)

    print(f"[*] Transition word fix done. All files saved to:\n{STAGE4A_DIR}")


if __name__ == "__main__":
    main()
