import os
import json
import re
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL,
    STAGE4C_DIR, STAGE4D_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE4D_DIR, exist_ok=True)

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPTS = {
    "Apparatus": """
You are an expert Patent Attorney tasked with drafting a finalized, MAXIMALLY BROAD, and legally bulletproof Independent Claim for an Apparatus/Device.

### Input Definitions:
1. "preamble" & "transition": The introductory phrase and transition word.
2. "original_elements": The full list of original physical features.
3. "generalized_elements": The broadened versions of those features.

### [CRITICAL RULE 1: THE OCCAM'S RAZOR (MINIMALIST CLAIMING)]
- You MUST perform a "Triage". Look at the `generalized_elements` and select ONLY the 3 to 5 absolute essential core structural components required for the invention to stand.
- You MUST STRIP AWAY all secondary features (optional layers, specific materials, exact dimensions, protective coatings, fasteners, aesthetic parts). Leave them out completely so they can be used later as dependent claims.

### [CRITICAL RULE 2: STATUTORY PURITY]
- You are drafting an APPARATUS. You are STRICTLY FORBIDDEN from including any METHOD steps (e.g., "washing", "heating", "manufacturing", "attaching"). If a generalized element describes a method, drop it or convert it to a static structural capability.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Triage]: Identify the absolute minimum core skeleton. List the items to KEEP.
- Step 2 [Delegation]: Explicitly list the secondary items to DROP from this independent claim.
- Step 3 [Drafting]: Assemble ONLY the kept items. Verify strict antecedent basis (A/The).

### Output Format:
Output ONLY a valid JSON object:
{
  "drafted_claim": "The maximally broad, minimalist independent claim text here."
}
""",

    "Method": """
You are an expert Patent Attorney tasked with drafting a finalized, MAXIMALLY BROAD, and legally bulletproof Independent Claim for a Method/Process.

### Input Definitions:
1. "preamble" & "transition": The introductory phrase and transition word.
2. "original_elements": The full list of original steps.
3. "generalized_elements": The broadened versions of those steps.

### [CRITICAL RULE 1: THE OCCAM'S RAZOR (MINIMALIST CLAIMING)]
- You MUST select ONLY the absolute core transformative or chronological steps (usually 3 to 5 steps) necessary to achieve the final result.
- You MUST STRIP AWAY all preparatory steps, post-processing steps, optional refinements, and specific operational conditions.

### [CRITICAL RULE 2: THE GERUND MANDATE & ANTECEDENT BASIS]
- Every single step you choose to keep MUST begin with a gerund (-ing verb).
- Ensure strict logical flow.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Triage]: Identify the core transformative steps. List the items to KEEP.
- Step 2 [Delegation]: List the preparatory, optional, or highly specific steps to DROP.
- Step 3 [Drafting]: Assemble ONLY the kept steps. Verify gerunds and antecedent basis.

### Output Format:
{
  "drafted_claim": "The maximally broad, minimalist independent claim text here."
}
""",

    "System": """
You are an expert Patent Attorney tasked with drafting a finalized, MAXIMALLY BROAD Independent Claim for a System or CRM.

### Input Definitions:
1. "preamble" & "transition": The introductory phrase and transition word.
2. "original_elements": The full list of hardware/software features.
3. "generalized_elements": The broadened versions of those features.

### [CRITICAL RULE 1: THE OCCAM'S RAZOR (MINIMALIST CLAIMING)]
- Select ONLY the essential hardware nodes (e.g., "a processor", "a memory") and the absolute core logical data flow.
- STRIP AWAY specific network protocols, secondary user interfaces, optional data filtering steps, and specific database structures.

### [CRITICAL RULE 2: ALICE COMPLIANCE (101 ELIGIBILITY)]
- Ensure the abstract logic/software steps are strictly tied to the physical hardware nodes you kept.
- Do not mix physical user actions into the system claims.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Triage]: Identify the core hardware nodes and main data flow.
- Step 2 [Delegation]: List the secondary modules/protocols to DROP.
- Step 3 [Drafting]: Assemble the minimalist system. Verify antecedent basis.

### Output Format:
{
  "drafted_claim": "The maximally broad, minimalist independent claim text here."
}
""",

    "Composition": """
You are an expert Patent Attorney tasked with drafting a finalized, MAXIMALLY BROAD Independent Claim for a Composition of Matter.

### Input Definitions:
1. "preamble" & "transition": The introductory phrase and transition word.
2. "original_elements": The full list of chemical/material features.
3. "generalized_elements": The broadened versions of those features.

### [CRITICAL RULE 1: THE OCCAM'S RAZOR (MINIMALIST CLAIMING)]
- Select ONLY the absolute base ingredients or core active centers (usually just 2 to 4 components).
- STRIP AWAY all specific ratios, exact formulas, optional additives, and specific micro-properties.

### [CRITICAL RULE 2: STATUTORY PURITY]
- You are drafting a COMPOSITION. You are STRICTLY FORBIDDEN from including any METHOD steps (e.g., "mixing", "washing", "drying", "applying to a surface"). If an element describes how the composition is made or used, DROP IT COMPLETELY.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Triage]: Identify the fundamental base ingredients/matrix.
- Step 2 [Delegation]: List the additives, properties, and all method steps to DROP.
- Step 3 [Drafting]: Assemble the minimalist composition using broad chemical genus terms.

### Output Format:
{
  "drafted_claim": "The maximally broad, minimalist independent claim text here."
}
"""
}

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["Apparatus"]

CATEGORY_MAP = {
    "apparatus": "Apparatus", "device": "Apparatus",
    "method": "Method", "process": "Method",
    "system": "System", "computer": "System",
    "composition": "Composition", "chemical": "Composition"
}


def resolve_category(raw_category):
    if not raw_category:
        return "Apparatus"
    lower = raw_category.lower()
    for key, val in CATEGORY_MAP.items():
        if key in lower:
            return val
    return "Apparatus"


def extract_json_from_response(response_text):
    text_without_think = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    json_match = re.search(r'\{.*\}', text_without_think, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def process_single_claim_drafting(claim_dict):
    category = resolve_category(claim_dict.get("subject_matter_category", "Apparatus"))
    system_prompt = SYSTEM_PROMPTS.get(category, DEFAULT_SYSTEM_PROMPT)

    preamble = claim_dict.get("preamble", "")
    transition = claim_dict.get("transition", "")
    original_elements = claim_dict.get("elements", [])
    generalized_elements = claim_dict.get("generalized_elements", [])

    user_prompt = f"""Here is your input context for drafting the final claim:

[Preamble]: {preamble}
[Transition]: {transition}

[Original Elements (Your Strict Blueprint)]:
{json.dumps(original_elements, indent=2)}

[Generalized Elements (Your Broadened Content)]:
{json.dumps(generalized_elements, indent=2)}

Please execute the Chain of Thought process in <think> tags, and then draft the final claim. Output ONLY the raw JSON.
"""

    while True:
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_PATH,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=8192,
                timeout=LLM_REQUEST_TIMEOUT
            )

            raw_output = response.choices[0].message.content
            result_json = extract_json_from_response(raw_output)

            if result_json and "drafted_claim" in result_json:
                claim_dict["drafted_claim"] = result_json["drafted_claim"]
                break
            else:
                time.sleep(3)
                continue

        except Exception:
            time.sleep(3)
            continue


def run_stage4d():
    json_files = glob.glob(os.path.join(STAGE4C_DIR, "*.json"))
    print(f"[*] Found {len(json_files)} files. Extracting all claim tasks...")

    file_data_map = {}
    all_claim_tasks = []

    for filepath in json_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            patent_data = json.load(f)
            file_data_map[filename] = patent_data
            for claim in patent_data.get("structured_claims", []):
                all_claim_tasks.append(claim)

    total_claims = len(all_claim_tasks)
    print(f"[*] Extraction complete. {total_claims} claims queued for minimalist drafting.")
    print(f"[*] Starting with MAX_WORKERS = {MAX_WORKERS}...")

    with tqdm(total=total_claims, desc="Claim reconstruction", colour="blue") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_claim_drafting, claim) for claim in all_claim_tasks]
            for _ in as_completed(futures):
                pbar.update(1)

    print("\n[*] Global reconstruction complete. Writing final results to disk...")

    for filename, patent_data in file_data_map.items():
        out_filepath = os.path.join(STAGE4D_DIR, filename)
        with open(out_filepath, 'w', encoding='utf-8') as f:
            json.dump(patent_data, f, indent=2, ensure_ascii=False)

    print(f"[*] Minimalist drafting complete. Final claims saved to:\n{STAGE4D_DIR}")


if __name__ == "__main__":
    run_stage4d()
