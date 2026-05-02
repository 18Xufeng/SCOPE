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
    STAGE4A_DIR, STAGE4B_DIR, STAGE4C_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE4C_DIR, exist_ok=True)

EVIDENCE_REGISTRY_PATH = os.path.join(STAGE4B_DIR, "global_evidence_registry.json")
client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPTS = {
    "Apparatus": """
You are an expert Patent Attorney performing a point-wise analysis on a single element of an Apparatus/Device patent claim.
Your task is to analyze the given claim element, compare it with the provided specification evidence (RAG Chunks), and decide whether to DROP it or GENERALIZE it to maximize the patent's protective scope.

### Decision Rules:
1. ACTION: "DROP"
   - ONLY drop the element if it is a 100% pure reference (e.g., "An apparatus according to claim 1") or completely non-technical noise.

2. ACTION: "GENERALIZE"
   - [MANDATORY PARAMETER CLEANING]: You MUST remove ALL exact mathematical formulas, specific numerical ratios, and exact dimensions/metrics. Replace them with functional descriptors (e.g., "a predefined ratio", "a predetermined thickness configured to..."). NEVER leave exact numbers or pure math equations.
   - [PRESERVE CORE COMPONENTS]: NEVER drop an essential physical entity or operating medium. Even if it lacks complex modifiers, it must be generalized or retained, NOT dropped.
   - Generalize specific materials to their broader functional or structural class supported by the evidence (e.g., "an aluminum casing" -> "a rigid conductive housing").
   - Retain the noun-based structure typical of apparatus claims. Maintain strict structural and coupling relationships.

### Output Format:
You are a reasoning model. Think in <think> tags, but your final output MUST be a valid JSON object matching this schema exactly:
{
  "action": "DROP" or "GENERALIZE",
  "generalized_text": "The newly rewritten element (ONLY if action is GENERALIZE, otherwise leave empty)"
}
""",

    "Method": """
You are an expert Patent Attorney performing a point-wise analysis on a single element of a Method/Process patent claim.
Your task is to analyze the given claim step, compare it with the provided specification evidence (RAG Chunks), and decide whether to DROP it or GENERALIZE it to maximize the protective scope.

### Decision Rules:
1. ACTION: "DROP"
   - ONLY drop pure reference clauses or non-essential, purely manual administrative steps that lack technical character.

2. ACTION: "GENERALIZE"
   - [MANDATORY PARAMETER CLEANING]: You MUST generalize ALL specific operational parameters, including exact temperatures, precise times, concentrations, and particle sizes. Replace them with functional limits. NEVER leave exact numbers or ranges.
   - [PRESERVE CORE STEPS]: NEVER drop a fundamental transformative step or a core mechanism.
   - ALL generalized steps MUST strictly begin with a gerund (an -ing verb).
   - Broaden specific tools/machines used to perform the step to functional actors. Preserve conditional dependencies.

### Output Format:
{
  "action": "DROP" or "GENERALIZE",
  "generalized_text": "The newly rewritten gerund-led step (ONLY if action is GENERALIZE, otherwise leave empty)"
}
""",

    "System": """
You are an expert Patent Attorney performing a point-wise analysis on a single element of a System/Computer-Readable Medium patent claim.
Your task is to analyze the given claim element, compare it with the provided specification evidence (RAG Chunks), and decide whether to DROP it or GENERALIZE it.

### Decision Rules:
1. ACTION: "DROP"
   - ONLY drop pure reference clauses or pure mathematical formulas lacking hardware implementation.

2. ACTION: "GENERALIZE"
   - [MANDATORY PARAMETER CLEANING]: You MUST remove any exact numerical thresholds, specific bit-rates, exact memory sizes, or precise timing sequences. Replace them with functional/dynamic conditions.
   - [PRESERVE CORE NODES/LOGIC]: NEVER drop an essential hardware node, critical data structure, or core logical operation.
   - Generalize specific commercial hardware/software brands or highly specific network protocols.
   - Retain language that ties logic to hardware, such as "a processor configured to..." or "a memory storing instructions executable to...".

### Output Format:
{
  "action": "DROP" or "GENERALIZE",
  "generalized_text": "The newly rewritten element (ONLY if action is GENERALIZE, otherwise leave empty)"
}
""",

    "Composition": """
You are an expert Patent Attorney performing a point-wise analysis on a single element of a Composition of Matter patent claim.
Your task is to analyze the given claim element, compare it with the provided specification evidence (RAG Chunks), and decide whether to DROP it or GENERALIZE it.

### Decision Rules:
1. ACTION: "DROP"
   - ONLY drop pure reference clauses or purely aesthetic components IF AND ONLY IF they do not contribute to the core technical effect.

2. ACTION: "GENERALIZE"
   - [MANDATORY PARAMETER CLEANING]: You MUST NOT leave any exact weight/volume percentages, molar concentrations, precise pH values, or instrumental data peaks. Replace them entirely with functional descriptions. NEVER output exact numbers.
   - [PRESERVE CORE CHEMICAL/STRUCTURAL FEATURES]: NEVER drop essential chemical bonds, active centers, or base ingredients. You MUST generalize them to their chemical genus or functional class, but DO NOT delete them.
   - Generalize specific chemical species to their functional class or chemical genus IF supported by the evidence.
   - Maintain structural bonding or state descriptions.

### Output Format:
{
  "action": "DROP" or "GENERALIZE",
  "generalized_text": "The newly rewritten element (ONLY if action is GENERALIZE, otherwise leave empty)"
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


def build_evidence_text(evidence_list):
    if not evidence_list:
        return "No specific evidence chunks found for this claim."
    snippets = [f"[Rank {e.get('rank', '?')}]: {e.get('text', '')}" for e in evidence_list]
    return "\n".join(snippets)


def process_single_element_task(task_info):
    claim_dict = task_info["claim_dict"]
    element_idx = task_info["element_idx"]
    current_element_text = task_info["element_text"]
    context_text = task_info["context_text"]
    evidence_text = task_info["evidence_text"]
    category = task_info["category"]

    system_prompt = SYSTEM_PROMPTS.get(category, DEFAULT_SYSTEM_PROMPT)

    user_prompt = f"""Please analyze the following single claim element.

[Context: Preamble & Transition]
{context_text}

[Evidence: RAG Chunks from Specification]
{evidence_text}

[Current Claim Element to Analyze]
{current_element_text}

Based on your specific category rules, decide whether to DROP or GENERALIZE this element. Output ONLY the raw JSON.
"""

    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_PATH,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                timeout=LLM_REQUEST_TIMEOUT
            )

            raw_output = response.choices[0].message.content
            result_json = extract_json_from_response(raw_output)

            if result_json:
                action = result_json.get("action", "GENERALIZE").upper()
                gen_text = result_json.get("generalized_text", "")
                if action == "DROP":
                    claim_dict["processed_elements"][element_idx] = "__DROP__"
                else:
                    claim_dict["processed_elements"][element_idx] = gen_text if gen_text else current_element_text
            else:
                claim_dict["processed_elements"][element_idx] = current_element_text

            break

        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError):
            time.sleep(3)
        except Exception:
            claim_dict["processed_elements"][element_idx] = current_element_text
            break


def run_stage4c():
    print(f"[*] Loading global evidence registry: {EVIDENCE_REGISTRY_PATH}")
    evidence_map = {}
    if os.path.exists(EVIDENCE_REGISTRY_PATH):
        with open(EVIDENCE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
            for app_num, claim_results in registry_data.items():
                evidence_map[app_num] = {}
                for result in claim_results:
                    c_id = result.get("claim_id")
                    ev_text = build_evidence_text(result.get("evidence", []))
                    evidence_map[app_num][c_id] = ev_text

    json_files = glob.glob(os.path.join(STAGE4A_DIR, "*.json"))
    print(f"[*] Found {len(json_files)} files to process. Decomposing claim elements...")

    file_data_map = {}
    all_element_tasks = []

    for filepath in json_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            patent_data = json.load(f)
            file_data_map[filename] = patent_data

            app_num_raw = str(patent_data.get("application_number", ""))
            app_num_match = re.search(r'\d+', app_num_raw)
            clean_app_num = app_num_match.group(0) if app_num_match else app_num_raw

            for claim in patent_data.get("structured_claims", []):
                claim_id = claim.get("claim_id")
                category = resolve_category(claim.get("subject_matter_category", "Apparatus"))
                elements = claim.get("elements", [])

                claim["processed_elements"] = [None] * len(elements)
                context_text = f"{claim.get('preamble', '')} {claim.get('transition', '')}"
                evidence_text = evidence_map.get(clean_app_num, {}).get(claim_id, "No evidence.")

                for idx, element_text in enumerate(elements):
                    all_element_tasks.append({
                        "claim_dict": claim,
                        "element_idx": idx,
                        "element_text": element_text,
                        "context_text": context_text,
                        "evidence_text": evidence_text,
                        "category": category
                    })

    total_elements_count = len(all_element_tasks)
    print(f"[*] Decomposition complete. {total_elements_count} elements queued for LLM analysis.")

    with tqdm(total=total_elements_count, desc="Element generalization", colour="green") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_element_task, task) for task in all_element_tasks]
            for _ in as_completed(futures):
                pbar.update(1)

    print("\n[*] Global inference complete. Filtering dropped elements and assembling final files...")

    for filename, patent_data in file_data_map.items():
        for claim in patent_data.get("structured_claims", []):
            if "processed_elements" in claim:
                generalized_list = [e for e in claim["processed_elements"] if e != "__DROP__" and e is not None]
                claim["generalized_elements"] = generalized_list
                del claim["processed_elements"]

        out_filepath = os.path.join(STAGE4C_DIR, filename)
        with open(out_filepath, 'w', encoding='utf-8') as f:
            json.dump(patent_data, f, indent=2, ensure_ascii=False)

    print(f"[*] RAG-augmented generalization complete. Structured JSON saved to:\n{STAGE4C_DIR}")


if __name__ == "__main__":
    run_stage4c()
