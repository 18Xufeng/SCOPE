import os
import json
import glob
import re
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import copy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL,
    STAGE2_DIR, STAGE3B_DIR, STAGE3C_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE3C_DIR, exist_ok=True)
os.makedirs(os.path.join(STAGE3C_DIR, "surgery_logs"), exist_ok=True)

EVIDENCE_REGISTRY_PATH = os.path.join(STAGE3B_DIR, "global_evidence_registry.json")
LOGS_DIR = os.path.join(STAGE3C_DIR, "surgery_logs")

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

PROMPT_GATEKEEPER_SYSTEM = """You are an Expert Patent Attorney and Structural Redundancy Analyzer. Your task is to compare a [TARGET CLAIM] against a [BASE CLAIM] (and supporting [EVIDENCE DOSSIER]) to determine if there is enough "Structural Textual Overlap" to warrant an element-by-element surgery.

### [CRITICAL DEFINITION OF REDUNDANCY]
Your goal is to detect textual and structural redundancy, NOT to judge the overall novelty of the Target Claim.

1. **The "Attention Hijacking" Rule (Systems/Apparatus):** If the Target Claim incorporates the exact same core foundational layers/structures of the Base Claim, but adds NEW system-level components, the structural overlap is STILL REDUNDANT. DO NOT output "NO" just because the Target Claim has new functional additions. You MUST output "YES" so we can surgically remove the repetitive structural text and convert it into a clean dependent claim.
2. **The Apparatus vs. Method Rule:** If the Base Claim is an Apparatus/Structure, and the Target Claim is a "Method of making" or "Method of using", look closely at the method steps. If the first several steps simply recount the manufacturing, mixing, or assembly of the exact same structure/composition described in the Base Claim, this IS REDUNDANT.

### [DECISION LOGIC]
- Output "YES" if the Target Claim spends multiple elements re-describing features, layers, or ingredients already protected by the Base Claim.
- Output "NO" ONLY IF the Target Claim is completely independent and shares almost no structural or compositional text with the Base Claim.

### [OUTPUT FORMAT (STRICT JSON ONLY)]
{
  "decision": "YES" | "NO",
  "justification": "<Briefly explain the structural overlap or lack thereof>"
}"""

PROMPT_ELEMENT_WORKER_SYSTEM = """You are an Expert Patent Surgeon. Your task is to evaluate a single element from a Target Claim and determine if it should be DELETED, KEPT, or MODIFIED based on its redundancy relative to the [BASE CLAIM].

### [CRITICAL RULES OF SURGERY]

**RULE 1: DO NOT OVER-DELETE (Strict KEEP)**
Does this element introduce a NEW physical component, a NEW material, or a NEW distinct physical property that is NOT explicitly mentioned in the Base Claim?
- If YES, you MUST action: "KEEP". Do not delete unique properties or new materials simply because they seem logically related or adjacent to the Base Claim.

**RULE 2: SEVERING TIES IS FORBIDDEN (The MODIFY Imperative)**
When an element contains BOTH redundant information AND a crucial spatial/relational context, YOU MUST NOT USE "DELETE".
- Action must be: "MODIFY".
- You must strip away the redundant descriptive details but PERFECTLY PRESERVE the connecting/locational phrase, replacing the redundant part with a reference to the Base Claim.
- *Example Original:* "an absorbent powder mixture disposed within the sheath, the powder mixture comprising 15-25% Typha latifolia and 5-10% Avena sativa;"
- *Example Modified:* "a powder mixture according to claim 1 disposed within the sheath;"

**RULE 3: SAFE DELETION (Strict DELETE)**
If the element is 100% textually and structurally covered by the Base Claim, and it DOES NOT contain any unique locational/relational context that connects it to new components, action: "DELETE".

### [OUTPUT FORMAT (STRICT JSON ONLY)]
{
  "action": "KEEP" | "DELETE" | "MODIFY",
  "reason": "<Explain your choice based strictly on the rules above>",
  "modified_text": "<Provide the modified text here ONLY if action is MODIFY, otherwise put original text>"
}"""


def extract_json_from_text(response_text):
    if not response_text or "[LLM Error]" in response_text:
        return None
    text_without_think = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
    text_without_think = re.sub(r'```json\s*', '', text_without_think)
    text_without_think = re.sub(r'```\s*', '', text_without_think)
    match = re.search(r'(\{[\s\S]*\})', text_without_think)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            tqdm.write(f" [JSON Decode Error]: {e}")
    return None


def extract_think_log(response_text):
    if not response_text:
        return "No response"
    match = re.search(r'<think>([\s\S]*?)</think>', response_text)
    return match.group(1).strip() if match else "No internal monologue found."


def call_llm(system_prompt, user_prompt, temperature=0.1, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_PATH,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=8192,
                timeout=LLM_REQUEST_TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                tqdm.write(f"  -> [Retry {attempt + 1}] Request failed ({error_msg}), retrying...")
                time.sleep(3)
            else:
                return f"[LLM Error] API calling failed after {max_retries} attempts: {error_msg}"


def normalize_claim_punctuation(elements):
    if not elements:
        return elements
    cleaned_elements = []
    for i, el in enumerate(elements):
        clean_text = re.sub(r'[\s;,\.]+(and)?[\s;,\.]*$', '', el.strip(), flags=re.IGNORECASE)
        if i == len(elements) - 1:
            cleaned_elements.append(clean_text + ".")
        elif i == len(elements) - 2:
            cleaned_elements.append(clean_text + "; and")
        else:
            cleaned_elements.append(clean_text + ";")
    return cleaned_elements


def strip_context_prefix(mod_text, preamble, transition):
    mod_text = mod_text.strip()
    combo_prefix = f"{preamble} {transition}".strip().lower()
    while True:
        original_len = len(mod_text)
        if mod_text.lower().startswith(combo_prefix):
            mod_text = mod_text[len(combo_prefix):].strip()
        if mod_text.lower().startswith(preamble.lower()):
            mod_text = mod_text[len(preamble):].strip()
        if mod_text.lower().startswith(transition.lower()):
            mod_text = mod_text[len(transition):].strip()
        if len(mod_text) == original_len:
            break
    return mod_text


def process_patent(patent_id, suspects_list):
    clean_id_match = re.search(r'\d+', patent_id)
    if not clean_id_match:
        return False, patent_id, "Invalid ID"
    clean_id = clean_id_match.group(0)

    original_files = glob.glob(os.path.join(STAGE2_DIR, f"*{clean_id}*.json"))
    if not original_files:
        return False, patent_id, "File not found"

    with open(original_files[0], 'r', encoding='utf-8') as f:
        patent_data = json.load(f)

    claims = patent_data.get("structured_claims", [])
    modified_claims_map = {c["claim_id"]: copy.deepcopy(c) for c in claims}

    log_buffer = [f"========== SURGERY LOG FOR PATENT {clean_id} ==========\n"]

    for suspect in suspects_list:
        t_id = suspect["target_claim_id"]
        b_id = suspect["cites_base_claim_id"]

        target_claim = modified_claims_map.get(t_id)
        base_claim = modified_claims_map.get(b_id)
        if not target_claim or not base_claim:
            continue

        log_buffer.append(f"\n{'#' * 50}")
        log_buffer.append(f"## ANALYZING Target Claim [{t_id}] vs Base Claim [{b_id}]")
        log_buffer.append(f"{'#' * 50}\n")

        evidence_text = "\n".join([f"- {ev['text']}" for ev in suspect['evidence']])
        r1_user_prompt = f"### [BASE CLAIM]\n{json.dumps(base_claim, indent=2)}\n\n"
        r1_user_prompt += f"### [TARGET CLAIM]\n{json.dumps(target_claim, indent=2)}\n\n"
        r1_user_prompt += f"### [EVIDENCE DOSSIER]\n{evidence_text}\n"

        log_buffer.append(">>> [PHASE 1: GATEKEEPER CHECK] <<<")
        r1_response = call_llm(PROMPT_GATEKEEPER_SYSTEM, r1_user_prompt)
        log_buffer.append("\n[DeepSeek Thinking]:\n" + extract_think_log(r1_response))

        r1_json = extract_json_from_text(r1_response)
        if not r1_json:
            log_buffer.append("\n[ERROR]: Failed to parse Gatekeeper JSON. Skipping.")
            continue

        decision = r1_json.get("decision", "")
        log_buffer.append(f"\n[DECISION]: {decision}")
        log_buffer.append(f"[JUSTIFICATION]: {r1_json.get('justification', '')}\n")

        if decision != "YES":
            log_buffer.append("-> Action: SKIPPED. Target Claim is either Unique or a Hallucination.\n")
            continue

        log_buffer.append("-> Action: REDUNDANCY CONFIRMED. Commencing Element-by-Element Surgery.\n")
        log_buffer.append(">>> [PHASE 2: ELEMENT-BY-ELEMENT SURGERY] <<<\n")

        t_preamble = target_claim.get("preamble", "")
        t_transition = target_claim.get("transition", "")
        t_elements = target_claim.get("elements", [])
        new_elements = []

        for i, original_element in enumerate(t_elements):
            log_buffer.append(f"--- Element [{i}] ---")
            log_buffer.append(f"[Original Text]: {original_element}")

            contextual_sentence = f"{t_preamble} {t_transition} {original_element}"
            r2_user_prompt = f"### [BASE CLAIM]\n{json.dumps(base_claim, indent=2)}\n\n"
            r2_user_prompt += f"### [EVIDENCE DOSSIER]\n{evidence_text}\n\n"
            r2_user_prompt += f"### [CURRENT ELEMENT TO EVALUATE]\nOriginal Index: {i}\nContextualized Text: \"{contextual_sentence}\"\nOriginal Raw Element: \"{original_element}\"\n"

            r2_response = call_llm(PROMPT_ELEMENT_WORKER_SYSTEM, r2_user_prompt)
            log_buffer.append(f"[Thinking]:\n" + extract_think_log(r2_response))

            r2_json = extract_json_from_text(r2_response)
            if not r2_json:
                new_elements.append(original_element)
                log_buffer.append("[Action]: KEEP (Fallback due to parse error)\n")
                continue

            action = r2_json.get("action", "KEEP")
            log_buffer.append(f"[Action]: {action}")
            log_buffer.append(f"[Reason]: {r2_json.get('reason', '')}")

            if action == "DELETE":
                log_buffer.append("-> Element successfully deleted.\n")
                continue
            elif action == "MODIFY":
                mod_text = r2_json.get("modified_text", original_element)
                clean_mod_text = strip_context_prefix(mod_text, t_preamble, t_transition)
                new_elements.append(clean_mod_text)
                log_buffer.append(f"[Modified Text (Raw)]: {mod_text}")
                log_buffer.append(f"[Modified Text (Cleaned)]: {clean_mod_text}\n")
            else:
                new_elements.append(original_element)
                log_buffer.append("-> Element kept intact.\n")

        if new_elements:
            b_preamble = base_claim.get("preamble", "device")
            clean_preamble = re.sub(r'^(a|an|the)\s+', '', b_preamble.lower(), flags=re.IGNORECASE)
            cross_reference = f"a {clean_preamble} according to claim {b_id}"
            new_elements.insert(0, cross_reference)
            final_elements = normalize_claim_punctuation(new_elements)

            target_claim["elements"] = final_elements
            # Always keep claim_type as 'independent'
            modified_claims_map[t_id] = target_claim

            log_buffer.append(">>> [PHASE 3: POST-PROCESSING] <<<")
            log_buffer.append(f"Inserted Cross-Reference: {cross_reference}")
            log_buffer.append("Punctuation normalization applied.\n")

    log_path = os.path.join(LOGS_DIR, f"log_{clean_id}.txt")
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write("\n".join(log_buffer))

    final_output = {
        "application_number": patent_data.get("application_number", clean_id),
        "structured_claims": list(modified_claims_map.values())
    }

    out_path = os.path.join(STAGE3C_DIR, f"purified_final_{clean_id}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    return True, clean_id, "Success"


def run_stage3c():
    print(f"[*] Starting element-by-element surgery with audit logging...")
    with open(EVIDENCE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {executor.submit(process_patent, p_id, suspects): p_id for p_id, suspects in registry.items()}
        with tqdm(total=len(registry), desc="Claim surgery", colour="green") as pbar:
            for future in as_completed(future_to_pid):
                pbar.update(1)

    print(f"\n[*] Surgery complete.")
    print(f"[*] Audit logs saved to: {LOGS_DIR}")
    print(f"[*] Purified JSON saved to: {STAGE3C_DIR}")


if __name__ == "__main__":
    run_stage3c()
