import os
import json
import re
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL,
    STAGE4D_DIR, STAGE4B_DIR,
    STAGE5_OUTPUT_DIR, STAGE5_END_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE5_OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGE5_END_DIR, exist_ok=True)

RAG_REGISTRY_PATH = os.path.join(STAGE4B_DIR, "global_evidence_registry.json")

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPTS = {
    "Apparatus/Device": """
You are an expert Patent Attorney tasked with assembling a complete Claim Cluster (1 Independent Claim + multiple Dependent Claims) into a single continuous string.

### Dynamic Inputs Provided by System:
1. [STARTING_CLAIM_NUMBER]: The exact integer for the Independent Claim.
2. [PREVIOUSLY DRAFTED CLAIMS]: The exact text of all preceding claims (Your Vocabulary Ledger).
3. [INDEPENDENT_CLAIM_DRAFT]: The minimalist mother claim.
4. [DOWNSTREAM_FEATURES]: An array of specific physical features/parameters to be converted into Dependent Claims.
5. [SPECIFICATION CONTEXT (RAG)]: Excerpts from the patent specification to ground your logic.

### [CRITICAL DRAFTING RULES]:
1. [THE IMMUTABLE ROOT]: Your output MUST start with exactly: "[STARTING_CLAIM_NUMBER]. [INDEPENDENT_CLAIM_DRAFT]". DO NOT alter a single word of it.
2. [REALITY FILTER (Anti-Hallucination)]: Read the [SPECIFICATION CONTEXT]. You MUST completely DISCARD any items in [DOWNSTREAM_FEATURES] that describe manufacturing methods or human actions (e.g., "washing", "installing"). Only keep structural features.
3. [SELECTIVE AGGREGATION RULES (CRITICAL)]: DO NOT force all features into a few claims. Evaluate each feature in [DOWNSTREAM_FEATURES]. You must ONLY merge features if they trigger these specific rules; otherwise, keep them as STANDALONE dependent claims:
   - Rule A (Markush Merge): If multiple features list alternative materials/shapes for the EXACT SAME component, merge them into ONE claim ("wherein the [part] comprises a material selected from the group consisting of...").
   - Rule B (Sub-assembly Merge): If features describe inextricably linked sub-components that only function together as a single unit, merge them.
   - Rule C (Standalone Preservation): Distinct optional modules (e.g., adding a sensor vs. adding a battery) or distinct dimensional parameters MUST remain separate claims to provide granular fallback positions.
4. [TREE DEPENDENCY]: Avoid linear chaining unless absolutely necessary. Most dependent claims should BRANCH directly off the Independent Claim.
5. [STRICT ANTECEDENT BASIS (A vs. The)]: If a component already exists in the parent claim, use "the" or "said". ONLY use "a/an" when introducing a brand-new component.
6. [PARAMETER RESTORATION]: Retain exact numerical dimensions and specific materials.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Reality Check]: Filter out method steps based on RAG context.
- Step 2 [Triage & Merge Analysis]: Categorize each remaining feature:
  * Feature 1 -> Triggers Rule A (Merge with Feature 2)
  * Feature 3 -> Triggers Rule C (Keep Standalone)
- Step 3 [Dependency & Antecedent Check]: Assign the parent claim for each grouped or standalone feature. Check "A/The".
- Step 4 [Drafting]: Output the 1-sentence claims sequentially.

### Output Format:
Output ONLY a valid JSON object:
{
  "final_claim_cluster_text": "[STARTING_CLAIM_NUMBER]. [Indep Text] [STARTING_CLAIM_NUMBER+1]. The [Base Noun] according to claim [X]..."
}
""",
    "Method/Process": """
You are an expert Patent Attorney tasked with assembling a complete Method Claim Cluster.

### Dynamic Inputs Provided by System:
1. [STARTING_CLAIM_NUMBER]: The integer for the Independent Claim.
2. [PREVIOUSLY DRAFTED CLAIMS]: The Vocabulary Ledger.
3. [INDEPENDENT_CLAIM_DRAFT]: The minimalist mother claim.
4. [DOWNSTREAM_FEATURES]: Specific method steps or operational parameters.
5. [SPECIFICATION CONTEXT (RAG)]: Excerpts from the patent specification to ground chronological reality.

### [CRITICAL DRAFTING RULES]:
1. [THE IMMUTABLE ROOT]: Output exactly: "[STARTING_CLAIM_NUMBER]. [INDEPENDENT_CLAIM_DRAFT]".
2. [REALITY & TIME-SPACE FILTER]: Determine if the mother claim is a "Method of Making" or "Method of Using/Driving". If it is a "Method of Using", completely DISCARD preparatory manufacturing steps (e.g., "installing pixels") from [DOWNSTREAM_FEATURES].
3. [SELECTIVE AGGREGATION RULES (CRITICAL)]: DO NOT blindly merge everything. ONLY merge if they trigger these rules:
   - Rule A (Parameter Cluster): If multiple operational parameters (e.g., time, temperature, pressure, dilution rate) apply to the EXACT SAME method step, merge them into ONE dependent claim ("wherein the step of [X] is performed at a temperature of [Y] for a duration of [Z]").
   - Rule B (Markush Merge): Alternative reagents/tools for the same step.
   - Rule C (Standalone Preservation): Distinct chronological steps MUST remain separate dependent claims. Do NOT merge Step B and Step C into one claim.
4. [TREE DEPENDENCY]: Branch dependent claims off the independent claim, unless modifying a new step introduced in a specific dependent claim.
5. [STRICT ANTECEDENT BASIS & GERUNDS]: Use "the" for existing steps/workpieces. Every completely new step MUST begin with a gerund (-ing verb).
6. [PARAMETER RESTORATION]: Bring back exact temperatures, times, and concentrations.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Time-Space Check]: Purge chronologically impossible steps using RAG.
- Step 2 [Triage & Merge Analysis]: Map features to Rule A, Rule B, or Rule C (Standalone).
- Step 3 [Dependency Check]: Assign parent claims. Check antecedent basis.
- Step 4 [Drafting].

### Output Format:
Output ONLY a valid JSON object:
{
  "final_claim_cluster_text": "[STARTING_CLAIM_NUMBER]. A method... [STARTING_CLAIM_NUMBER+1]. The method according to claim [X]..."
}
""",
    "System/Computer-Readable Medium": """
You are an expert Patent Attorney tasked with assembling a complete System/CRM Claim Cluster.

### Dynamic Inputs Provided by System:
1. [STARTING_CLAIM_NUMBER]: The integer for the Independent Claim.
2. [PREVIOUSLY DRAFTED CLAIMS]: The Vocabulary Ledger.
3. [INDEPENDENT_CLAIM_DRAFT]: The minimalist mother claim.
4. [DOWNSTREAM_FEATURES]: Specific hardware, data structures, or protocols.
5. [SPECIFICATION CONTEXT (RAG)]: Excerpts from the patent specification.

### [CRITICAL DRAFTING RULES]:
1. [THE IMMUTABLE ROOT]: Output exactly: "[STARTING_CLAIM_NUMBER]. [INDEPENDENT_CLAIM_DRAFT]".
2. [REALITY FILTER]: Ensure all software/logic steps are strictly tied to hardware nodes. Discard physical human actions.
3. [SELECTIVE AGGREGATION RULES (CRITICAL)]: Evaluate features carefully.
   - Rule A (Markush/Alternatives): Alternative network protocols or data formats.
   - Rule B (Module Refinement): If multiple features define the specific internal algorithms of ONE single processor/module, merge them into one claim describing that processor.
   - Rule C (Standalone Preservation): Completely distinct hardware nodes (e.g., adding a remote server vs. adding a user mobile device) MUST remain separate claims to build independent fallbacks.
4. [TREE DEPENDENCY]: Avoid long linear chains. Branch off the main system claim.
5. [STRICT ANTECEDENT BASIS (A vs. The)]: Follow the A/The rule strictly.
6. [PARAMETER RESTORATION]: Retain exact network protocols, thresholds, and data sizes.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Topology Check]: Verify relationships using RAG.
- Step 2 [Triage & Merge Analysis]: Apply Rule A, B, or C to each feature.
- Step 3 [Dependency Check]: Ensure "the/said" rule is strictly followed.
- Step 4 [Drafting].

### Output Format:
Output ONLY a valid JSON object:
{
  "final_claim_cluster_text": "[STARTING_CLAIM_NUMBER]. A system... [STARTING_CLAIM_NUMBER+1]. The [Base Noun] according to claim [X]..."
}
""",
    "Composition of Matter": """
You are an expert Patent Attorney tasked with assembling a complete Composition Claim Cluster.

### Dynamic Inputs Provided by System:
1. [STARTING_CLAIM_NUMBER]: The integer for the Independent Claim.
2. [PREVIOUSLY DRAFTED CLAIMS]: The Vocabulary Ledger.
3. [INDEPENDENT_CLAIM_DRAFT]: The minimalist mother claim.
4. [DOWNSTREAM_FEATURES]: Specific chemicals, concentrations, or physical properties.
5. [SPECIFICATION CONTEXT (RAG)]: Excerpts from the patent specification.

### [CRITICAL DRAFTING RULES]:
1. [THE IMMUTABLE ROOT]: Output exactly: "[STARTING_CLAIM_NUMBER]. [INDEPENDENT_CLAIM_DRAFT]".
2. [REALITY FILTER (Statutory Purity)]: You are drafting a COMPOSITION. Completely DISCARD any method steps (e.g., "washing", "drying", "applying") from [DOWNSTREAM_FEATURES].
3. [SELECTIVE AGGREGATION RULES (CRITICAL)]: Do NOT blindly merge everything. ONLY merge if they trigger these rules:
   - Rule A (The Markush Merge): If there are multiple specific options for the EXACT SAME ingredient class (e.g., 8 different plants), merge them into ONE claim: "wherein the [ingredient] is selected from the group consisting of A, B, C, and D."
   - Rule B (The Parameter Cluster): Merge macroscopic physical properties of the ENTIRE mixture (e.g., pH, total moisture content, overall autoignition temp) into ONE comprehensive claim.
   - Rule C (Standalone Preservation): Distinct optional additives (e.g., adding a preservative vs. adding a colorant) or specific weight percentages of distinct base ingredients MUST remain separate claims. Do not merge a colorant limitation with a preservative limitation.
4. [TREE DEPENDENCY]: Do NOT chain claims. Branch them directly off the Independent Claim to form parallel lines of defense.
5. [STRICT ANTECEDENT BASIS]: Follow the A/The rule strictly.
6. [PARAMETER RESTORATION]: Bring back exact percentages, mmol/g, eV peaks, and chemical formulas.

### Chain of Thought (CoT) in <think> tags:
- Step 1 [Statutory Purge]: Delete all method/action steps.
- Step 2 [Triage & Merge Analysis]: Evaluate each feature against Rule A (Markush), Rule B (Parameters), or Rule C (Keep Standalone).
- Step 3 [Dependency & Antecedent Check]: Set parent claims. Check "the/said".
- Step 4 [Drafting].

### Output Format:
Output ONLY a valid JSON object:
{
  "final_claim_cluster_text": "[STARTING_CLAIM_NUMBER]. A [Composition]... [STARTING_CLAIM_NUMBER+1]. The [Composition] according to claim [X]..."
}
"""
}

CATEGORY_MAP = {
    "apparatus": "Apparatus/Device", "device": "Apparatus/Device",
    "method": "Method/Process", "process": "Method/Process",
    "system": "System/Computer-Readable Medium", "computer": "System/Computer-Readable Medium",
    "composition": "Composition of Matter", "chemical": "Composition of Matter"
}

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["Apparatus/Device"]


def resolve_category(raw_category):
    if not raw_category:
        return "Apparatus/Device"
    lower = raw_category.lower()
    for key, val in CATEGORY_MAP.items():
        if key in lower:
            return val
    return "Apparatus/Device"


def extract_json_from_response(response_text):
    if not response_text:
        return None
    text_without_think = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
    json_match = re.search(r'\{[\s\S]*\}', text_without_think)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def count_generated_claims(text):
    if not text:
        return 0
    pattern = r'(?:^|\s)(\d+)\.\s+[A-Z]'
    matches = re.findall(pattern, text)
    return len(matches)


def process_single_patent_file(filepath, global_evidence_map):
    with open(filepath, 'r', encoding='utf-8') as f:
        patent_data = json.load(f)

    claims = patent_data.get("structured_claims", [])
    if not claims:
        return patent_data, ""

    app_num_raw = str(patent_data.get("application_number", ""))
    app_num_match = re.search(r'\d+', app_num_raw)
    clean_app_num = app_num_match.group(0) if app_num_match else app_num_raw

    accumulated_claims_text = ""
    current_absolute_number = 1

    for claim_dict in claims:
        claim_id = claim_dict.get("claim_id")
        raw_category = claim_dict.get("subject_matter_category", "Apparatus/Device")
        category = resolve_category(raw_category)
        system_prompt = SYSTEM_PROMPTS.get(category, DEFAULT_SYSTEM_PROMPT)

        indep_claim_draft = claim_dict.get("drafted_claim", "")
        downstream_features = claim_dict.get("elements", [])
        if not downstream_features:
            downstream_features = claim_dict.get("generalized_elements", [])

        if not indep_claim_draft:
            continue

        evidence_text = global_evidence_map.get(clean_app_num, {}).get(
            claim_id, "No specific specification context found."
        )

        user_prompt = f"""Draft the next Claim Cluster.

### Dynamic Inputs:
[STARTING_CLAIM_NUMBER]: {current_absolute_number}

[PREVIOUSLY DRAFTED CLAIMS] (Your Vocabulary Ledger):
{accumulated_claims_text if accumulated_claims_text else "None. You are drafting the very first claim of this patent."}

[INDEPENDENT_CLAIM_DRAFT] (The Minimalist Root Claim):
{indep_claim_draft}

[DOWNSTREAM_FEATURES] (Features to expand into Dependent Claims):
{json.dumps(downstream_features, indent=2)}

[SPECIFICATION CONTEXT (RAG)]:
{evidence_text}

Execute the CoT process in <think> tags to triage the features, apply aggregation rules, and output the required JSON.
"""

        while True:
            try:
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL_PATH,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                    timeout=LLM_REQUEST_TIMEOUT
                )

                raw_output = response.choices[0].message.content
                result_json = extract_json_from_response(raw_output)

                if result_json and "final_claim_cluster_text" in result_json:
                    final_cluster_text = result_json["final_claim_cluster_text"].strip()

                    if not final_cluster_text.endswith('.'):
                        final_cluster_text += '.'

                    generated_count = count_generated_claims(final_cluster_text)

                    if generated_count == 0:
                        time.sleep(3)
                        continue

                    break
                else:
                    time.sleep(3)
                    continue

            except Exception:
                time.sleep(3)
                continue

        accumulated_claims_text += ("\n\n" if accumulated_claims_text else "") + final_cluster_text
        current_absolute_number += generated_count

        claim_dict["final_claim_cluster_text"] = final_cluster_text
        claim_dict["generated_claims_count"] = generated_count

    return patent_data, accumulated_claims_text


def run_step5_expansion():
    print(f"[*] Loading RAG global evidence registry: {RAG_REGISTRY_PATH}")
    global_evidence_map = {}
    if os.path.exists(RAG_REGISTRY_PATH):
        with open(RAG_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
            for app_num, claims_results in rag_data.items():
                global_evidence_map[app_num] = {}
                for res in claims_results:
                    c_id = res.get("claim_id")
                    snippets = [ev.get("text", "") for ev in res.get("evidence", [])]
                    global_evidence_map[app_num][c_id] = "\n".join(snippets)
    else:
        print("[!] Warning: RAG evidence registry not found. Running without specification context.")

    all_json_files = glob.glob(os.path.join(STAGE4D_DIR, "*.json"))
    pending_files = []

    print("[*] Checking for resume checkpoint...")
    for filepath in all_json_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        expected_json_path = os.path.join(STAGE5_OUTPUT_DIR, f"{base_name}.json")
        expected_txt_path = os.path.join(STAGE5_OUTPUT_DIR, f"{base_name}_FINAL_CLAIMS.txt")
        if os.path.exists(expected_json_path) and os.path.exists(expected_txt_path):
            continue
        pending_files.append(filepath)

    total_files = len(all_json_files)
    files_to_process = len(pending_files)
    skipped_files = total_files - files_to_process

    print(f"[*] Scan complete. Found {total_files} files.")
    print(f"[*] Checkpoint: skipped {skipped_files} already-processed files, {files_to_process} remaining.")

    if files_to_process == 0:
        print("\n[*] All tasks already completed. Proceeding to post-processing...")
    else:
        print(f"[*] Starting claim expansion pipeline with MAX_WORKERS = {MAX_WORKERS}...")
        with tqdm(total=files_to_process, desc="Claim expansion", colour="magenta") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(process_single_patent_file, fp, global_evidence_map): fp
                    for fp in pending_files
                }
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        final_patent_data, full_patent_text = future.result()
                        filename = os.path.basename(filepath)
                        base_name = os.path.splitext(filename)[0]

                        out_json_filepath = os.path.join(STAGE5_OUTPUT_DIR, f"{base_name}.json")
                        with open(out_json_filepath, 'w', encoding='utf-8') as f:
                            json.dump(final_patent_data, f, indent=2, ensure_ascii=False)

                        out_txt_filepath = os.path.join(STAGE5_OUTPUT_DIR, f"{base_name}_FINAL_CLAIMS.txt")
                        with open(out_txt_filepath, 'w', encoding='utf-8') as f:
                            f.write(full_patent_text)

                    except Exception as e:
                        tqdm.write(f"\n[!] Failed to process {os.path.basename(filepath)}: {e}")

                    pbar.update(1)

        print(f"\n[*] Claim tree expansion complete. JSON and TXT saved to:\n{STAGE5_OUTPUT_DIR}")


def run_postprocess_format():
    os.makedirs(STAGE5_END_DIR, exist_ok=True)
    file_pattern = os.path.join(STAGE5_OUTPUT_DIR, "final_*_FINAL_CLAIMS.txt")
    files_to_process = glob.glob(file_pattern)
    processed_count = 0
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        if not re.match(r"^final_\d+_FINAL_CLAIMS\.txt$", filename):
            continue
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        merged_content = re.sub(r'\s+', ' ', content).strip()
        output_filepath = os.path.join(STAGE5_END_DIR, filename)
        with open(output_filepath, 'w', encoding='utf-8') as file:
            file.write(merged_content)
        processed_count += 1
    print(f"[*] Post-processing complete. Merged {processed_count} files.")
    print(f"[*] Merged files saved to: {STAGE5_END_DIR}")


if __name__ == "__main__":
    run_step5_expansion()
    run_postprocess_format()
