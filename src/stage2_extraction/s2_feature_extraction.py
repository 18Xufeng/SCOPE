import os
import json
import glob
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL, LLM_REQUEST_TIMEOUT,
    DATA_INPUT_DIR, STAGE1_TWO_DIR, STAGE2_DIR,
    MAX_WORKERS
)

os.makedirs(STAGE2_DIR, exist_ok=True)
os.makedirs(os.path.join(STAGE2_DIR, "debug_logs"), exist_ok=True)

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

DEBUG_DIR = os.path.join(STAGE2_DIR, "debug_logs")

# ================= Expert System Prompts =================
PROMPTS = {
    "Apparatus": """You are an expert Patent Attorney specializing in "Apparatus/Device" claims. Your task is to extract an EXHAUSTIVE MASTER LIST of protectable features for the specified object.

### [STRICT EXTRACTION CONSTRAINTS]
1. ALL-INCLUSIVE PROTECTABLE FEATURES: Extract ALL structural components, core parts, and ALL valuable secondary features (specific materials, geometric shapes, numerical dimensions, tolerances). Leave no technical limitation behind.
2. STRICT BOUNDARY (NO CROSS-CONTAMINATION): Only extract features that belong strictly to the internal structure of THIS specific apparatus. Do not extract features of the larger system it might be part of, or the external environment it interacts with.
3. EXTREME RELATIONAL PRECISION: Define exactly how elements connect physically in 3D space. Use highly precise spatial and structural verbs (e.g., "disposed below", "having an upper major surface opposing...", "spaced apart by a groove", "extending through"). Do not merely list isolated parts.
4. FLAT STRUCTURE: Put ALL extracted features into the "elements" array of a single independent claim.

### [CHAIN OF THOUGHT REQUIREMENT]
Before outputting, include a "thought_process" object:
- "1_boundary_definition": Define the exact physical boundaries of this apparatus. What is IN scope and what is OUT of scope?
- "2_exhaustive_inventory": Sentence-by-sentence extraction of every component, material, and numerical parameter. DO NOT SUMMARIZE.
- "3_spatial_mapping": Explicitly state the 3D spatial/mechanical linkage for every item (what touches what, what is inside what).
- "4_exclusion_audit": State exactly what fluff or out-of-bounds features you intentionally rejected to prevent cross-contamination.

### [OUTPUT FORMAT (JSON ONLY)]
{
  "thought_process": {
    "1_boundary_definition": "...",
    "2_exhaustive_inventory": "...",
    "3_spatial_mapping": "...",
    "4_exclusion_audit": "..."
  },
  "claims": [
    {
      "claim_id": 1,
      "claim_type": "independent",
      "preamble": "A [Object Name]",
      "transition": "comprising:",
      "elements": [
        "a first metallic layer having an upper major surface and a lower major surface;",
        "a flexible dielectric layer disposed below the first metallic layer..."
      ]
    }
  ]
}""",

    "System": """You are an expert Patent Attorney specializing in "System" claims. Your task is to extract an EXHAUSTIVE MASTER LIST of protectable features for the specified system.

### [STRICT EXTRACTION CONSTRAINTS]
1. ENCAPSULATION PRINCIPLE (BLACK-BOXING): You MUST treat sub-components as black boxes. Describe what the sub-components are and how they connect to EACH OTHER at the system level. DO NOT describe the internal microscopic/structural details of the sub-components.
2. EXHAUSTIVE SYSTEM FEATURES: Extract all nodes, modules, specific communication protocols, power linkages, and overarching system configurations.
3. DATA FLOW & INTERACTION: You must explicitly define how modules interact (e.g., "a processor communicatively coupled to the memory to receive the signal", "sandwiched between").
4. FLAT STRUCTURE: Put ALL extracted features into the "elements" array of a single independent claim.

### [CHAIN OF THOUGHT REQUIREMENT]
Before outputting, include a "thought_process" object:
- "1_boundary_definition": List the top-level modules. Explicitly state which internal component details will be black-boxed.
- "2_exhaustive_inventory": List all system-level nodes, connections, and operating logic.
- "3_interaction_mapping": Describe the precise data flow, electrical signaling, or macro-physical stacking between the top-level modules.
- "4_exclusion_audit": State what internal component details you intentionally rejected to maintain the System-level perspective.

### [OUTPUT FORMAT (JSON ONLY)]
{
  "thought_process": {
    "1_boundary_definition": "...",
    "2_exhaustive_inventory": "...",
    "3_interaction_mapping": "...",
    "4_exclusion_audit": "..."
  },
  "claims": [
    {
      "claim_id": 1,
      "claim_type": "independent",
      "preamble": "A [Object Name]",
      "transition": "comprising:",
      "elements": [
        "an array substrate configured to output data;",
        "a color film substrate disposed opposite to the array substrate;",
        "liquid crystal molecules sandwiched between the array substrate and the color film substrate..."
      ]
    }
  ]
}""",

    "Method": """You are an expert Patent Attorney specializing in "Method/Process" claims. Your task is to extract an EXHAUSTIVE MASTER LIST of protectable features for the specified method.

### [STRICT EXTRACTION CONSTRAINTS]
1. EXHAUSTIVE PROCESS PARAMETERS: You MUST extract every single active step, AND every highly specific condition threshold (e.g., "temperature above 200°C", "total partial pressure up to 10 kPa", "for 7 to 12 hours", "concentration of 0.5%"). DO NOT generalize or summarize conditions.
2. SEPARATE REACTANTS FROM PRODUCTS: If this is a manufacturing method, focus on the STEPS and the CONDITIONS applied to the input materials. Do not merely describe the final structural properties of the output product.
3. CHRONOLOGICAL DEPENDENCY: Capture the exact sequence of operations.
4. FLAT STRUCTURE: Put ALL extracted features into the "elements" array of a single independent claim. Every element should start with a gerund (e.g., "heating", "maintaining").

### [CHAIN OF THOUGHT REQUIREMENT]
Before outputting, include a "thought_process" object:
- "1_boundary_definition": Define the input (starting materials/state) and the ultimate output of this method.
- "2_exhaustive_inventory": Sentence-by-sentence extraction of every action verb and its associated numerical condition/parameter. DO NOT SUMMARIZE.
- "3_chronological_mapping": Map the strict chronological order and conditional dependencies.
- "4_exclusion_audit": State what product-only structural features or non-technical fluff you rejected.

### [OUTPUT FORMAT (JSON ONLY)]
{
  "thought_process": {
    "1_boundary_definition": "...",
    "2_exhaustive_inventory": "...",
    "3_chronological_mapping": "...",
    "4_exclusion_audit": "..."
  },
  "claims": [
    {
      "claim_id": 1,
      "claim_type": "independent",
      "preamble": "A method for [Object Name]",
      "transition": "comprising the steps of:",
      "elements": [
        "subjecting a starting carbon material to preliminary heating in an inert gas medium at a temperature above 200° C;",
        "contacting the heated carbon material with fluoro-containing compounds..."
      ]
    }
  ]
}""",

    "Composition": """You are an expert Patent Attorney specializing in "Composition/Chemical" claims. Your task is to extract an EXHAUSTIVE MASTER LIST of protectable features for the specified composition.

### [STRICT EXTRACTION CONSTRAINTS]
1. EXHAUSTIVE INGREDIENTS & RANGES: Extract EVERY base ingredient, optional additive, exact weight/volume percentages, and specific numerical ranges (e.g., "between 15% and 25% by weight").
2. PRESERVE MARKUSH GROUPS & ALTERNATIVES: If options are listed, preserve them strictly (e.g., "selected from the group consisting of A, B, and C").
3. INTRINSIC PROPERTIES & PERFORMANCE: Extract highly specific performance metrics or targets defined in the text.
4. NO MANUFACTURING STEPS: Do not describe the method used to make the composition unless the text explicitly defines it as a product-by-process. Focus on the final chemical/structural makeup.
5. FLAT STRUCTURE: Put ALL extracted features into the "elements" array of a single independent claim.

### [CHAIN OF THOUGHT REQUIREMENT]
Before outputting, include a "thought_process" object:
- "1_boundary_definition": Clarify that extraction is strictly limited to the ingredients, chemical states, and intrinsic properties of the final composition.
- "2_exhaustive_inventory": List every ingredient, every specific percentage range, every alternative option, and every performance metric. DO NOT SUMMARIZE.
- "3_chemical_state_mapping": Describe the micro-structural or chemical state.
- "4_exclusion_audit": State exactly what manufacturing process steps or out-of-bounds system applications you intentionally rejected.

### [OUTPUT FORMAT (JSON ONLY)]
{
  "thought_process": {
    "1_boundary_definition": "...",
    "2_exhaustive_inventory": "...",
    "3_chemical_state_mapping": "...",
    "4_exclusion_audit": "..."
  },
  "claims": [
    {
      "claim_id": 1,
      "claim_type": "independent",
      "preamble": "A [Object Name]",
      "transition": "comprising:",
      "elements": [
        "Typha latifolia in an amount between 15% and 25% by weight of the dry powder mixture;",
        "a binder selected from the group consisting of a mixture of Opuntia spp mucilage and aqueous Linum usitatissimum seeds;",
        "wherein the powder mixture is configured to absorb 4.5 liters of crude oil"
      ]
    }
  ]
}"""
}

# ================= Helpers =================

def extract_json_from_text(response_text):
    if not response_text:
        return None
    text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None


def call_llm(system_prompt, user_prompt, max_retries=3):
    for attempt in range(max_retries):
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
            content = response.choices[0].message.content
            if not content:
                return "[Error] API returned empty content."
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return f"[Error] API call failed: {str(e)}"


def normalize_category(cat_str):
    cat_str = str(cat_str).lower()
    if "apparatus" in cat_str or "device" in cat_str:
        return "Apparatus"
    if "method" in cat_str or "process" in cat_str:
        return "Method"
    if "system" in cat_str:
        return "System"
    if "composition" in cat_str or "chemical" in cat_str:
        return "Composition"
    return "Apparatus"


# ================= Core Processing =================

def process_single_patent(phase1_file_path):
    filename = os.path.basename(phase1_file_path)
    raw_filename = filename.replace("step2_", "")
    raw_file_path = os.path.join(DATA_INPUT_DIR, raw_filename)
    debug_log_path = os.path.join(DEBUG_DIR, f"debug_{raw_filename}.txt")

    try:
        if not os.path.exists(raw_file_path):
            return f"[Error] {filename}: source file not found ({raw_filename})"

        with open(phase1_file_path, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
            core_objects = phase1_data.get("selected_core_objects", [])

        with open(raw_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            full_description = raw_data.get("full_description", "")

        if not core_objects or not full_description:
            return f"[Skipped] {filename}: missing core objects or description."
    except Exception as e:
        return f"[Error] Failed to read {filename}: {e}"

    final_patent_claims = []
    global_claim_id = 1

    with open(debug_log_path, 'w', encoding='utf-8') as df:
        df.write(f"=== Debug log: {raw_filename} ===\n\n")

    for obj in core_objects:
        obj_name = obj.get("object_name", "Unknown Object")
        # Use the category already determined in Stage 1 — no re-classification
        category = normalize_category(obj.get("category", "Apparatus"))

        system_prompt = PROMPTS.get(category, PROMPTS["Apparatus"])
        user_prompt = f"Extract the features specifically for the object named '{obj_name}' (Category: {category}).\n\nPatent Description:\n{full_description}"

        raw_result_text = call_llm(system_prompt, user_prompt)

        with open(debug_log_path, 'a', encoding='utf-8') as df:
            df.write(f"--- Object: {obj_name} ({category}) ---\n")
            df.write(raw_result_text if raw_result_text else "[Empty API response]")
            df.write("\n\n" + "=" * 50 + "\n\n")

        if not raw_result_text:
            continue

        clean_json_str = extract_json_from_text(raw_result_text)
        if not clean_json_str:
            print(f"[Warning] {filename} -> {obj_name}: no JSON found in response.")
            continue

        try:
            extracted_json = json.loads(clean_json_str)
            claims_array = extracted_json.get("claims", [])

            id_map = {}
            for claim in claims_array:
                old_id = claim.get("claim_id")
                id_map[old_id] = global_claim_id
                claim["claim_id"] = global_claim_id

                if "depends_on" in claim:
                    dep = claim["depends_on"]
                    if dep in id_map:
                        claim["depends_on"] = id_map[dep]

                # Pass the Stage 1 category through to each claim record
                claim["subject_matter_category"] = category

                final_patent_claims.append(claim)
                global_claim_id += 1

        except json.JSONDecodeError as e:
            print(f"[Error] {filename} -> {obj_name}: JSON decode failed ({e}). See debug log.")
            continue
        except Exception as e:
            print(f"[Error] {filename} -> {obj_name}: unexpected error ({e}).")
            continue

    if not final_patent_claims:
        return f"[Error] {filename}: no claims extracted. Check {debug_log_path}"

    final_output = {
        "application_number": raw_data.get("application_number", raw_filename),
        "structured_claims": final_patent_claims
    }

    output_path = os.path.join(STAGE2_DIR, f"final_{raw_filename}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    return f"[OK] {filename} -> {len(final_patent_claims)} claims generated."


# ================= Main =================

def run_stage2():
    phase1_files = glob.glob(os.path.join(STAGE1_TWO_DIR, "*.json"))
    total_files = len(phase1_files)

    if total_files == 0:
        print(f"[Warning] No JSON files found in {STAGE1_TWO_DIR}.")
        return

    print(f"[*] Stage 2 started: {total_files} patents to process.")
    print(f"[*] Debug logs will be saved to: {DEBUG_DIR}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_patent, fp): fp for fp in phase1_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"({completed}/{total_files}) {future.result()}")

    print("\n[*] All tasks complete.")


if __name__ == "__main__":
    run_stage2()
