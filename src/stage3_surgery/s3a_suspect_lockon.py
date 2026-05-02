import os
import json
import glob
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL,
    STAGE2_DIR, STAGE3A_DIR,
    MAX_WORKERS, LLM_REQUEST_TIMEOUT
)

os.makedirs(STAGE3A_DIR, exist_ok=True)
os.makedirs(os.path.join(STAGE3A_DIR, "debug_logs"), exist_ok=True)

DEBUG_DIR = os.path.join(STAGE3A_DIR, "debug_logs")

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)
log_lock = threading.Lock()

PROMPT_STEP1_SYSTEM = """You are an Expert Patent Architect. Your task is to analyze a complete set of purified independent claims and map out their EXHAUSTIVE dependency graph based on patent law doctrines of cross-referencing.

### [DEPENDENCY RULES]
1. **The Black-Box Principle_1 (Domain-Specific)**: A higher-level Target claim often redundantly incorporates a fundamental object defined in a lower-level Base claim.
   - *Apparatus/System*: Redundantly claiming all internal layers/components of a base device.
   - *Method of Using*: Claiming the entire multi-step manufacturing process of a material before claiming its actual use.
   - *CRITICAL EXCEPTION*: A "Method of Making/Producing" claim MUST NEVER cite the product it makes as a base. Do NOT create a dependency between a Method of Making and its corresponding Product claim.
2. **The Black-Box Principle_2 (Any Domain)**: If a Target claim repeats a cohesive sub-combination, a complete sequence of algorithmic/logical steps, or structural configurations that independently define the entirety of a Base claim, it is a suspected redundancy.
3. **Directionality**: A Target claim can ONLY cite a Base claim if `Target.claim_id > Base.claim_id`.
4. **Category Firewall (CRITICAL)**: An Apparatus/System claim MUST NEVER cite a "Method of Using" or "Method of Operating" claim. Physical objects cannot contain abstract human actions.

### [FOCUS AREA EXTRACTION (CRITICAL NEW RULE)]
For every dependency pair you find, you MUST extract the `focus_area`.
The `focus_area` is a list of EXACT text snippets from the Target Claim's `elements` that you suspect are redundantly copying the Base Claim.
Do NOT summarize. Extract the exact strings (or highly specific partial strings) from the Target Claim. These snippets will be used as semantic search queries to verify the redundancy against the original patent description.

### [OUTPUT FORMAT]
Do ALL your comparative analysis and reasoning exclusively inside your `<think>...</think>` tags.
After thinking, output ONLY a valid, minified JSON array containing the exact dependency pairs and their focus areas.

Schema:
[
  {
    "target_claim_id": <int>,
    "cites_base_claim_id": <int>,
    "focus_area": [
      "<exact text snippet 1 from Target Claim>",
      "<exact text snippet 2 from Target Claim>"
    ]
  }
]
(Output an empty array `[]` if no redundancies exist.)
"""


def extract_json_from_text(response_text):
    if not response_text:
        return None
    text_without_think = re.sub(r'<think>[\s\S]*?</think>', '', response_text)
    match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text_without_think)
    return match.group(0) if match else None


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
                timeout=LLM_REQUEST_TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                tqdm.write(f"  -> [Retry {attempt + 1}] Request failed ({error_msg}), retrying...")
                time.sleep(3)
            else:
                return f"[Error] API calling failed: {error_msg}"


def write_debug_log(filename, content):
    debug_file_path = os.path.join(DEBUG_DIR, f"debug_step1_{filename}.txt")
    with log_lock:
        with open(debug_file_path, 'a', encoding='utf-8') as df:
            df.write(content)


def detect_suspects_task(filepath):
    filename = os.path.basename(filepath)
    patent_id = os.path.splitext(filename)[0]

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    claims_list = data.get("structured_claims", [])
    log_content = f"\n{'=' * 20} [Stage 3a] Suspect Lock-On: {filename} {'=' * 20}\n"

    if len(claims_list) < 2:
        log_content += "Not enough claims to build dependency graph. Skipping.\n"
        write_debug_log(filename, log_content)
        return patent_id, []

    user_prompt = f"Analyze the following set of independent claims and output the dependency graph JSON with focus_areas.\n\nRaw Claims Array:\n{json.dumps(claims_list, ensure_ascii=False, indent=2)}"
    response = call_llm(PROMPT_STEP1_SYSTEM, user_prompt)
    log_content += f"--- Raw Output ---\n{response}\n"

    suspects_graph = []
    clean_json = extract_json_from_text(response)
    if clean_json:
        try:
            suspects_graph = json.loads(clean_json)
        except json.JSONDecodeError as e:
            log_content += f"[ERROR] JSON decode failed: {e}\n"
    else:
        log_content += "[ERROR] Returned no recognizable JSON.\n"

    write_debug_log(filename, log_content)
    return patent_id, suspects_graph


def run_stage3a():
    input_files = glob.glob(os.path.join(STAGE2_DIR, "*.json"))
    if not input_files:
        print(f"[Warning] No JSON files found in {STAGE2_DIR}.")
        return

    print(f"[*] Stage 3a: Suspect lock-on (DeepSeek-R1). Found {len(input_files)} files...")

    for filepath in input_files:
        filename = os.path.basename(filepath)
        with open(os.path.join(DEBUG_DIR, f"debug_step1_{filename}.txt"), 'w', encoding='utf-8') as df:
            df.write(f"=== Stage 3a Log: {filename} ===\n")

    all_suspects_registry = {}
    total_suspects_found = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fp = {executor.submit(detect_suspects_task, fp): fp for fp in input_files}
        with tqdm(total=len(input_files), desc="Suspect lock-on", unit="file", colour="blue") as pbar:
            for future in as_completed(future_to_fp):
                patent_id, suspects_graph = future.result()
                if suspects_graph:
                    all_suspects_registry[patent_id] = suspects_graph
                    total_suspects_found += len(suspects_graph)
                pbar.update(1)

    registry_path = os.path.join(STAGE3A_DIR, "global_suspects_registry.json")
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(all_suspects_registry, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print("[*] Stage 3a complete.")
    print(f"    Patents with redundancies : {len(all_suspects_registry)}")
    print(f"    Total suspect pairs found : {total_suspects_found}")
    print(f"    Registry saved to         : {registry_path}")
    print("=" * 40)


if __name__ == "__main__":
    run_stage3a()
