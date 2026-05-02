import os
import json
import re
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    DEEPSEEK_MODEL_PATH, DEEPSEEK_BASE_URL, LLM_REQUEST_TIMEOUT,
    DATA_INPUT_DIR, STAGE1_ONE_DIR, STAGE1_TWO_DIR,
    MAX_WORKERS
)

os.makedirs(STAGE1_ONE_DIR, exist_ok=True)
os.makedirs(STAGE1_TWO_DIR, exist_ok=True)

client = OpenAI(api_key="EMPTY", base_url=DEEPSEEK_BASE_URL)

# ================= System Prompts =================

SYSTEM_PROMPT_1 = """You are an expert Patent Analyst. Your task is to perform an exhaustive reading of the provided patent "Description" and identify EVERY potential inventive object that could theoretically be protected under patent law.
Your goal is maximum recall. Do not worry about claim limits or drafting strategy yet.

### [STATUTORY CATEGORIES]
1. Apparatus (devices, machines, physical structures)
2. Method (processes, manufacturing steps, algorithms, methods of use)
3. System (networks, interacting modules, computer-implemented architectures)
4. Composition (chemical formulas, alloys, material mixtures)

### [WORKFLOW & CHAIN OF THOUGHT]
1. Read the user-provided description thoroughly.
2. Formulate your reasoning process first: briefly analyze the technical problem and the various solutions/components mentioned.
3. Extract every distinct inventive concept into the JSON array.

### [OUTPUT FORMAT]
Output ONLY a valid JSON object. Do not include markdown formatting or conversational text.
{
  "reasoning_process": "<Provide a concise 2-3 sentence summary of your reading process and the overall technical landscape of the invention.>",
  "discovered_objects": [
    {
      "object_name": "<Specific name>",
      "category": "<Apparatus/Method/System/Composition>",
      "brief_justification": "<1 sentence explaining where it is found>"
    }
  ]
}"""

SYSTEM_PROMPT_2 = """You are a Lead Patent Attorney and former Patent Examiner. You will be provided with the original patent "Description" AND a list of "discovered_objects" extracted from it.
Your task is to review the description, evaluate the provided objects, formulate a holistic Claim Drafting Strategy, and select the core independent objects to protect.

### [STRATEGIC GUIDELINES & CHAIN OF THOUGHT]
1. IDENTIFY THE CORE: Determine the absolute heart of the invention based on the full description (Rank #1).
2. BUILD THE MOAT: Decide how the remaining claims should support the core logically (e.g., product + method of making it).
3. STRATEGIC ANALYSIS: Explain this strategic logic BEFORE finalizing the list.
4. STRICT LIMIT: Select the 1 to 4 most important objects.
5. NO DOWNSTREAM FEATURES: Only identify the object category and name. Do NOT extract specific technical features or parts.

### [OUTPUT FORMAT]
Output ONLY a valid JSON object. Do not include markdown formatting or conversational text.
{
  "strategic_analysis": "<Explain your logic for selecting the core objects and determining their dependency/hierarchy to build a patent moat based on the full text.>",
  "selected_core_objects": [
    {
      "rank": 1,
      "object_name": "<Specific name>",
      "category": "<Apparatus/Method/System/Composition>"
    }
  ]
}"""

# ================= Helpers =================

def clean_json_response(response_text):
    text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


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
            return clean_json_response(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                print(f"[Error] API call failed after {max_retries} attempts: {e}")
    return None


# ================= Worker =================

def process_single_file(file_path):
    filename = os.path.basename(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            full_description = data.get("full_description", "")
            if not full_description:
                return f"[Skipped] {filename}: full_description is empty"
    except Exception as e:
        return f"[Error] Failed to read {filename}: {e}"

    # Step A: exhaustive object discovery
    user_prompt_1 = f"Here is the patent Description. Please execute the exhaustive discovery task and return the JSON:\n\n{full_description}"
    step_1_result_text = call_llm(SYSTEM_PROMPT_1, user_prompt_1)

    if not step_1_result_text:
        return f"[Error] {filename}: Step A returned empty response"

    step_1_out_path = os.path.join(STAGE1_ONE_DIR, f"step1_{filename}")
    try:
        step_1_json = json.loads(step_1_result_text)
        with open(step_1_out_path, 'w', encoding='utf-8') as f:
            json.dump(step_1_json, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        with open(step_1_out_path, 'w', encoding='utf-8') as f:
            f.write(step_1_result_text)

    # Step B: strategic selection
    user_prompt_2 = (
        "Here is the ORIGINAL PATENT DESCRIPTION:\n\n"
        f"{full_description}\n\n"
        "=========================================\n\n"
        "Here is the LIST OF DISCOVERED OBJECTS from Step A:\n\n"
        f"{step_1_result_text}\n\n"
        "=========================================\n\n"
        "Please review the description and the discovered objects, formulate the drafting strategy, and return the final JSON:"
    )
    step_2_result_text = call_llm(SYSTEM_PROMPT_2, user_prompt_2)

    if not step_2_result_text:
        return f"[Error] {filename}: Step B returned empty response"

    step_2_out_path = os.path.join(STAGE1_TWO_DIR, f"step2_{filename}")
    try:
        step_2_json = json.loads(step_2_result_text)
        with open(step_2_out_path, 'w', encoding='utf-8') as f:
            json.dump(step_2_json, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        with open(step_2_out_path, 'w', encoding='utf-8') as f:
            f.write(step_2_result_text)

    return f"[OK] {filename} done"


# ================= Main =================

def run_stage1():
    json_files = glob.glob(os.path.join(DATA_INPUT_DIR, "*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print(f"[Warning] No JSON files found in {DATA_INPUT_DIR}.")
        return

    print(f"[*] Found {total_files} JSON files. Starting Stage 1 with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in json_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result_msg = future.result()
            print(f"({completed}/{total_files}) {result_msg}")

    print("\n[*] All files processed.")


if __name__ == "__main__":
    run_stage1()
