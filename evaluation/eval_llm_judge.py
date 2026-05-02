import os
import json
import csv
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STAGE5_END_DIR, LLAMA_NEMOTRON_PATH

# Directories to evaluate — add or replace with other model output dirs as needed
TARGET_DIRS = [
    STAGE5_END_DIR,
]

MODEL_PATH = LLAMA_NEMOTRON_PATH

DIMENSIONS = [
    "Concept_Generalization",
    "Subject_Diversity",
    "Feature_Synergy",
    "Hierarchical_Fallback",
    "Boundary_Control",
    "Drafting_Norms"
]

SYSTEM_PROMPT = """You are a highly experienced Senior Patent Attorney and Patent Examiner. Your task is to rigorously evaluate the quality of AI-generated draft patent claims.

[Evaluation Methodology]
You must evaluate the claims across 6 professional dimensions based on absolute patent drafting standards. For each dimension, you must perform a step-by-step Chain-of-Thought (CoT) reasoning—analyzing the strategic intent, identifying strengths, and catching fatal flaws—BEFORE assigning a score from 1 to 10.

[Evaluation Criteria]
1. Concept_Generalization: Does the draft extract broad concepts and use defensive drafting (e.g., negative limitations) to prevent circumvention, avoiding overly restrictive "picture claims"?
2. Subject_Diversity: Does it cover multiple statutory categories (product, system, method, downstream hardware) to build a full supply-chain protection ecosystem?
3. Feature_Synergy: Do the features form a logical, synergistic relationship? Are functional limitations strictly grounded by structural or compositional features?
4. Hierarchical_Fallback: Do dependent claims form a healthy flat/star-shaped topology? Does it avoid dangerous "fatal linear chains" and mindless exhaustive combinatorial loops?
5. Boundary_Control: Are numerical ranges, physical parameters, and material properties scientifically plausible, avoiding hallucinated limits or arbitrary, rigid dimensions?
6. Drafting_Norms: Are product features strictly decoupled from manufacturing method steps? Are antecedent bases clear and unambiguous?
"""

USER_PROMPT_TEMPLATE = """
Please evaluate the following Draft Claims based on the criteria provided in your system instructions.

[Input Data]
Draft Claims (To be evaluated):
<<<
{draft_claims}
>>>

[Output Format Requirement]
You MUST return your evaluation STRICTLY in the following JSON format. Do NOT include any markdown formatting outside the JSON block. Do NOT add any introductory or concluding text.
{{
  "reasoning": {{
    "Concept_Generalization": "Your detailed CoT reasoning here...",
    "Subject_Diversity": "Your detailed CoT reasoning here...",
    "Feature_Synergy": "Your detailed CoT reasoning here...",
    "Hierarchical_Fallback": "Your detailed CoT reasoning here...",
    "Boundary_Control": "Your detailed CoT reasoning here...",
    "Drafting_Norms": "Your detailed CoT reasoning here..."
  }},
  "scores": {{
    "Concept_Generalization": <int 1-10>,
    "Subject_Diversity": <int 1-10>,
    "Feature_Synergy": <int 1-10>,
    "Hierarchical_Fallback": <int 1-10>,
    "Boundary_Control": <int 1-10>,
    "Drafting_Norms": <int 1-10>
  }}
}}
"""


def extract_json_from_response(response_text):
    try:
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {
        "reasoning": {d: "Failed to parse." for d in DIMENSIONS},
        "scores": {d: 0 for d in DIMENSIONS}
    }


def main():
    print("[*] Warming up vLLM engine and loading Llama-3.1-Nemotron-70B...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        quantization=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=25000,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=4096
    )

    print("[*] Model loaded. Starting batch tasks...\n")

    for dir_idx, gen_dir in enumerate(TARGET_DIRS, 1):
        print("=" * 60)
        print(f"[{dir_idx}/{len(TARGET_DIRS)}] Analyzing directory: {gen_dir}")

        if not os.path.exists(gen_dir):
            print(f"[!] Directory not found, skipping: {gen_dir}")
            continue

        out_csv = os.path.join(gen_dir, "llm_judge_evaluation_results.csv")

        gen_files = [
            f for f in os.listdir(gen_dir)
            if f.endswith(".txt") and os.path.isfile(os.path.join(gen_dir, f))
        ]

        if not gen_files:
            print("[!] No .txt files found in this directory, skipping.")
            continue

        file_ids = []
        prompts = []

        for filename in gen_files:
            file_id = os.path.splitext(filename)[0]
            gen_file_path = os.path.join(gen_dir, filename)

            with open(gen_file_path, 'r', encoding='utf-8') as f:
                gen_text = f.read().strip()

            if gen_text:
                user_content = USER_PROMPT_TEMPLATE.format(draft_claims=gen_text)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                file_ids.append(file_id)
                prompts.append(formatted_prompt)

        num_tasks = len(prompts)
        if num_tasks == 0:
            continue

        print(f"[*] Loaded {num_tasks} valid txt files. Submitting to vLLM for inference...")
        outputs = llm.generate(prompts, sampling_params)

        print(f"[*] Inference complete. Writing results to: {out_csv}")
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            headers = (
                ["File_Name"]
                + [f"Score_{dim}" for dim in DIMENSIONS]
                + [f"Reasoning_{dim}" for dim in DIMENSIONS]
            )
            writer.writerow(headers)

            for i, output in enumerate(tqdm(outputs, desc="Saving results", unit="file")):
                generated_text = output.outputs[0].text
                parsed_result = extract_json_from_response(generated_text)

                scores = parsed_result.get("scores", {})
                reasonings = parsed_result.get("reasoning", {})

                row = [file_ids[i]]
                for dim in DIMENSIONS:
                    row.append(scores.get(dim, 0))
                for dim in DIMENSIONS:
                    row.append(reasonings.get(dim, "N/A"))

                writer.writerow(row)

    print("\n[*] LLM-as-a-judge evaluation complete.")


if __name__ == "__main__":
    main()
