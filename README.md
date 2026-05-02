# Patent Claim Generator

An automated, multi-stage pipeline for generating legally structured patent claims from raw patent descriptions, leveraging large language models (LLMs) and retrieval-augmented generation (RAG).

---

## Overview

This system transforms an unstructured patent description into a complete, hierarchical claim set (independent + dependent claims) through six sequential stages:

```
Stage 0  →  Stage 1  →  Stage 2  →  Stage 3  →  Stage 4  →  Stage 5
Chunking    Discovery    Extraction   Surgery     Generalization  Assembly
```

Each stage refines the output of the previous one. The final output is a numbered, court-ready claim string saved as both JSON and plain text.

---

## Pipeline Architecture

| Stage | Script | Role |
|---|---|---|
| **0 — Pre-processing** | `chunk_description.py` | Splits raw patent descriptions into overlapping token-bounded chunks for RAG retrieval |
| **1 — Object Discovery** | `s1_object_discovery.py` | Identifies all patentable inventive objects (Apparatus, Method, System, Composition) and selects up to 4 core objects via a two-pass LLM strategy |
| **2 — Feature Extraction** | `s2_feature_extraction.py` | Extracts an exhaustive list of protectable features for each selected object using category-specific expert prompts |
| **3a — Suspect Lock-On** | `s3a_suspect_lockon.py` | Detects structurally redundant claim pairs using DeepSeek-R1's chain-of-thought reasoning |
| **3b — Evidence Collection** | `s3b_evidence_collector.py` | Retrieves the most relevant specification chunks for each suspect pair using BGE-Reranker-v2-m3 |
| **3c — Claim Surgery** | `s3c_claim_surgeon.py` | Performs element-by-element redundancy removal (DELETE / MODIFY / KEEP) guided by the retrieved evidence |
| **4a — Transition Fixer** | `s4a_transition_fixer.py` | Selects the legally correct transition word (`comprising:`, `consisting of:`, etc.) for each claim based on its category |
| **4b — RAG Binding** | `s4b_rag_binding.py` | Retrieves specification evidence for each claim to anchor the generalization step |
| **4c — Generalizer** | `s4c_generalizer.py` | Broadens each claim element: removes overly specific numerical parameters and lifts specific materials to functional classes |
| **4d — Minimalist Drafter** | `s4d_minimalist_drafter.py` | Applies Occam's Razor claiming: selects only the 3–5 essential core elements for each independent claim |
| **5 — Claim Assembler** | `s5_claim_assembler.py` | Expands each minimalist independent claim into a full hierarchical cluster (independent + dependent claims) with resume-checkpoint support |

---

## Requirements

### Hardware

- **GPU**: NVIDIA A100 80 GB recommended
- The BGE reranker stages run on GPU via `device_map="auto"` (fp16)
- The LLM stages call a separately hosted vLLM server

### Models

Download and place the following models locally:

| Model | Used in | Purpose |
|---|---|---|
| `DeepSeek-R1-70B` | Stages 1, 2, 3a, 3c, 4a, 4c, 4d, 5 | Main reasoning LLM (served via vLLM) |
| `BAAI/bge-reranker-v2-m3` | Stages 3b, 4b | Cross-encoder reranker for RAG evidence retrieval |
| `BAAI/bge-m3` | Evaluation | BERTScore semantic similarity |
| `Llama-3.1-Nemotron-70B-Instruct` | Evaluation | LLM-as-a-Judge quality scoring |

### Python Dependencies

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10. Key packages: `openai`, `transformers`, `torch`, `vllm`, `bert-score`, `sacrebleu`, `rouge-score`, `tiktoken`, `spacy`.

Install the spaCy `xx` model after pip install:
```bash
python -m spacy download xx_ent_wiki_sm
```

---

## Setup

### 1. Configure paths

Edit `config/config.py` and fill in the four `# TODO` lines:

```python
PROJECT_DATA_ROOT   = "/path/to/project/data"
DEEPSEEK_MODEL_PATH = "/path/to/models/DeepSeek-R1-70B"
BGE_RERANKER_PATH   = "/path/to/models/bge-reranker-v2-m3"
BGE_M3_PATH         = "/path/to/models/bge-m3"
LLAMA_NEMOTRON_PATH = "/path/to/models/llama-3.1-nemotron-70b-instruct"
```

All intermediate and output directories are derived automatically from `PROJECT_DATA_ROOT`.

### 2. Prepare input data

Place your patent JSON files in `data/input/`. Each file must contain a `full_description` field:

```json
{
  "application_number": "US12345678",
  "full_description": "The present invention relates to ..."
}
```

For the evaluation scripts, place human-written reference claims in `data/human_dataset/` using the same filename convention, with a `claims` field:

```json
{
  "application_number": "US12345678",
  "claims": "1. A device comprising ..."
}
```

### 3. Start the DeepSeek-R1 vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/models/DeepSeek-R1-70B \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port 8010
```

The pipeline expects the server at `http://localhost:8010/v1`. Adjust `--tensor-parallel-size` to match your GPU count (e.g., `2` for 2×A100-80G, `4` for 4×A100-80G). Change `DEEPSEEK_BASE_URL` in `config.py` if you use a different port or host.

---

## Running the Pipeline

Run each stage in order. All stages support concurrent processing via `ThreadPoolExecutor`.

```bash
# Stage 0 — Chunk patent descriptions for RAG
python src/stage0_preprocess/chunk_description.py

# Stage 1 — Discover and select core patentable objects
python src/stage1_discovery/s1_object_discovery.py

# Stage 2 — Extract exhaustive feature lists per object
python src/stage2_extraction/s2_feature_extraction.py

# Stage 3a — Detect structurally redundant claim pairs
python src/stage3_surgery/s3a_suspect_lockon.py

# Stage 3b — Retrieve evidence for each suspect pair (requires GPU)
python src/stage3_surgery/s3b_evidence_collector.py

# Stage 3c — Surgically remove redundant elements
python src/stage3_surgery/s3c_claim_surgeon.py

# Stage 4a — Fix transition words per claim category
python src/stage4_generalization/s4a_transition_fixer.py

# Stage 4b — Bind RAG evidence to each claim (requires GPU)
python src/stage4_generalization/s4b_rag_binding.py

# Stage 4c — Generalize claim elements (broaden scope)
python src/stage4_generalization/s4c_generalizer.py

# Stage 4d — Apply minimalist Occam's Razor drafting
python src/stage4_generalization/s4d_minimalist_drafter.py

# Stage 5 — Assemble full hierarchical claim clusters
python src/stage5_assembly/s5_claim_assembler.py
```

Stage 5 includes a built-in **resume checkpoint**: already-completed files are automatically skipped on re-run.

---

## Evaluation

Three evaluation scripts are provided. All read from `data/output/end/` and match generated `.txt` files against human reference JSON files by application number.

```bash
# N-gram metrics: BLEU, ROUGE-1, ROUGE-L
python evaluation/eval_ngram.py

# Semantic similarity: BERTScore (BGE-M3)
python evaluation/eval_bertscore.py

# LLM-as-a-Judge: 6-dimensional quality scoring (Llama-3.1-Nemotron-70B)
python evaluation/eval_llm_judge.py
```

Each script writes a CSV file to the same output directory.

### LLM-as-a-Judge Dimensions

| Dimension | Description |
|---|---|
| `Concept_Generalization` | Breadth of abstract concepts; avoidance of "picture claims" |
| `Subject_Diversity` | Coverage of multiple statutory categories (product, method, system) |
| `Feature_Synergy` | Logical coherence and functional grounding of claim elements |
| `Hierarchical_Fallback` | Star-shaped dependency topology; no fatal linear chains |
| `Boundary_Control` | Scientific plausibility of numerical parameters |
| `Drafting_Norms` | Antecedent basis clarity; product/method decoupling |

---

## Project Structure

```
patent_claim_generator/
├── config/
│   └── config.py                  # All paths, model names, and hyperparameters
├── src/
│   ├── stage0_preprocess/
│   │   └── chunk_description.py
│   ├── stage1_discovery/
│   │   └── s1_object_discovery.py
│   ├── stage2_extraction/
│   │   └── s2_feature_extraction.py
│   ├── stage3_surgery/
│   │   ├── s3a_suspect_lockon.py
│   │   ├── s3b_evidence_collector.py
│   │   └── s3c_claim_surgeon.py
│   ├── stage4_generalization/
│   │   ├── s4a_transition_fixer.py
│   │   ├── s4b_rag_binding.py
│   │   ├── s4c_generalizer.py
│   │   └── s4d_minimalist_drafter.py
│   └── stage5_assembly/
│       └── s5_claim_assembler.py
├── evaluation/
│   ├── eval_ngram.py
│   ├── eval_bertscore.py
│   └── eval_llm_judge.py
├── requirements.txt
└── README.md
```

### Data Directory Layout (auto-created under `PROJECT_DATA_ROOT`)

```
data/
├── input/                         # Raw patent JSONs (your input)
├── human_dataset/                 # Human-written reference claims (for evaluation)
├── chunks/                        # Stage 0 output: tokenized description chunks
├── intermediate/
│   ├── stage1/{one,two}/          # Stage 1 output: discovered and selected objects
│   ├── stage2/                    # Stage 2 output: structured feature lists
│   ├── stage3a/                   # Stage 3a output: suspect pairs registry
│   ├── stage3b/                   # Stage 3b output: evidence registry
│   ├── stage3c/                   # Stage 3c output: purified claims
│   ├── stage4a/                   # Stage 4a output: transition-fixed claims
│   ├── stage4b/                   # Stage 4b output: RAG evidence registry
│   ├── stage4c/                   # Stage 4c output: generalized claims
│   └── stage4d/                   # Stage 4d output: minimalist drafted claims
└── output/
    ├── *.json                     # Stage 5 output: full structured claim clusters
    ├── *_FINAL_CLAIMS.txt         # Stage 5 output: formatted plain-text claims
    └── end/
        └── *_FINAL_CLAIMS.txt     # Post-processed single-line claim strings
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `MAX_WORKERS` | `20` | Thread pool size for all concurrent LLM stages |
| `LLM_REQUEST_TIMEOUT` | `300` | API call timeout in seconds |
| `CHUNK_TARGET_TOKENS` | `800` | Target token count per description chunk |
| `CHUNK_OVERLAP_SENTENCES` | `2` | Sentence overlap between adjacent chunks |
| `BGE_BATCH_SIZE` | `64` | Reranker inference batch size (reduce if OOM) |
| `BERT_SCORE_BATCH_SIZE` | `16` | BERTScore batch size (reduce to 8 if OOM) |

---

## Output Format

The final plain-text output in `data/output/end/` is a continuous numbered claim string:

```
1. A [device] comprising: a first component configured to ...; a second component coupled to the first component .... 2. The [device] according to claim 1, wherein the first component comprises .... 3. The [device] according to claim 1, further comprising ...
```
