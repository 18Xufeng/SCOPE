import os

# ========================= Required Configuration (edit before first run) =========================

# Project data root — all intermediate and output directories are derived from this
PROJECT_DATA_ROOT = "/path/to/project/data"  # TODO: set to your actual data root

# ========================= Local Model Paths =========================

DEEPSEEK_MODEL_PATH = "/path/to/models/DeepSeek-R1-70B"            # TODO
DEEPSEEK_BASE_URL   = "http://localhost:8010/v1"

BGE_RERANKER_PATH   = "/path/to/models/bge-reranker-v2-m3"        # TODO (Stage 3b / 4b)
BGE_M3_PATH         = "/path/to/models/bge-m3"                    # TODO (Stage 6b evaluation)

LLAMA_NEMOTRON_PATH = "/path/to/models/llama-3.1-nemotron-70b-instruct"  # TODO (Stage 6c evaluation)

# ========================= Data Directories (auto-derived from PROJECT_DATA_ROOT) =========================

DATA_INPUT_DIR    = os.path.join(PROJECT_DATA_ROOT, "data/input")
DATA_HUMAN_DATASET = os.path.join(PROJECT_DATA_ROOT, "data/human_dataset")

CHUNK_DIR         = os.path.join(PROJECT_DATA_ROOT, "data/chunks")

STAGE1_ONE_DIR    = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage1/one")
STAGE1_TWO_DIR    = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage1/two")
STAGE2_DIR        = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage2")
STAGE3A_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage3a")
STAGE3B_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage3b")
STAGE3C_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage3c")
STAGE4A_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage4a")
STAGE4B_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage4b")
STAGE4C_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage4c")
STAGE4D_DIR       = os.path.join(PROJECT_DATA_ROOT, "data/intermediate/stage4d")
STAGE5_OUTPUT_DIR = os.path.join(PROJECT_DATA_ROOT, "data/output")
STAGE5_END_DIR    = os.path.join(PROJECT_DATA_ROOT, "data/output/end")

# ========================= Concurrency =========================

MAX_WORKERS = 20  # thread pool size shared by all concurrent pipeline stages

# ========================= Chunking Parameters (Stage 0) =========================

CHUNK_TARGET_TOKENS     = 800
CHUNK_OVERLAP_SENTENCES = 2

# ========================= BGE Batch Sizes (Stage 3b / 4b / evaluation) =========================

BGE_BATCH_SIZE        = 64   # A100-80G; reduce if OOM
BERT_SCORE_BATCH_SIZE = 16   # A100-80G; reduce to 8 if OOM

# ========================= LLM Request Timeout (seconds) =========================

LLM_REQUEST_TIMEOUT = 300  # shared timeout for all DeepSeek-R1 API calls
