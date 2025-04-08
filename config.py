import os
import torch

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 설정
LLAMA_MODEL_PATH = os.path.join(BASE_DIR, "models/torchtorchkimtorch-Llama-3.2-Korean-GGACHI-1B-Instruct-v1")
EMBEDDING_MODEL_NAME = os.path.join(BASE_DIR, "models/ko-sroberta-multitask")

# 하드웨어 설정
DEVICE = "cpu"

# 모델 생성 파라미터
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 128

# 검색 설정
RETRIEVAL_TOP_K = 2
CONTEXT_MAX_TOKENS = 512

# 청크 설정
CHUNK_SIZE = 256
CHUNK_OVERLAP = 30

# 저장소 설정
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")


# CPU 스레드 설정 최적화
def setup_cpu_optimization():
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

