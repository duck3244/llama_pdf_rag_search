import os
import torch

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# 모델 설정
# HuggingFace Hub 리포지토리 ID 또는 로컬 경로. HF 캐시(~/.cache/huggingface/hub)에
# 이미 받아둔 모델이 있으면 그대로 사용된다. 오프라인에서도 캐시 히트 시 동작.
LLAMA_MODEL_PATH = _env(
    "LLAMA_MODEL_PATH",
    "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1",
)
EMBEDDING_MODEL_NAME = _env(
    "EMBEDDING_MODEL_NAME",
    os.path.join(BASE_DIR, "models/ko-sroberta-multitask"),
)

# 하드웨어 설정
DEVICE = _env("DEVICE", "cpu")

# 모델 생성 파라미터
TEMPERATURE = _env_float("TEMPERATURE", 0.1)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 128)

# 검색 설정
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 2)
CONTEXT_MAX_TOKENS = _env_int("CONTEXT_MAX_TOKENS", 512)

# 청크 설정
CHUNK_SIZE = _env_int("CHUNK_SIZE", 256)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 30)

# 저장소 및 업로드 설정
VECTOR_DB_PATH = _env("VECTOR_DB_PATH", os.path.join(BASE_DIR, "vector_db"))
UPLOAD_FOLDER = _env("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
MAX_CONTENT_LENGTH = _env_int("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)

# 로깅 설정
LOG_FILE = _env("LOG_FILE", os.path.join(BASE_DIR, "rag_system.log"))
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

# CPU 스레드 설정 (값을 환경 변수로 조정 가능)
CPU_THREADS = _env_int("CPU_THREADS", 8)

# Flask 서버 설정 (프로덕션에서는 FLASK_DEBUG=0, FLASK_HOST=127.0.0.1 권장)
FLASK_HOST = _env("FLASK_HOST", "127.0.0.1")
FLASK_PORT = _env_int("FLASK_PORT", 5000)
FLASK_DEBUG = _env_bool("FLASK_DEBUG", False)


def setup_cpu_optimization() -> None:
    """CPU 스레드 수를 환경 변수 및 torch에 적용한다."""
    threads = str(CPU_THREADS)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = threads
    torch.set_num_threads(CPU_THREADS)
