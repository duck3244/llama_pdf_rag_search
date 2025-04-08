import torch
import logging

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import LLAMA_MODEL_PATH, TEMPERATURE, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)


def initialize_llm_model(model_path: str = LLAMA_MODEL_PATH):
    """한국어 LLM 모델을 초기화하는 함수"""
    
    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 모델 로드 최적화 옵션
    logger.info("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    # 모델 최적화 (CPU)
    model = model.to("cpu")
    model.eval()  # 추론 모드로 설정

    # 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=1.1,
        batch_size=1  # CPU에서는 작은 배치 사이즈
    )

    # LangChain 모델 생성
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

