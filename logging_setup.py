import logging


def setup_logging(level=logging.INFO):
    """로깅 설정 초기화 함수"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
            logging.FileHandler('rag_system.log')  # 파일 출력
        ]
    )
    return logging.getLogger(__name__)

