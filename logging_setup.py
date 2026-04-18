import logging
from typing import Optional

from config import LOG_FILE, LOG_LEVEL

_LOGGER_NAME = "rag_system"
_configured = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """루트 로거에 스트림/파일 핸들러를 한 번만 설정한다.

    여러 모듈에서 호출되어도 핸들러가 중복 추가되지 않도록 보장한다.
    """
    global _configured

    resolved_level = getattr(
        logging,
        (level or LOG_LEVEL).upper(),
        logging.INFO,
    )
    resolved_file = log_file or LOG_FILE

    root = logging.getLogger()
    if not _configured:
        root.setLevel(resolved_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        try:
            file_handler = logging.FileHandler(resolved_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as e:
            root.warning("로그 파일 핸들러 설정 실패 (%s): %s", resolved_file, e)

        _configured = True
    else:
        root.setLevel(resolved_level)

    return logging.getLogger(_LOGGER_NAME)
