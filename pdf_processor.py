import os
import fitz  # PyMuPDF
import logging

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF 파일을 처리하여 텍스트 추출 및 청크로 분할하는 클래스"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF 파일에서 텍스트를 추출하는 함수"""
        logger.info(f"PDF 파일 처리 중: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
            raise

        return text


    def split_text_into_chunks(self, text: str) -> List[str]:
        """추출된 텍스트를 청크로 분할하는 함수"""
        logger.info("텍스트를 청크로 분할 중...")
        return self.text_splitter.split_text(text)

