import os
import faiss
import logging

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """벡터 저장소를 관리하는 클래스"""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        # 환경 변수로 토크나이저 캐싱 활성화
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬처리 비활성화로 메모리 사용 감소

        try:
            # 직접 sentence-transformers 사용
            import sentence_transformers
            self.model = sentence_transformers.SentenceTransformer(embedding_model_name)

            # 커스텀 임베딩 클래스 생성
            class CustomEmbeddings:
                def __init__(self, st_model):
                    self.model = st_model

                def embed_documents(self, texts):
                    embeddings = self.model.encode(
                        texts,
                        batch_size=8,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    return embeddings.tolist()

                def embed_query(self, text):
                    embedding = self.model.encode(
                        text,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    return embedding.tolist()

            self.embeddings = CustomEmbeddings(self.model)
            logger.info(f"임베딩 모델 로드 성공: {embedding_model_name}")

        except Exception as e:
            logger.error(f"임베딩 모델 로드 오류: {e}")
            logger.info("기본 임베딩 모델 사용으로 전환합니다.")

            # 기본 임베딩 모델 사용
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e2:
                logger.error(f"기본 임베딩 모델 로드 오류: {e2}")
                # 최후의 수단으로 직접 sentence-transformers 사용
                import sentence_transformers
                self.model = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = CustomEmbeddings(self.model)


    def create_vector_store(self, text_chunks: List[str]) -> FAISS:
        """텍스트 청크로부터 벡터 저장소를 생성하는 함수"""
        logger.info("벡터 저장소 생성 중...")
        return FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)


    def save_vector_store(self, vector_store: FAISS, save_path: str):
        """벡터 저장소를 저장하는 함수"""
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vector_store.save_local(save_path)
        logger.info(f"벡터 저장소가 저장되었습니다: {save_path}")


    def load_vector_store(self, load_path: str) -> FAISS:
        """저장된 벡터 저장소를 로드하는 함수"""
        logger.info(f"벡터 저장소를 불러오는 중: {load_path}")

        # allow_dangerous_deserialization 파라미터를 True로 설정하여 보안 경고 우회
        # 참고: 신뢰할 수 있는 소스에서만 이 옵션을 사용하세요
        try:
            return FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"벡터 저장소 로드 실패: {e}")
            logger.info("새 벡터 저장소를 생성합니다.")
            # 로드에 실패하면 빈 벡터 저장소 반환
            return FAISS.from_texts(["더미 텍스트"], self.embeddings)

