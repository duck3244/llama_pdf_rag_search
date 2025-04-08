import os

from rag_system import KoreanLlamaRAG
from pdf_processor import PDFProcessor
from search_engine import SearchEngine
from logging_setup import setup_logging
from vector_store import VectorStoreManager
from config import VECTOR_DB_PATH, LLAMA_MODEL_PATH, setup_cpu_optimization


def main():
    # 로깅 설정
    logger = setup_logging()
    
    # CPU 최적화 설정
    setup_cpu_optimization()
    
    # 예제 PDF 파일 경로
    pdf_path = "PLAYGROUND_JUNGGU.pdf"
    vector_store_path = os.path.join(VECTOR_DB_PATH, "faiss_index")

    # 1. 필요한 객체 초기화
    pdf_processor = PDFProcessor()
    vector_store_manager = VectorStoreManager()
    search_engine = SearchEngine()

    # 2. 벡터 저장소 생성 또는 로드
    if os.path.exists(vector_store_path):
        vector_store = vector_store_manager.load_vector_store(vector_store_path)
    else:
        # PDF에서 텍스트 추출 및 청크 분할
        pdf_text = pdf_processor.extract_text_from_pdf(pdf_path)
        text_chunks = pdf_processor.split_text_into_chunks(pdf_text)

        # 벡터 저장소 생성 및 저장
        vector_store = vector_store_manager.create_vector_store(text_chunks)
        vector_store_manager.save_vector_store(vector_store, vector_store_path)

    # 3. 한국어 Llama RAG 시스템 초기화
    rag_system = KoreanLlamaRAG(
        vector_store=vector_store,
        model_path=LLAMA_MODEL_PATH,
        search_engine=search_engine
    )

    # 4. 예제 질문 답변
    questions = [
        "이 문서에서 가장 중요한 정보는 무엇인가요?",  # 문서 내용에서 답변 가능
        "최신 인공지능 모델의 발전 방향은 무엇인가요?"  # 문서에 없을 가능성이 높은 질문
    ]

    for question in questions:
        print(f"\n질문: {question}")
        result = rag_system.answer_question(question)

        print(f"답변 출처: {result['source']}")
        print(f"답변: {result['answer']}")

        if result['source'] == 'document' and 'source_documents' in result:
            print("\n참조 문서:")
            for doc in result['source_documents'][:2]:  # 처음 2개만 표시
                print(f"- {doc.page_content[:150]}...")

        elif result['source'] == 'search' and 'search_results' in result:
            print("\n검색 결과:")
            for i, res in enumerate(result['search_results'][:2], 1):  # 처음 2개만 표시
                print(f"- {res.get('title', '제목 없음')}: {res.get('body', '')[:150]}...")


if __name__ == "__main__":
    main()

