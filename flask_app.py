import os
import threading

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from rag_system import KoreanLlamaRAG
from pdf_processor import PDFProcessor
from search_engine import SearchEngine
from logging_setup import setup_logging
from vector_store import VectorStoreManager
from config import VECTOR_DB_PATH, LLAMA_MODEL_PATH, EMBEDDING_MODEL_NAME, setup_cpu_optimization

# 로깅 설정
logger = setup_logging()

# CPU 최적화 설정
setup_cpu_optimization()

# Flask 앱 초기화
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB 파일

# 필요한 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LLAMA_MODEL_PATH), exist_ok=True)

# 전역 변수로 RAG 시스템 저장
rag_system = None
pdf_processor = PDFProcessor()
vector_store_manager = VectorStoreManager(embedding_model_name=EMBEDDING_MODEL_NAME)
search_engine = SearchEngine()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_pdf_in_background(pdf_path, vector_store_path):
    """백그라운드에서 PDF 처리하는 함수"""
    global rag_system

    try:
        # PDF에서 텍스트 추출 및 청크 분할
        pdf_text = pdf_processor.extract_text_from_pdf(pdf_path)
        text_chunks = pdf_processor.split_text_into_chunks(pdf_text)

        # 벡터 저장소 생성 및 저장
        vector_store = vector_store_manager.create_vector_store(text_chunks)
        vector_store_manager.save_vector_store(vector_store, vector_store_path)

        # 한국어 Llama RAG 시스템 초기화
        rag_system = KoreanLlamaRAG(
            vector_store=vector_store,
            model_path=LLAMA_MODEL_PATH,
            search_engine=search_engine
        )

        logger.info("PDF 처리 및 RAG 시스템 초기화가 완료되었습니다.")
    except Exception as e:
        logger.error(f"PDF 처리 중 오류 발생: {e}")
        logger.exception("상세 오류:")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 파일이 요청에 포함되어 있는지 확인
    if 'file' not in request.files:
        return jsonify({'error': '파일이 포함되어 있지 않습니다.'}), 400

    file = request.files['file']

    # 파일 이름이 비어있지 않은지 확인
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        vector_store_path = os.path.join(VECTOR_DB_PATH, f"{filename.split('.')[0]}_vectors")

        # 파일 저장
        file.save(pdf_path)

        # 백그라운드에서 PDF 처리 시작
        thread = threading.Thread(target=process_pdf_in_background, args=(pdf_path, vector_store_path))
        thread.daemon = True  # 메인 쓰레드 종료 시 함께 종료되도록 설정
        thread.start()

        return jsonify({
            'success': True,
            'message': 'PDF 파일이 업로드되었습니다. 처리 중입니다...',
            'filename': filename
        })

    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': '질문이 비어있습니다.'}), 400

    if rag_system is None:
        return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다. PDF를 먼저 업로드해주세요.'}), 400

    try:
        # 질문에 답변
        result = rag_system.answer_question(question)

        response = {
            'answer': result['answer'],
            'source': result['source']
        }

        # 응답에 참조 문서 추가
        if result['source'] == 'document' and 'source_documents' in result:
            response['references'] = [
                {'content': doc.page_content[:200] + '...'}
                for doc in result['source_documents'][:2]
            ]

        # 응답에 검색 결과 추가
        elif result['source'] == 'search' and 'search_results' in result:
            response['references'] = [
                {
                    'title': res.get('title', '제목 없음'),
                    'snippet': res.get('body', '')[:200] + '...',
                    'url': res.get('href', '')
                }
                for res in result['search_results'][:2]
            ]

        return jsonify(response)

    except Exception as e:
        logger.error(f"질문 응답 중 오류 발생: {e}")
        logger.exception("상세 오류:")
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500


@app.route('/status')
def system_status():
    model_ready = os.path.exists(LLAMA_MODEL_PATH)
    embedding_ready = os.path.exists(EMBEDDING_MODEL_NAME)

    return jsonify({
        'system_ready': rag_system is not None,
        'search_enabled': True,
        'model_ready': model_ready,
        'embedding_ready': embedding_ready,
        'processing_active': threading.active_count() > 1  # 백그라운드 스레드가 활성화되어 있는지 확인
    })


@app.route('/health')
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

