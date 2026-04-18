import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import try_to_load_from_cache
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from rag_system import KoreanLlamaRAG
from pdf_processor import PDFProcessor, is_valid_pdf
from search_engine import SearchEngine
from logging_setup import setup_logging
from vector_store import VectorStoreManager
from config import (
    VECTOR_DB_PATH,
    LLAMA_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
    UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    setup_cpu_optimization,
)

# 업로드/처리 작업 상태 관리
_job_lock = threading.Lock()
_job_state = {
    'status': 'idle',  # idle | processing | ready | failed
    'filename': None,
    'error': None,
}


def _set_job_state(**kwargs):
    with _job_lock:
        _job_state.update(kwargs)


def _get_job_state():
    with _job_lock:
        return dict(_job_state)

# 로깅 설정
logger = setup_logging()

# CPU 최적화 설정
setup_cpu_optimization()

# Flask 앱 초기화
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 필요한 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)


def _model_available(name_or_path: str) -> bool:
    """로컬 경로이면 디렉토리 존재+파일 보유 여부, 아니면 HF 캐시 존재 여부로 판정."""
    if os.path.isabs(name_or_path) or name_or_path.startswith(("./", "../")):
        return os.path.isdir(name_or_path) and any(os.scandir(name_or_path))
    # HF repo id로 간주: 캐시에 config.json이 있으면 사용 가능
    cached = try_to_load_from_cache(name_or_path, "config.json")
    return cached is not None and os.path.exists(cached)

# 전역 변수로 RAG 시스템 저장
rag_system = None
pdf_processor = PDFProcessor()
vector_store_manager = VectorStoreManager(embedding_model_name=EMBEDDING_MODEL_NAME)
search_engine = SearchEngine()

# 단일 워커 실행기: 동시에 하나의 PDF만 처리, 종료 시 진행 중인 작업이 완료될 때까지 대기
_pdf_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-worker")
atexit.register(_pdf_executor.shutdown, wait=True)


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

        _set_job_state(status='ready', error=None)
        logger.info("PDF 처리 및 RAG 시스템 초기화가 완료되었습니다.")
    except Exception:
        logger.exception("PDF 처리 중 오류 발생")
        _set_job_state(status='failed', error='PDF 처리 실패')


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
        if not filename:
            return jsonify({'error': '유효하지 않은 파일명입니다.'}), 400

        if _get_job_state()['status'] == 'processing':
            return jsonify({'error': '이전 PDF를 처리 중입니다. 잠시 후 다시 시도해주세요.'}), 409

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        vector_store_name = os.path.splitext(filename)[0]
        vector_store_path = os.path.join(VECTOR_DB_PATH, f"{vector_store_name}_vectors")

        # 파일 저장 후 PDF 유효성 검증
        file.save(pdf_path)
        if not is_valid_pdf(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                logger.warning(f"유효하지 않은 파일 삭제 실패: {pdf_path}")
            return jsonify({'error': '유효한 PDF 파일이 아닙니다.'}), 400

        # 백그라운드에서 PDF 처리 시작 (서버 종료 시 graceful하게 대기)
        _set_job_state(status='processing', filename=filename, error=None)
        _pdf_executor.submit(process_pdf_in_background, pdf_path, vector_store_path)

        return jsonify({
            'success': True,
            'message': 'PDF 파일이 업로드되었습니다. 처리 중입니다...',
            'filename': filename
        })

    return jsonify({'error': 'PDF 파일만 업로드 가능합니다.'}), 400


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

    except Exception:
        logger.exception("질문 응답 중 오류 발생")
        return jsonify({'error': '답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'}), 500


@app.route('/status')
def system_status():
    model_ready = _model_available(LLAMA_MODEL_PATH)
    embedding_ready = _model_available(EMBEDDING_MODEL_NAME)
    job = _get_job_state()

    return jsonify({
        'system_ready': rag_system is not None,
        'search_enabled': True,
        'model_ready': model_ready,
        'embedding_ready': embedding_ready,
        'processing_active': job['status'] == 'processing',
        'job_status': job['status'],
        'job_filename': job['filename'],
        'job_error': job['error'],
    })


@app.route('/health')
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)

