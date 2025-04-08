import logging

from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from search_engine import SearchEngine
from llm_model import initialize_llm_model
from config import LLAMA_MODEL_PATH, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class KoreanLlamaRAG:
    """한국어 Llama 모델을 사용한 RAG 시스템 클래스"""

    def __init__(
            self,
            vector_store: FAISS,
            model_path: str = LLAMA_MODEL_PATH,
            search_engine: Optional[SearchEngine] = None
    ):
        # LLM 모델 초기화
        self.llm = initialize_llm_model(model_path)
        
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
        self.search_engine = search_engine

        # RAG 프롬프트 템플릿 (한국어 버전)
        self.rag_prompt_template = """
당신은 제공된 문서 내용과 질문을 바탕으로 답변하는 도우미입니다.

문서 내용:
{context}

질문: {question}

제공된 문서 내용을 기반으로 답변해주세요. 문서에 관련 정보가 없는 경우 "제공된 문서에서 이 정보를 찾을 수 없습니다"라고 답변하세요.
답변:
"""

        self.search_prompt_template = """
다음의 검색 결과와 질문을 바탕으로 답변해주세요.

검색 결과:
{search_results}

질문: {question}

검색 결과를 바탕으로 정확하고 관련성 높은 답변을 해주세요. 확신이 없는 내용은 포함하지 마세요.
답변:
"""

        # RAG 체인 설정 (최신 LangChain API 사용)
        self.rag_prompt = PromptTemplate.from_template(self.rag_prompt_template)
        self.search_prompt = PromptTemplate.from_template(self.search_prompt_template)

        # LCEL(LangChain Expression Language) 스타일의 체인 설정
        self.qa_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
        )


    def answer_question(self, question: str) -> Dict[str, Any]:
        """질문에 답변하는 함수"""
        logger.info(f"질문 처리 중: {question}")

        try:
            # 먼저 RAG 시스템으로 답변 시도
            # 문서 검색 수동 실행
            docs = self.retriever.get_relevant_documents(question)

            # 답변 생성
            answer = self.qa_chain.invoke(question)

            # 문서에서 정보를 찾지 못했는지 확인
            if "제공된 문서에서 이 정보를 찾을 수 없습니다" in answer and self.search_engine:
                logger.info("문서에서 정보를 찾지 못했습니다. 검색 엔진을 사용합니다.")
                return self._search_and_answer(question)

            return {
                "answer": answer,
                "source": "document",
                "source_documents": docs
            }

        except Exception as e:
            logger.error(f"답변 생성 중 오류 발생: {e}")
            if self.search_engine:
                return self._search_and_answer(question)
            return {"answer": f"오류가 발생했습니다: {str(e)}", "source": "error"}


    def _search_and_answer(self, question: str) -> Dict[str, Any]:
        """검색 엔진을 사용하여 답변하는 함수"""
        logger.info("검색 엔진을 사용하여 답변을 생성합니다.")

        # DuckDuckGo로 검색
        search_results = self.search_engine.search_duckduckgo(question)

        if not search_results:
            return {
                "answer": "제공된 문서와 검색 결과에서 이 정보를 찾을 수 없습니다.",
                "source": "none"
            }

        # 검색 결과 텍스트 구성
        search_content = ""
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "제목 없음")
            body = result.get("body", "")
            url = result.get("href", "")

            search_content += f"{i}. {title}\n{body}\n출처: {url}\n\n"

            # 필요한 경우 웹페이지 내용 추가 가져오기
            if len(body) < 100 and url:  # 본문이 짧은 경우
                page_content = self.search_engine.get_webpage_content(url)
                if page_content:
                    search_content += f"추가 내용: {page_content[:1000]}...\n\n"

        # 검색 결과로 답변 생성
        try:
            chain = (
                    {"search_results": lambda _: search_content, "question": lambda _: question}
                    | self.search_prompt
                    | self.llm
                    | StrOutputParser()
            )

            answer = chain.invoke({})

            return {
                "answer": answer,
                "source": "search",
                "search_results": search_results
            }
        except Exception as e:
            logger.error(f"검색 기반 답변 생성 중 오류 발생: {e}")
            # 대안으로 단순 프롬프트 사용
            formatted_prompt = self.search_prompt.format(
                search_results=search_content,
                question=question
            )

            answer = self.llm.invoke(formatted_prompt)

            return {
                "answer": answer,
                "source": "search",
                "search_results": search_results
            }

