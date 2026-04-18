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

        # RAG 프롬프트 템플릿 (한국어)
        # 의도적으로 "없을 경우" 분기를 제시하지 않는다. 소형 모델은 그 옵션이
        # 있으면 회피로 기울어지므로, 문서 기반 답변에만 집중시키고 관련성이
        # 낮을 때의 폴백은 상위 로직(answer_question)에서 처리한다.
        self.rag_prompt_template = """다음 문서를 참고하여 질문에 한국어로 구체적으로 답하세요. 문서에 등장한 장소/이름/특징을 최대한 활용하세요.

[문서]
{context}

[질문] {question}

[답변]"""

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
            answer = self.qa_chain.invoke(question).strip()

            # 문서에서 정보를 찾지 못한 시그널: 답변이 비어있거나 "찾을 수 없" 패턴 포함
            not_found = (
                not answer
                or "찾을 수 없" in answer
                or "정보가 없" in answer
            )
            if not_found and self.search_engine:
                logger.info("문서에서 정보를 찾지 못했습니다. 검색 엔진을 사용합니다.")
                return self._search_and_answer(question)

            return {
                "answer": answer,
                "source": "document",
                "source_documents": docs
            }

        except Exception:
            logger.exception("답변 생성 중 오류 발생")
            if self.search_engine:
                return self._search_and_answer(question)
            return {"answer": "답변 생성 중 오류가 발생했습니다.", "source": "error"}


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

        # 본문이 짧은 결과의 URL만 선별해 병렬로 페이지 내용 조회
        urls_to_fetch = [
            r.get("href", "")
            for r in search_results
            if r.get("href") and len(r.get("body", "")) < 100
        ]
        page_contents = (
            self.search_engine.get_webpage_contents(urls_to_fetch)
            if urls_to_fetch
            else {}
        )

        # 검색 결과 텍스트 구성
        search_content = ""
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "제목 없음")
            body = result.get("body", "")
            url = result.get("href", "")

            search_content += f"{i}. {title}\n{body}\n출처: {url}\n\n"

            page_content = page_contents.get(url)
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

