import logging
import requests

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List
from bs4 import BeautifulSoup

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


class SearchEngine:
    """외부 검색 엔진을 사용하는 클래스"""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, max_workers: int = 4):
        self.timeout = timeout
        self._max_workers = max_workers

    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """DuckDuckGo 검색 엔진을 사용하여 검색하는 함수"""
        logger.info(f"DuckDuckGo로 검색 중: {query}")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo 검색 중 오류 발생: {e}")
            return []

    def get_webpage_content(self, url: str) -> str:
        """웹 페이지 내용을 가져오는 함수"""
        try:
            headers = {"User-Agent": DEFAULT_USER_AGENT}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 태그 및 스크립트 제거
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.decompose()

            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"웹 페이지 내용 가져오기 실패 ({url}): {e}")
            return ""

    def get_webpage_contents(self, urls: Iterable[str]) -> Dict[str, str]:
        """여러 웹 페이지 내용을 병렬로 가져온다."""
        url_list = [u for u in urls if u]
        if not url_list:
            return {}

        workers = min(self._max_workers, len(url_list))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fetched = list(executor.map(self.get_webpage_content, url_list))
        return dict(zip(url_list, fetched))
