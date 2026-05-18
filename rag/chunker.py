"""
기사 본문 청크 생성 모듈
- Supabase articles 테이블에서 가져온 기사 데이터를 Document 청크로 변환
"""

from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    기사 본문을 일정 길이의 청크로 분할하기 위한 TextSplitter 생성
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "。 ",
            " ",
            "",
        ],
    )


def clean_text(text: str | None) -> str:
    """
    기사 본문 텍스트 정리
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return text.strip()


def chunk_article(article: dict[str, Any]) -> list[Document]:
    """
    단일 기사 dict를 여러 Document 청크로 변환

    article 예시:
    {
        "id": 1,
        "url": "...",
        "title": "...",
        "body": "...",
        "published_date": "...",
        "crawled_at": "..."
    }
    """
    article_id = article.get("id")
    url = article.get("url", "")
    title = article.get("title") or "제목 없음"
    body = clean_text(article.get("body"))
    published_date = article.get("published_date")
    crawled_at = article.get("crawled_at")

    if article_id is None:
        print(f"[청커] article_id 없음. 제외: {title[:30]}")
        return []

    if not body:
        print(f"[청커] 본문 없음. 제외: {title[:30]}")
        return []

    splitter = get_text_splitter()
    chunks = splitter.split_text(body)

    documents = []

    for chunk_index, chunk_text in enumerate(chunks):
        if not chunk_text.strip():
            continue

        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "article_id": article_id,
                    "url": url,
                    "title": title,
                    "published_date": published_date,
                    "crawled_at": crawled_at,
                    "chunk_index": chunk_index,
                },
            )
        )

    return documents


def chunk_articles(articles: list[dict[str, Any]]) -> list[Document]:
    """
    여러 기사 dict를 Document 청크 리스트로 변환
    """
    documents = []

    for article in articles:
        docs = chunk_article(article)
        documents.extend(docs)

    print(
        f"[청커] 기사 {len(articles)}개 → 청크 {len(documents)}개 생성 완료"
    )

    return documents