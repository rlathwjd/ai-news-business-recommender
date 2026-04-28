"""
기사 본문 청킹 모듈
JSON으로 저장된 기사를 청크로 분할하고 메타데이터 태깅
"""

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_articles(json_path: str = "./data/raw_articles.json") -> list[dict]:
    """크롤링된 기사 JSON 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"[청커] {len(articles)}개 기사 로드 완료")
    return articles


def chunk_articles(articles: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """기사 본문을 청크로 분할 + 메타데이터 태깅"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    documents = []
    for article in articles:
        chunks = splitter.split_text(article["body"])

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": "aitimes",
                    "title": article["title"],
                    "url": article["url"],
                    "published_date": article["published_date"],
                    "crawled_at": article["crawled_at"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            documents.append(doc)

    print(f"[청커] 총 {len(documents)}개 청크 생성 완료")
    return documents


if __name__ == "__main__":
    articles = load_articles()
    docs = chunk_articles(articles)

    # 샘플 출력
    print("\n--- 샘플 청크 ---")
    print(f"내용: {docs[0].page_content[:200]}")
    print(f"메타데이터: {docs[0].metadata}")