"""
기사 본문 청킹 모듈
SQLite DB에 저장된 기사 중 임베딩되지 않은 기사를 청크로 분할
"""

import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DB_PATH = "./data/articles.db"


def load_unembedded_articles(db_path: str = DB_PATH) -> list[dict]:
    """DB에서 아직 임베딩되지 않은 기사 로드"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, url, title, body, published_date, crawled_at
        FROM articles
        WHERE embedded = 0
        ORDER BY id ASC
    """)

    rows = cur.fetchall()
    conn.close()

    articles = [dict(row) for row in rows]

    print(f"[청커] 임베딩 대상 기사 {len(articles)}개 로드 완료")
    return articles


def chunk_articles(
    articles: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
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
                    "article_id": article["id"],
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
    articles = load_unembedded_articles()
    docs = chunk_articles(articles)

    if docs:
        print("\n--- 샘플 청크 ---")
        print(f"내용: {docs[0].page_content[:200]}")
        print(f"메타데이터: {docs[0].metadata}")
    else:
        print("[청커] 청크 생성 대상 없음")