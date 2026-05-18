import os
import sys
import warnings
from typing import Any

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

sys.path.append(".")

from db.supabase_client import get_supabase
from rag.chunker import chunk_articles

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 한 번에 가져올 미임베딩 기사 수
ARTICLE_BATCH_SIZE = 200

# Supabase insert payload가 너무 커지는 것 방지
CHUNK_INSERT_BATCH_SIZE = 200


def get_embeddings() -> HuggingFaceEmbeddings:
    """HuggingFace 임베딩 모델 초기화"""
    print(f"[임베더] 임베딩 모델 로딩 중... ({EMBEDDING_MODEL})")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_unembedded_articles(limit: int = ARTICLE_BATCH_SIZE) -> list[dict[str, Any]]:
    """
    Supabase articles 테이블에서 아직 임베딩되지 않은 기사 조회
    """
    supabase = get_supabase()

    result = (
        supabase.table("articles")
        .select("id,url,title,body,published_date,crawled_at,embedded")
        .eq("embedded", False)
        .limit(limit)
        .execute()
    )

    articles = result.data or []

    print(f"[임베더] 임베딩 대상 기사 {len(articles)}개 조회 완료")

    return articles


def normalize_documents_metadata(
    documents: list[Document],
    articles: list[dict[str, Any]],
) -> list[Document]:
    """
    chunk_articles() 결과 Document metadata에 article_id/url/title 등이 없을 경우 보완합니다.

    chunk_articles()가 이미 metadata를 잘 넣고 있다면 그대로 유지됩니다.
    """
    article_by_url = {
        article["url"]: article
        for article in articles
        if article.get("url")
    }

    article_by_id = {
        article["id"]: article
        for article in articles
        if article.get("id") is not None
    }

    normalized_docs = []

    for idx, doc in enumerate(documents):
        metadata = dict(doc.metadata or {})

        article = None

        if metadata.get("url") in article_by_url:
            article = article_by_url[metadata["url"]]
        elif metadata.get("article_id") in article_by_id:
            article = article_by_id[metadata["article_id"]]

        if article:
            metadata["article_id"] = article["id"]
            metadata["url"] = article["url"]
            metadata["title"] = article.get("title", "제목 없음")
            metadata["published_date"] = article.get("published_date")
        else:
            # chunker에서 article_id/url을 못 넘기는 경우를 대비한 최소 fallback
            # 단, 이 경우 정확한 article 매핑이 어려우므로 chunker 수정 권장
            metadata.setdefault("article_id", None)
            metadata.setdefault("url", "")
            metadata.setdefault("title", "제목 없음")

        metadata.setdefault("chunk_index", idx)

        normalized_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    return normalized_docs


def build_chunk_payloads(
    documents: list[Document],
    vectors: list[list[float]],
) -> list[dict[str, Any]]:
    """
    Supabase article_chunks 테이블에 저장할 payload 생성
    """
    payloads = []

    for doc, vector in zip(documents, vectors):
        metadata = doc.metadata or {}

        article_id = metadata.get("article_id")
        url = metadata.get("url")
        title = metadata.get("title", "제목 없음")
        chunk_index = metadata.get("chunk_index", 0)

        if article_id is None:
            print(f"[임베더] article_id 누락 청크 제외: {title[:30]}")
            continue

        payloads.append(
            {
                "article_id": article_id,
                "url": url,
                "title": title,
                "content": doc.page_content,
                "chunk_index": chunk_index,
                "embedding": vector,
            }
        )

    return payloads


def insert_chunks_to_supabase(payloads: list[dict[str, Any]]):
    """
    article_chunks 테이블에 청크와 임베딩 저장
    """
    if not payloads:
        print("[임베더] 저장할 청크 없음")
        return

    supabase = get_supabase()

    total_inserted = 0

    for start in range(0, len(payloads), CHUNK_INSERT_BATCH_SIZE):
        batch = payloads[start:start + CHUNK_INSERT_BATCH_SIZE]

        result = (
            supabase.table("article_chunks")
            .upsert(
                batch,
                on_conflict="article_id,chunk_index",
                ignore_duplicates=True,
            )
            .execute()
        )

        inserted_count = len(result.data or [])
        total_inserted += inserted_count

        print(
            f"[임베더] 청크 저장 배치 완료: "
            f"{start + 1}~{start + len(batch)} / {len(payloads)}"
        )

    print(f"[임베더] Supabase article_chunks 저장 요청 완료: {len(payloads)}개")
    print(f"[임베더] Supabase 신규 저장 응답 수: {total_inserted}개")


def mark_articles_embedded(article_ids: list[int]):
    """
    articles 테이블의 embedded 값을 true로 업데이트
    """
    if not article_ids:
        return

    supabase = get_supabase()

    for article_id in article_ids:
        (
            supabase.table("articles")
            .update({"embedded": True})
            .eq("id", article_id)
            .execute()
        )

    print(f"[임베더] {len(article_ids)}개 기사 embedded=true 업데이트 완료")


def embed_new_articles():
    """
    신규 기사 임베딩 전체 실행
    """
    articles = load_unembedded_articles()

    if not articles:
        print("[임베더] 임베딩할 신규 기사 없음")
        return

    documents = chunk_articles(articles)

    if not documents:
        print("[임베더] 생성된 청크 없음. embedded 업데이트 생략")
        return

    documents = normalize_documents_metadata(documents, articles)

    print(f"[임베더] 신규 {len(documents)}개 청크 임베딩 중...")

    embeddings = get_embeddings()
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)

    payloads = build_chunk_payloads(documents, vectors)

    if not payloads:
        print("[임베더] 저장 가능한 payload 없음. embedded 업데이트 생략")
        return

    try:
        insert_chunks_to_supabase(payloads)

        article_ids = sorted(
            {
                payload["article_id"]
                for payload in payloads
                if payload.get("article_id") is not None
            }
        )

        mark_articles_embedded(article_ids)

        print("[임베더] 신규 기사 임베딩 처리 완료")

    except Exception as e:
        print(f"[임베더] Supabase 저장 실패. embedded 업데이트 안 함: {e}")
        raise


def embed_all_articles():
    total_round = 0

    while True:
        articles = load_unembedded_articles()

        if not articles:
            print("[임베더] 모든 기사 임베딩 완료")
            break

        total_round += 1
        print(f"\n[임베더] {total_round}번째 배치 처리 시작")

        documents = chunk_articles(articles)

        if not documents:
            print("[임베더] 생성된 청크 없음. 해당 배치 건너뜀")
            break

        documents = normalize_documents_metadata(documents, articles)

        print(f"[임베더] 신규 {len(documents)}개 청크 임베딩 중...")

        embeddings = get_embeddings()
        texts = [doc.page_content for doc in documents]
        vectors = embeddings.embed_documents(texts)

        payloads = build_chunk_payloads(documents, vectors)

        if not payloads:
            print("[임베더] 저장 가능한 payload 없음. 종료")
            break

        insert_chunks_to_supabase(payloads)

        article_ids = sorted({
            payload["article_id"]
            for payload in payloads
            if payload.get("article_id") is not None
        })

        mark_articles_embedded(article_ids)

    print("[임베더] 전체 임베딩 작업 종료")
    

if __name__ == "__main__":
    embed_all_articles()