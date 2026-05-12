from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
from rag.chunker import load_unembedded_articles, chunk_articles
import sys
import sqlite3

import os
import warnings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

DB_PATH = "./data/articles.db"

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aitimes_ai_industry")

# 한국어 지원 잘 되는 다국어 임베딩 모델
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """HuggingFace 임베딩 모델 초기화 (로컬 실행)"""
    print(f"[임베더] 임베딩 모델 로딩 중... ({EMBEDDING_MODEL})")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def save_to_chromadb(documents: list[Document]) -> Chroma:
    """청크를 임베딩 후 ChromaDB에 저장"""
    print(f"[임베더] 신규 {len(documents)}개 청크 임베딩 중...")

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    before_count = vectorstore._collection.count()
    print(f"[확인] 저장 전 ChromaDB 청크 수: {before_count}")

    vectorstore.add_documents(documents)

    after_count = vectorstore._collection.count()
    print(f"[확인] 저장 후 ChromaDB 청크 수: {after_count}")

    if after_count <= before_count:
        raise RuntimeError("ChromaDB 저장 실패: 저장 후 count가 증가하지 않았습니다.")

    print("[임베더] 신규 문서 추가 완료")
    return vectorstore


def load_chromadb() -> Chroma:
    """저장된 ChromaDB 로드"""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    print(f"[임베더] ChromaDB 로드 완료 (컬렉션: {COLLECTION_NAME})")
    return vectorstore


def test_search(vectorstore: Chroma, query: str, k: int = 3):
    """검색 테스트"""
    print(f"\n[검색 테스트] 쿼리: '{query}'")


def mark_articles_embedded(article_ids: list[int], db_path: str = DB_PATH):
    if not article_ids:
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executemany(
        "UPDATE articles SET embedded = 1 WHERE id = ?",
        [(article_id,) for article_id in article_ids]
    )

    conn.commit()
    conn.close()

    print(f"[임베더] {len(article_ids)}개 기사 embedded=1 업데이트 완료")
    
    
if __name__ == "__main__":

    sys.path.append(".")

    articles = load_unembedded_articles()

    if not articles:
        print("[임베더] 임베딩할 신규 기사 없음")
    else:
        documents = chunk_articles(articles)

        if not documents:
            print("[임베더] 생성된 청크 없음. embedded 업데이트 생략")
        else:
            try:
                vectorstore = save_to_chromadb(documents)

                article_ids = list({article["id"] for article in articles})
                mark_articles_embedded(article_ids)

            except Exception as e:
                print(f"[임베더] ChromaDB 저장 실패. embedded 업데이트 안 함: {e}")