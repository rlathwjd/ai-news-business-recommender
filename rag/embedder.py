"""
임베딩 + ChromaDB 저장 모듈 (HuggingFace 버전)
임베딩: sentence-transformers (로컬)
LLM: HuggingFace Inference API
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

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
    print(f"[임베더] {len(documents)}개 청크 임베딩 중... (첫 실행은 모델 다운로드로 시간 걸려요)")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    print(f"  → ChromaDB 저장 완료 ({CHROMA_DB_PATH})")
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
    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\n--- 결과 {i+1} ---")
        print(f"제목: {doc.metadata['title']}")
        print(f"내용: {doc.page_content[:150]}...")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from rag.chunker import load_articles, chunk_articles

    # 1. 기사 로드 + 청킹
    articles = load_articles()
    documents = chunk_articles(articles)

    # 2. ChromaDB 저장
    vectorstore = save_to_chromadb(documents)

    # 3. 검색 테스트
    test_search(vectorstore, "LG CNS AI 사업")
    test_search(vectorstore, "최근 AI 트렌드")