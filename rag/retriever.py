"""
RAG 체인 모듈 (Groq API 버전)
검색된 청크 + Groq LLM으로 최종 답변 생성
"""

import os
import sys
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

sys.path.append(".")
from rag.embedder import load_chromadb

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq 지원 모델 예시
LLM_MODEL = "llama-3.3-70b-versatile"
# 빠른/가벼운 모델을 쓰고 싶으면:
# LLM_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """당신은 AI 기술 동향을 분석하고 LG CNS의 신규 사업 아이템을 추천하는 전문 애널리스트입니다.
아래 제공된 AI타임스 최신 기사를 기반으로만 답변하세요.

LG CNS 사업 영역:
- 공공 SI (정부, 공공기관 시스템 구축)
- 금융 IT (은행, 보험, 증권 시스템)
- 제조 DX (스마트팩토리, 생산 자동화)
- 클라우드 MSP (AWS, Azure, GCP 운영 관리)
- 물류/유통 IT

답변 규칙:
1. 반드시 제공된 기사 내용을 근거로만 답변하세요
2. 기사에 없는 내용은 "기사에서 확인되지 않았습니다"라고 하세요
3. 사업 추천 시 아래 형식을 따르세요
   - 사업명:
   - 관련 AI 트렌드:
   - LG CNS 적용 포인트:
   - 근거 기사:

[참고 기사]
{context}

[질문]
{question}

[답변]"""


def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY가 없습니다. .env 파일에 GROQ_API_KEY를 설정하세요.")

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )


def format_docs(docs) -> str:
    """검색된 청크를 프롬프트용 텍스트로 변환"""
    if not docs:
        return "검색된 참고 기사가 없습니다."

    result = ""
    for i, doc in enumerate(docs):
        result += f"\n[기사 {i+1}] {doc.metadata.get('title', '')}\n"
        result += f"{doc.page_content}\n"
        result += f"출처: {doc.metadata.get('url', '')}\n"
    return result


def build_rag_chain():
    """RAG 체인 구성"""
    print("[리트리버] RAG 체인 구성 중...")

    vectorstore = load_chromadb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT,
    )

    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[리트리버] RAG 체인 준비 완료!")
    return chain


def ask(chain, question: str) -> str:
    """질문 입력 → 답변 반환"""
    print(f"\n[질문] {question}")
    print("[답변] 생성 중...\n")

    answer = chain.invoke(question)
    print(answer)
    return answer


if __name__ == "__main__":
    chain = build_rag_chain()

    ask(chain, "최근 가장 주목받는 AI 기술 트렌드가 뭐야?")
    ask(chain, "LG CNS가 추진할 만한 AI 사업 아이템 3가지 추천해줘")