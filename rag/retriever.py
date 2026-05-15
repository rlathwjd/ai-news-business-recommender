import os
import sys
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from datetime import datetime

sys.path.append(".")
from rag.embedder import load_chromadb

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"


INDUSTRY_PROMPT = """
당신은 AI 기술 동향을 분석하고 신규 사업 아이템을 추천하는 전문 애널리스트입니다.
아래 제공된 AI타임스 최신 기사 내용을 기반으로만 답변하세요.

사용자가 선택한 산업:
{industry}

답변 규칙:
1. 반드시 제공된 기사 내용을 근거로만 답변하세요.
2. 사용자가 선택한 산업인 "{industry}"과 관련성이 높은 기사 내용을 우선 분석하세요.
3. "{industry}" 산업에 적용 가능한 신규 AI 사업 아이템을 추천하세요.
4. 기사에 없는 내용은 "기사에서 확인되지 않았습니다"라고 답변하세요.

답변 형식:
- 사업명:
- 대상 산업:
- 관련 AI 트렌드:
- 근거 기사:
- 기대 효과:
- 추천 이유:

[참고 기사]
{context}

[질문]
{question}

[답변]
"""


TREND_PROMPT = """
당신은 AI 기술 동향을 분석하는 전문 애널리스트입니다.
아래 제공된 AI타임스 최신 기사 내용을 기반으로만 답변하세요.

답변 규칙:
1. 반드시 제공된 기사 내용을 근거로만 답변하세요.
2. 기사에 없는 내용은 "기사에서 확인되지 않았습니다"라고 답변하세요.
3. 최근 AI 기술 트렌드를 중심으로 분석하세요.
4. 반드시 아래 출력 형식을 그대로 따르세요.
5. "핵심 AI 트렌드 3가지" 같은 별도 제목은 출력하지 마세요.

답변 형식 예시:

1. 핵심 트렌드 1
   - 주요 내용 :
   - 근거 기사 :
   - 시장/산업적 의미 :


2. 핵심 트렌드 2
   - 주요 내용 :
   - 근거 기사 :
   - 시장/산업적 의미 :


3. 핵심 트렌드 3
   - 주요 내용 :
   - 근거 기사 :
   - 시장/산업적 의미 :


4. 종합 의견
   - 앞으로 주목할 포인트 :

[참고 기사]
{context}

[질문]
{question}

[답변]
"""


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
    if not docs:
        return "검색된 참고 기사가 없습니다."

    result = ""

    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "제목 없음")
        url = doc.metadata.get("url", "출처 없음")

        result += f"\n[기사 {i + 1}] {title}\n"
        result += f"{doc.page_content}\n"
        result += f"출처: {url}\n"

    return result


def get_retriever(k: int = 5):
    vectorstore = load_chromadb()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def filter_docs_by_date(docs, start_date, end_date):
    if not start_date or not end_date:
        return docs

    filtered = []

    for doc in docs:
        date_str = doc.metadata.get("published_date")

        if not date_str:
            filtered.append(doc)
            continue

        try:
            doc_date = datetime.strptime(
                date_str,
                "%Y-%m-%d %H:%M"
            ).date()

            if start_date <= doc_date <= end_date:
                filtered.append(doc)

        except Exception:
            filtered.append(doc)

    return filtered


def analyze_trend(start_date=None, end_date=None) -> str:
    retriever = get_retriever(k=5)

    docs = retriever.invoke(
        "최근 AI 기술 트렌드 생성형 AI AI 에이전트 자동화 로봇 클라우드 반도체"
    )

    # 기간 필터링
    docs = filter_docs_by_date(docs, start_date, end_date)
    context = format_docs(docs)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=TREND_PROMPT,
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    ).with_config({
        "run_name": "analyze_ai_trend",
        "tags": ["ai-news", "trend-analysis", "groq", "llama-3.3"]
    })

    return chain.invoke({
        "context": context,
        "question": "최근 AI 기술 트렌드를 분석해줘.",
    })


def recommend_by_industry(industry, start_date=None, end_date=None) -> str:
    retriever = get_retriever(k=5)

    # 검색에도 사용자가 선택한 산업을 그대로 사용
    search_query = f"{industry} AI 생성형 AI 자동화 디지털전환 신규 사업"

    docs = retriever.invoke(search_query)
    
    # 기간 필터링
    docs = filter_docs_by_date(docs, start_date, end_date)
    context = format_docs(docs)

    prompt = PromptTemplate(
        input_variables=["context", "industry", "question"],
        template=INDUSTRY_PROMPT,
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    ).with_config({
        "run_name": "recommend_ai_business_by_industry",
        "tags": ["ai-news", "industry-recommendation", "rag", "groq", "llama-3.3"],
        "metadata": {
            "industry": industry,
            "search_query": search_query,
        }
    })
    # 프롬프트에도 사용자가 선택한 산업을 그대로 주입
    return chain.invoke({
        "context": context,
        "industry": industry,
        "question": f"{industry} 산업에 적용 가능한 AI 기반 신규 사업 아이템을 추천해줘.",
    })


if __name__ == "__main__":
    print(analyze_trend())
    print(recommend_by_industry("금융 IT"))