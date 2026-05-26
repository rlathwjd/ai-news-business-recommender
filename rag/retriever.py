import os
import sys
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END

sys.path.append(".")

from db.supabase_client import get_supabase

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# AI 트렌드
TREND_PROMPT = """
당신은 AI 기술 동향을 분석하는 전문 애널리스트입니다.
아래 제공된 AI타임스 기사 내용만 근거로 답변하세요.

답변 규칙:
1. 반드시 제공된 기사 내용만 근거로 답변하세요.
2. 기사에 없는 내용은 추측하지 말고 "기사에서 확인되지 않았습니다"라고 답변하세요.
3. 핵심 트렌드는 반드시 3개만 작성하세요.
4. 각 트렌드는 2~3문장 이내로 간결하게 작성하세요.
5. 근거 기사는 기사 제목과 URL을 함께 제시하세요.
6. 아래 출력 형식을 반드시 그대로 따르세요.
7. 마지막 "종합 의견"까지 반드시 완성하세요.

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

# 사업 추천
INDUSTRY_PROMPT = """
당신은 AI 기술 동향을 분석하고 신규 사업 아이템을 추천하는 전문 애널리스트입니다.
아래 제공된 AI타임스 기사 내용만 근거로 답변하세요.

사용자가 선택한 산업:
{industry}

답변 규칙:
1. 반드시 제공된 기사 내용만 근거로 답변하세요.
2. "{industry}" 산업과 관련성이 높은 내용만 우선 활용하세요.
3. 신규 AI 사업 아이템은 1개만 추천하세요.
4. 각 항목은 2~3문장 이내로 간결하게 작성하세요.
5. 기사에 없는 내용은 "기사에서 확인되지 않았습니다"라고 답변하세요.
6. 근거 기사는 기사 제목과 URL을 함께 제시하세요.
7. 아래 출력 형식을 반드시 그대로 따르세요.

답변 형식:
- 사업명:
- 대상 산업:
- 관련 AI 트렌드:
- 적용 포인트:
- 근거 기사:
- 기대 효과:
- 추천 이유:

[참고 기사]
{context}

[질문]
{question}

[답변]
"""


class NewsState(TypedDict):
    mode: Literal["trend", "industry"]
    industry: Optional[str]
    search_query: str
    question: str
    chunks: list[dict]
    context: str
    answer: str


def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY가 없습니다. .env 파일에 GROQ_API_KEY를 설정하세요.")

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_tokens=2048,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def search_article_chunks(query: str, k: int = 5) -> list[dict]:
    """
    Supabase pgvector RPC를 사용해 유사 기사 청크를 검색합니다.
    """
    supabase = get_supabase()
    embeddings = get_embeddings()

    query_embedding = embeddings.embed_query(query)

    result = (
        supabase.rpc(
            "match_article_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": k,
            },
        )
        .execute()
    )

    return result.data or []


def format_chunks(chunks: list[dict]) -> str:
    """
    Supabase article_chunks 검색 결과를 LLM context 형식으로 변환합니다.
    """
    if not chunks:
        return "검색된 참고 기사가 없습니다."

    result = ""

    for i, chunk in enumerate(chunks):
        title = chunk.get("title", "제목 없음")
        url = chunk.get("url", "출처 없음")
        content = chunk.get("content", "")
        similarity = chunk.get("similarity")

        result += f"\n[기사 {i + 1}] {title}\n"
        result += f"{content}\n"
        result += f"출처: {url}\n"

        if similarity is not None:
            result += f"유사도: {similarity:.4f}\n"

    return result


def build_query(state: NewsState) -> NewsState:
    mode = state["mode"]
    industry = state.get("industry")

    if mode == "trend":
        search_query = (
        "최신 AI 기술 트렌드 생성형 AI AI 에이전트 멀티모달 LLM "
        "자동화 로봇 반도체 온디바이스 AI 기업 AI 도입 "
        "AI 서비스 산업 변화 기술 동향"
    )
        question = "최근 AI 기술 트렌드를 분석해줘."

    else:
        if not industry:
            raise ValueError("산업별 사업 추천에는 industry 값이 필요합니다.")

        search_query = f"{industry} AI 활용 사례 생성형 AI 에이전트 디지털전환"
        question = f"{industry} 산업에 적용 가능한 AI 기반 신규 사업 아이템을 추천해줘."

    return {
        **state,
        "search_query": search_query,
        "question": question,
    }


def retrieve_chunks(state: NewsState) -> NewsState:
    chunks = search_article_chunks(state["search_query"], k=5)

    return {
        **state,
        "chunks": chunks,
    }


def format_context(state: NewsState) -> NewsState:
    context = format_chunks(state["chunks"])

    return {
        **state,
        "context": context,
    }


def generate_answer(state: NewsState) -> NewsState:
    mode = state["mode"]

    if mode == "trend":
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=TREND_PROMPT,
        )

        chain = (
            prompt
            | get_llm()
            | StrOutputParser()
        ).with_config({
            "run_name": "generate_ai_trend_analysis",
            "tags": ["ai-news", "trend-analysis", "supabase", "pgvector", "groq", "llama-3.3"],
            "metadata": {
                "mode": mode,
                "search_query": state["search_query"],
            },
        })

        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"],
        })

    else:
        prompt = PromptTemplate(
            input_variables=["context", "industry", "question"],
            template=INDUSTRY_PROMPT,
        )

        chain = (
            prompt
            | get_llm()
            | StrOutputParser()
        ).with_config({
            "run_name": "generate_ai_business_recommendation",
            "tags": ["ai-news", "industry-recommendation", "supabase", "pgvector", "groq", "llama-3.3"],
            "metadata": {
                "mode": mode,
                "industry": state.get("industry"),
                "search_query": state["search_query"],
            },
        })

        answer = chain.invoke({
            "context": state["context"],
            "industry": state["industry"],
            "question": state["question"],
        })

    return {
        **state,
        "answer": answer,
    }


def build_news_graph():
    graph = StateGraph(NewsState)

    graph.add_node("build_query", build_query)
    graph.add_node("retrieve_chunks_from_supabase", retrieve_chunks)
    graph.add_node("format_context", format_context)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("build_query")

    graph.add_edge("build_query", "retrieve_chunks_from_supabase")
    graph.add_edge("retrieve_chunks_from_supabase", "format_context")
    graph.add_edge("format_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


news_graph = build_news_graph()


def analyze_trend(start_date=None, end_date=None) -> str:
    """
    Streamlit에서 호출하는 AI 트렌드 분석 함수.
    start_date, end_date는 기존 app.py 호환을 위해 유지하지만,
    현재 Supabase pgvector 검색에서는 사용하지 않습니다.
    """
    result = news_graph.invoke(
        {
            "mode": "trend",
            "industry": None,
            "search_query": "",
            "question": "",
            "chunks": [],
            "context": "",
            "answer": "",
        },
        config={
            "run_name": "ai_trend_analysis_graph",
            "tags": ["ai-news", "trend-analysis", "langgraph", "supabase"],
        },
    )

    return result["answer"]


def recommend_by_industry(industry, start_date=None, end_date=None) -> str:
    """
    Streamlit에서 호출하는 산업별 사업 추천 함수.
    start_date, end_date는 기존 app.py 호환을 위해 유지하지만,
    현재 Supabase pgvector 검색에서는 사용하지 않습니다.
    """
    result = news_graph.invoke(
        {
            "mode": "industry",
            "industry": industry,
            "search_query": "",
            "question": "",
            "chunks": [],
            "context": "",
            "answer": "",
        },
        config={
            "run_name": "ai_business_recommendation_graph",
            "tags": ["ai-news", "industry-recommendation", "langgraph", "supabase"],
            "metadata": {
                "industry": industry,
            },
        },
    )

    return result["answer"]


if __name__ == "__main__":
    print(analyze_trend())
    print(recommend_by_industry("금융"))