from rag.retriever import (
    analyze_trend as rag_analyze_trend,
    recommend_by_industry as rag_recommend_by_industry,
)

def analyze_trend(start_date, end_date) -> str:
    return rag_analyze_trend(start_date, end_date)


def recommend_by_industry(industry, start_date, end_date) -> str:
    return rag_recommend_by_industry(industry, start_date, end_date)