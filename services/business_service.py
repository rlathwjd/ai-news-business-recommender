from rag.retriever import build_rag_chain

_chain = None

def get_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain

def generate_answer(question: str) -> str:
    chain = get_chain()
    return chain.invoke(question)

def analyze_trend() -> str:
    return generate_answer("최근 AI 기술 트렌드를 요약해줘")

def recommend_business() -> str:
    return generate_answer("LG CNS가 추진할 수 있는 AI 사업 아이템을 3가지 추천해줘")

def recommend_by_industry(industry: str) -> str:
    return generate_answer(f"{industry} 산업에서 적용 가능한 AI 사업 아이템을 추천해줘")