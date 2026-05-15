# AI 뉴스 기반 사업 추천 시스템

## 개요
최신 AI 뉴스 데이터를 기반으로 기업의 신규 사업 아이템을 추천하는 RAG 기반 시스템

## 주요 기능
- 최신 AI 뉴스 기반 AI 트렌드 분석
- 산업별 AI 기반 신규 사업 아이템 추천
- LangSmith를 활용한 LLM 실행 흐름 모니터링
- LangGraph 기반 RAG 처리 흐름 시각화

## 기술 스택

#### 서비스 구현

- Python
- Streamlit
- Streamlit Community Cloud

#### 데이터 수집

- Selenium
- BeautifulSoup

#### RAG / LLM

- LangChain
- ChromaDB
- Groq API
- Llama 3.3

#### 모니터링 / 시각화

- LangSmith
- LangGraph
- LangSmith Studio


## 환경 변수 설정
- GROQ_API_KEY=your_groq_api_key
- LANGSMITH_TRACING=true
- LANGSMITH_ENDPOINT=https://api.smith.langchain.com
- LANGSMITH_API_KEY=your_langsmith_api_key
- LANGSMITH_PROJECT=project_name

  


