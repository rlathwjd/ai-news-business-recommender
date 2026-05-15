# AI 뉴스 기반 사업 추천 시스템

## 개요
최신 AI 뉴스 데이터를 기반으로 기업의 신규 사업 아이템을 추천하는 RAG 기반 시스템

## 주요 기능
- AI 트렌드 분석
- 산업별 사업 아이템 추천

## 기술 스택
- Python, Streamlit(Streamlit Community Cloud 서버에 배포)
- Selenium, BeautifulSoup
- LangChain, ChromaDB
- Groq API(Llama 3.3)

## 환경 변수 설정
- .env 파일에 GROQ_API_KEY="api_key" 입력

## 실행 방법
- python -m venv venv (가상환경 생성)
- venv/Scripts/activate (가상환경 활성화)
- python -m pip install -r requirements.txt (필요 패키지 설치)
- python -m crawler.crawl (크롤링)
- python -m rag.embedder (임베딩)
- python -m streamlit run app.py

- python -m venv venv-studio(LangSmith Studio 전용 가상환경 생성)
- venv-studio/Scripts/activate (가상환경 활성화)
- python -m pip install -r dev-requirements.txt (LangSmith Studio 전용 필요 패키지 설치)
- langgraph dev (LangGraph 서버 실행)
- LangSmith(https://smith.langchain.com/) 접속
- Studio 메뉴에서 연결 추가 후 Base URL 입력 (http://127.0.0.1:2024)


## 사이트 접속
https://ai-news-business-recommender-ezuhhysyrrg36xjy6wjkfq.streamlit.app/
