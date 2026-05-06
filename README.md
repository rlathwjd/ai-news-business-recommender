# AI 뉴스 기반 사업 추천 시스템

## 개요
최신 AI 뉴스 데이터를 기반으로 기업의 신규 사업 아이템을 추천하는 RAG 기반 시스템

## 주요 기능
- AI 트렌드 분석
- 산업별 사업 아이템 추천

## 기술 스택
- Python, Streamlit
- LangChain, ChromaDB
- Groq API

## 실행 방법(터미널)
- .env 파일에 GROQ_API_KEY="api_key" 입력
- py -3.12 -m venv venv (파이썬 3.12 설치 후 가상환경 생성)
- venv/Scripts/activate (가상환경 활성화)
- python -m pip install -r requirements.txt (필요 패키지 설치)
- python crawler/aitimes_crawler.py (크롤링)
- python rag/embedder.py (임베딩)
- python -m streamlit run app.py