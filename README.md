# AI 뉴스 기반 사업 추천 시스템

최신 AI 뉴스 데이터를 기반으로 기업의 신규 사업 아이템을 추천하는 RAG 기반 시스템

https://ai-news-business-recommender-ezuhhysyrrg36xjy6wjkfq.streamlit.app/

## 1. 주요 기능
- 최신 AI 뉴스 기반 AI 트렌드 분석
- 산업별 AI 기반 신규 사업 아이템 추천
- LangSmith를 활용한 LLM 실행 흐름 모니터링

## 2. 기술 스택

### 서비스 구현
- Python
- Streamlit
- Supabase

### 데이터 수집
- Selenium
- BeautifulSoup

### RAG / LLM
- LangChain
- Groq API
- Llama 3.3

### 모니터링
- LangSmith

## 3. 환경 변수 설정 (.env 파일에 추가)
- GROQ_API_KEY=api_key
- NEXT_PUBLIC_SUPABASE_URL=supabase_url
- SUPABASE_SERVICE_ROLE_KEY=supbase_service_role_key
- LANGSMITH_TRACING=true
- LAGNSMITH_API_KEY=langsmith_api_key
- LANGSMITH_PROJECT=project_name
- Streamlit Sharing 접속 (https://share.streamlit.io/)
- My apps에서 프로젝트 Settings-Secrets에 위 6가지 따옴표로 묶어서 입력
- ex) GROQ_API_KEY="groq_api_key"

## 4. 로컬 개발 및 테스트
- python -m venv venv (가상환경 생성)
- venv/Scripts/activate (가상환경 활성화)
- python -m pip install -r requirements.txt (필요 패키지 설치)
- python -m crawler.crawl (크롤링)
- python -m rag.embedder (임베딩)
- python -m streamlit run app.py

## 5. LangSmith 연동
- python -m venv venv-studio (LangSmith Studio 전용 가상환경 생성)
- venv-studio/Scripts/activate (가상환경 활성화)
- python -m pip install -r dev-requirements.txt (LangSmith Studio 전용 필요 패키지 설치)
- langgraph dev (LangGraph 서버 실행)
- LangSmith 접속 (https://smith.langchain.com/)
- Studio 메뉴에서 연결 추가 후 Base URL 입력 (http://127.0.0.1:2024)

## 6. 데이터 업데이트 (Github Actions)
- 자동 실행 : 매주 월요일 오전 9시 AI타임스 최신 기사 크롤링 및 임베딩
- 수동 실행 : GitHub Actions의 'Run workflow' 버튼 클릭 