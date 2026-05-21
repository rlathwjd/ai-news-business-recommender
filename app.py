import os
import warnings
from datetime import date, timedelta
from dotenv import load_dotenv

# 1. 로컬 실행용 .env 로드
load_dotenv()

# 2. Streamlit Cloud Secrets가 있으면 환경변수에 주입
SECRET_KEYS = [
    "GROQ_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "NEXT_PUBLIC_SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
]

for key in SECRET_KEYS:
    try:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(page_title="AI Business Recommender", layout="wide")


        
from services.business_service import (
    analyze_trend,
    recommend_by_industry,
)

st.title("🤖 AI 뉴스 기반 사업 추천 시스템")
st.markdown("최신 AI 뉴스 분석을 통해 트렌드를 분석하고 신규 사업 아이템을 추천합니다.")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🔍 기능 선택")
    mode = st.selectbox(
        "분석 옵션",
        ["AI 트렌드 분석", "산업별 사업 추천"]
    )

    industry = ""

    if mode == "산업별 사업 추천":
        industry = st.selectbox(
            "산업 선택",
            ["공공 SI", "금융", "제조", "물류/유통"]
        )

with col_right:
    st.markdown("### 🗓️ 기간 설정")

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "시작일",
            value=date.today() - timedelta(days=7)
        )

    with date_col2:
        end_date = st.date_input(
            "종료일",
            value=date.today()
        )

st.divider()

run_button = st.button(
    "🚀 분석 실행",
    use_container_width=True
)

if run_button:
    if start_date > end_date:
        st.warning("시작일은 종료일보다 늦을 수 없습니다.")
        st.stop()

    with st.spinner("AI가 뉴스를 분석 중입니다..."):
        if mode == "AI 트렌드 분석":
            answer = analyze_trend(start_date, end_date)
        else:
            answer = recommend_by_industry(industry, start_date, end_date)

    st.markdown("## 📊 분석 결과")
    st.markdown(answer)