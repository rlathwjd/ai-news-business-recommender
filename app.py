import streamlit as st
from services.business_service import (
    analyze_trend,
    recommend_business,
    recommend_by_industry,
)

st.set_page_config(page_title="AI Business Recommender", layout="wide")

st.title("🧠 AI 뉴스 기반 사업 추천 시스템")
st.markdown("최신 AI 뉴스 분석을 통해 LG CNS 신규 사업 아이템을 추천합니다")

mode = st.selectbox(
    "기능 선택",
    ["트렌드 분석", "사업 추천", "산업별 추천"]
)

industry = ""
if mode == "산업별 추천":
    industry = st.text_input("산업 입력 (예: 금융, 제조, 공공, 물류)")

if st.button("실행하기"):
    with st.spinner("분석 중..."):
        if mode == "트렌드 분석":
            answer = analyze_trend()

        elif mode == "사업 추천":
            answer = recommend_business()

        elif mode == "산업별 추천":
            if not industry:
                st.warning("산업을 입력해주세요!")
                st.stop()
            answer = recommend_by_industry(industry)

    st.markdown("### 📊 결과")
    st.success(answer)