import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

LAST_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 데이터만 남김
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 연도 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.sort_values(
        "연도"
    ).reset_index(drop=True)

    return annual


annual = load_data()


# =========================================================
# 선형회귀
# =========================================================
x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# y = 기울기 × 연도 + 절편
slope, intercept = np.polyfit(
    x,
    y,
    1
)

# 회귀모델의 예측값
predicted = (
    slope * x + intercept
)

annual["회귀예측기온"] = predicted


# =========================================================
# 평가 지표
# =========================================================

# MAE
mae = np.mean(
    np.abs(y - predicted)
)

# MSE
mse = np.mean(
    (y - predicted) ** 2
)

# R²
ss_res = np.sum(
    (y - predicted) ** 2
)

ss_tot = np.sum(
    (y - np.mean(y)) ** 2
)

r2 = 1 - (
    ss_res / ss_tot
)

# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# =========================================================
# 제목
# =========================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용하여 "
    "전체 데이터에 대한 선형회귀 모델을 만들었습니다."
)


# =========================================================
# 전체 데이터 정보
# =========================================================
st.subheader("② 전체 데이터로 만든 회귀모델")

st.caption(
    f"사용 데이터: {int(annual['연도'].min())}~"
    f"{int(annual['연도'].max())}년 · "
    f"관측일수 300일 이상인 연도만 사용"
)


# =========================================================
# 성능 카드
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "연간 기온 변화",
        f"{slope:.2f} ℃",
        help="1년이 지날 때 회귀선이 얼마나 상승 또는 하락하는지 나타냅니다."
    )

with col2:
    st.metric(
        "상관계수 r",
        f"{correlation:.3f}",
        help="연도와 연평균기온 사이의 선형 관계를 나타냅니다."
    )

with col3:
    st.metric(
        "MAE",
        f"{mae:.3f} ℃",
        help="실제 기온과 회귀선의 예측값 사이의 평균 절대 오차입니다."
    )

with col4:
    st.metric(
        "MSE",
        f"{mse:.3f}",
        help="실제값과 예측값의 차이를 제곱하여 평균한 값입니다."
    )

with col5:
    st.metric(
        "R²",
        f"{r2:.3f}",
        help="회귀모델이 연평균기온의 변동을 얼마나 설명하는지 나타냅니다."
    )


# =========================================================
# 회귀식
# =========================================================
st.write(
    f"**회귀식:** "
    f"연평균기온 = "
    f"{slope:.5f} × 연도 "
    f"{intercept:+.3f}"
)


# =========================================================
# 산점도 + 회귀선
# =========================================================
st.subheader("📈 연도별 평균기온과 회귀선")

fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예측기온"],
        mode="lines",
        name="선형회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선 예측: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="구분"
)

fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 모델 해석
# =========================================================
st.subheader("📌 회귀모델 해석")

st.write(
    f"""
    이 회귀모델의 기울기는 **{slope:.5f} ℃/년**입니다.
    
    즉, 다른 조건을 고려하지 않고 단순히 선형 추세만 보면 
    연도가 1년 증가할 때 연평균기온이 약 **{abs(slope):.5f} ℃**
    {'상승' if slope >= 0 else '하락'}하는 추세로 나타납니다.
    
    상관계수는 **{correlation:.3f}**, 
    R²는 **{r2:.3f}**입니다.
    
    회귀선의 평균적인 예측 오차를 나타내는 MAE는 
    **{mae:.3f} ℃**이고, MSE는 **{mse:.3f}**입니다.
    """
)


# =========================================================
# 연도 예측
# =========================================================
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 예측값
selected_prediction = (
    slope * selected_year + intercept
)


# =========================================================
# 큰 예측값 표시
# =========================================================
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 20px;
        margin-bottom: 20px;
    ">

        <div style="
            font-size: 24px;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 60px;
            font-weight: bold;
        ">
            {selected_prediction:.2f} ℃
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 실제 데이터 범위 밖의 연도 안내
# =========================================================
if (
    selected_year < int(annual["연도"].min())
    or selected_year > int(annual["연도"].max())
):
    st.warning(
        f"{selected_year}년은 실제 회귀에 사용한 데이터 범위 "
        f"({int(annual['연도'].min())}~"
        f"{int(annual['연도'].max())}년) 밖입니다. "
        "따라서 회귀선을 연장하여 계산한 추정값입니다."
    )


# =========================================================
# 데이터 사용 정보
# =========================================================
st.subheader("📋 분석에 사용한 데이터")

c1, c2, c3 = st.columns(3)

c1.metric(
    "사용 연도 수",
    f"{len(annual)}개"
)

c2.metric(
    "시작 연도",
    f"{int(annual['연도'].min())}년"
)

c3.metric(
    "끝 연도",
    f"{int(annual['연도'].max())}년"
)
