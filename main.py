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

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 연평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수 300일 미만인 연도 제거
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.sort_values(
        "연도"
    ).reset_index(drop=True)

    return annual


annual = load_data()


# =========================================================
# 선형회귀 함수
# =========================================================
def linear_regression(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # y = 기울기 × x + 절편
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


# =========================================================
# 예측 함수
# =========================================================
def predict(x, slope, intercept):

    x = np.asarray(
        x,
        dtype=float
    )

    return slope * x + intercept


# =========================================================
# 평가 지표
# =========================================================
def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    # MAE
    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    # MSE
    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    # R²
    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# =========================================================
# 전체 데이터 회귀
# =========================================================
# 연도 자체를 독립변수로 사용
x_all = annual["연도"]
y_all = annual["연평균기온"]

full_slope, full_intercept = linear_regression(
    x_all,
    y_all
)

annual["전체회귀예측"] = predict(
    x_all,
    full_slope,
    full_intercept
)


# =========================================================
# 화면 제목
# =========================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 이용하여 선형회귀 모델을 만들고, "
    "과거 기간으로 학습한 모델이 최근 20년의 기온을 "
    "얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 전체 데이터 정보
# =========================================================
st.subheader("📊 전체 데이터")

c1, c2, c3, c4 = st.columns(4)

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

c4.metric(
    "전체 회귀 기울기",
    f"{full_slope:.5f} ℃/년"
)

st.caption(
    "2025년 이후 데이터와 연평균기온 관측일수가 "
    "300일 미만인 연도는 분석에서 제외했습니다."
)


# =========================================================
# 전체 데이터 산점도 + 회귀선
# =========================================================
st.subheader("📈 전체 연평균기온과 선형회귀선")

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["전체회귀예측"],
        mode="lines",
        name="전체 데이터 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

fig_all.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)


# =========================================================
# 훈련 / 테스트 데이터 분리
# =========================================================

# -----------------------------------------
# 최근 50년 훈련 데이터
# 1956~2005
# -----------------------------------------
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# -----------------------------------------
# 최근 100년 훈련 데이터
# 1906~2005
#
# 실제 데이터가 1906년에 없으면
# 존재하는 데이터만 자동 사용
# -----------------------------------------
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# -----------------------------------------
# 공통 테스트 데이터
# 2006~2025
# -----------------------------------------
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# =========================================================
# 데이터 분할 표시
# =========================================================
st.subheader("🧪 훈련 데이터와 테스트 데이터")

st.write(
    "두 모델의 성능을 공정하게 비교하기 위해 "
    "**2006~2025년을 공통 테스트 데이터**로 사용합니다."
)

data_split = pd.DataFrame({
    "구분": [
        "50년 훈련 데이터",
        "100년 훈련 데이터",
        "공통 테스트 데이터"
    ],
    "기간": [
        "1956~2005",
        "1906~2005",
        "2006~2025"
    ],
    "실제 사용 연도 수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    data_split,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 50년 회귀 모델
# =========================================================
slope_50, intercept_50 = linear_regression(
    train_50["연도"],
    train_50["연평균기온"]
)

pred_50 = predict(
    test["연도"],
    slope_50,
    intercept_50
)


# =========================================================
# 100년 회귀 모델
# =========================================================
slope_100, intercept_100 = linear_regression(
    train_100["연도"],
    train_100["연평균기온"]
)

pred_100 = predict(
    test["연도"],
    slope_100,
    intercept_100
)


# =========================================================
# 모델 평가
# =========================================================
mae_50, mse_50, r2_50 = calculate_metrics(
    test["연평균기온"],
    pred_50
)

mae_100, mse_100, r2_100 = calculate_metrics(
    test["연평균기온"],
    pred_100
)


# =========================================================
# 회귀선 기울기 비교
# =========================================================
st.subheader("📐 50년 vs 100년 회귀선 기울기")

c1, c2, c3 = st.columns(3)

c1.metric(
    "50년 학습 기울기",
    f"{slope_50:.5f} ℃/년"
)

c2.metric(
    "100년 학습 기울기",
    f"{slope_100:.5f} ℃/년"
)

c3.metric(
    "기울기 차이",
    f"{slope_50 - slope_100:+.5f} ℃/년"
)

st.write(
    f"""
    **50년 회귀식:**  
    연평균기온 = {slope_50:.5f} × 연도 + {intercept_50:.3f}

    **100년 회귀식:**  
    연평균기온 = {slope_100:.5f} × 연도 + {intercept_100:.3f}
    """
)


# =========================================================
# 두 회귀선 비교 그래프
# =========================================================
st.subheader("📈 50년 학습 회귀선과 100년 학습 회귀선")

line_years = np.arange(
    int(annual["연도"].min()),
    2026
)

line_pred_50 = predict(
    line_years,
    slope_50,
    intercept_50
)

line_pred_100 = predict(
    line_years,
    slope_100,
    intercept_100
)

fig_models = go.Figure()

# 실제 연평균기온
fig_models.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온"
    )
)

# 50년 회귀선
fig_models.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred_50,
        mode="lines",
        name="1956~2005 학습"
    )
)

# 100년 회귀선
fig_models.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred_100,
        mode="lines",
        name="1906~2005 학습"
    )
)

# 테스트 기간 표시
fig_models.add_vrect(
    x0=2006,
    x1=2025,
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간"
)

fig_models.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

fig_models.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_models,
    use_container_width=True
)


# =========================================================
# MAE / MSE / R² 성능 비교
# =========================================================
st.subheader("📊 2006~2025년 테스트 성능 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "훈련 연도 수": [
        len(train_50),
        len(train_100)
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "MAE (℃)": "{:.4f}",
        "MSE (℃²)": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 성능 지표 설명
# =========================================================
st.markdown(
    """
    **평가 지표 해석**

    - **MAE**: 실제 기온과 예측 기온의 평균적인 절대 오차
      → **작을수록 좋음**
    - **MSE**: 오차를 제곱해서 평균한 값
      → **작을수록 좋음**
    - **R²**: 모델이 테스트 데이터의 변동을 얼마나 설명하는지 나타내는 값
      → **높을수록 좋음**
    """
)


# =========================================================
# 실제값 vs 예측값
# =========================================================
st.subheader("🔎 공통 테스트 데이터의 실제값과 예측값")

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온"
    )
)

# 50년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

# 100년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 예측"
    )
)

fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)

fig_test.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 자동 비교 결과
# =========================================================
st.subheader("💡 50년 학습과 100년 학습 비교")

if mae_50 < mae_100:
    mae_winner = "50년 학습"
elif mae_50 > mae_100:
    mae_winner = "100년 학습"
else:
    mae_winner = "동일"

if mse_50 < mse_100:
    mse_winner = "50년 학습"
elif mse_50 > mse_100:
    mse_winner = "100년 학습"
else:
    mse_winner = "동일"

if r2_50 > r2_100:
    r2_winner = "50년 학습"
elif r2_50 < r2_100:
    r2_winner = "100년 학습"
else:
    r2_winner = "동일"

comparison_result = pd.DataFrame({
    "비교 항목": [
        "기울기",
        "MAE",
        "MSE",
        "R²"
    ],
    "50년 학습": [
        f"{slope_50:.5f} ℃/년",
        f"{mae_50:.4f} ℃",
        f"{mse_50:.4f}",
        f"{r2_50:.4f}"
    ],
    "100년 학습": [
        f"{slope_100:.5f} ℃/년",
        f"{mae_100:.4f} ℃",
        f"{mse_100:.4f}",
        f"{r2_100:.4f}"
    ],
    "더 좋은 결과": [
        "기울기 자체는 성능의 좋고 나쁨을 판단하지 않음",
        mae_winner,
        mse_winner,
        r2_winner
    ]
})

st.dataframe(
    comparison_result,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 기존 연도 예측 기능
# =========================================================
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temperature = predict(
    selected_year,
    full_slope,
    full_intercept
)

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 15px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px;">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 60px;
            font-weight: bold;
            margin-top: 5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if (
    selected_year < int(annual["연도"].min())
    or selected_year > int(annual["연도"].max())
):
    st.warning(
        f"{selected_year}년은 실제 데이터 범위 "
        f"({int(annual['연도'].min())}~"
        f"{int(annual['연도'].max())}년) "
        "밖이므로 회귀선을 연장한 추정값입니다."
    )
