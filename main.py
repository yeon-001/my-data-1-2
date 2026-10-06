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

BASE_YEAR = 1908
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

    # 필요한 데이터의 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온과 관측일수 계산
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

    # 1908년부터 지난 연수
    annual["지난연수"] = (
        annual["연도"] - BASE_YEAR
    )

    annual = annual.sort_values(
        "연도"
    ).reset_index(drop=True)

    return annual


annual = load_data()


# =========================================================
# 선형회귀 함수
# =========================================================
def make_regression(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # y = slope * x + intercept
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


# =========================================================
# 예측 함수
# =========================================================
def predict_temperature(
    x,
    slope,
    intercept
):

    x = np.asarray(
        x,
        dtype=float
    )

    return slope * x + intercept


# =========================================================
# 평가 지표 함수
# =========================================================
def calculate_metrics(
    y_true,
    y_pred
):

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
        np.abs(
            y_true - y_pred
        )
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

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (
            ss_res / ss_tot
        )

    return mae, mse, r2


# =========================================================
# 제목
# =========================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 이용하여 선형회귀 모델을 만들고, "
    "과거 기간으로 학습한 모델이 최근 20년의 기온을 "
    "얼마나 잘 예측하는지 평가합니다."
)


# =========================================================
# 데이터 사용 범위
# =========================================================
st.subheader("📋 분석 데이터")

data_col1, data_col2, data_col3, data_col4 = st.columns(4)

data_col1.metric(
    "사용한 연도 수",
    f"{len(annual)}개"
)

data_col2.metric(
    "시작 연도",
    f"{int(annual['연도'].min())}년"
)

data_col3.metric(
    "끝 연도",
    f"{int(annual['연도'].max())}년"
)

data_col4.metric(
    "기준 연도",
    f"{BASE_YEAR}년"
)

st.caption(
    "2025년 이후 데이터와 관측일수가 300일 미만인 연도는 제외했습니다."
)


# =========================================================
# ① 전체 데이터 회귀모델
# =========================================================
st.subheader("① 전체 데이터로 만든 회귀모델")

# 전체 데이터 회귀
x_all = annual["지난연수"]
y_all = annual["연평균기온"]

full_slope, full_intercept = make_regression(
    x_all,
    y_all
)

full_prediction = predict_temperature(
    x_all,
    full_slope,
    full_intercept
)

annual["전체회귀예측"] = full_prediction


# 전체 데이터 평가
full_mae, full_mse, full_r2 = calculate_metrics(
    y_all,
    full_prediction
)

full_correlation = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# =========================================================
# 전체 모델 카드
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "연간 기온 변화",
    f"{full_slope:.2f} ℃"
)

c2.metric(
    "상관계수 r",
    f"{full_correlation:.3f}"
)

c3.metric(
    "MAE",
    f"{full_mae:.3f} ℃"
)

c4.metric(
    "MSE",
    f"{full_mse:.3f}"
)

c5.metric(
    "R²",
    f"{full_r2:.3f}"
)

st.write(
    f"**전체 회귀식:** "
    f"연평균기온 = "
    f"{full_slope:.5f} × (연도 - {BASE_YEAR}) "
    f"{full_intercept:+.3f}"
)


# =========================================================
# 전체 데이터 산점도 + 회귀선
# =========================================================
st.subheader("📈 전체 연평균기온과 회귀선")

fig_full = go.Figure()

fig_full.add_trace(
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

fig_full.add_trace(
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

fig_full.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

fig_full.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_full,
    use_container_width=True
)


# =========================================================
# ② 훈련 / 테스트 데이터 분리
# =========================================================
st.subheader("② 훈련 데이터와 테스트 데이터")

st.write(
    "과거 데이터를 이용해 회귀모델을 학습한 뒤, "
    "두 모델 모두 동일한 2006~2025년 데이터를 테스트 데이터로 사용합니다."
)


# ---------------------------------------------------------
# 최근 50년 훈련 데이터
# 1956~2005
# ---------------------------------------------------------
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# ---------------------------------------------------------
# 최근 100년 훈련 데이터
# 1906~2005
# ---------------------------------------------------------
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# ---------------------------------------------------------
# 공통 테스트 데이터
# 2006~2025
# ---------------------------------------------------------
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# =========================================================
# 데이터 분할 정보
# =========================================================
split_table = pd.DataFrame({
    "구분": [
        "최근 50년 훈련",
        "최근 100년 훈련",
        "공통 테스트"
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
    split_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ③ 최근 50년 모델 학습
# =========================================================
slope_50, intercept_50 = make_regression(
    train_50["지난연수"],
    train_50["연평균기온"]
)

test_prediction_50 = predict_temperature(
    test["지난연수"],
    slope_50,
    intercept_50
)

mae_50, mse_50, r2_50 = calculate_metrics(
    test["연평균기온"],
    test_prediction_50
)


# =========================================================
# ④ 최근 100년 모델 학습
# =========================================================
slope_100, intercept_100 = make_regression(
    train_100["지난연수"],
    train_100["연평균기온"]
)

test_prediction_100 = predict_temperature(
    test["지난연수"],
    slope_100,
    intercept_100
)

mae_100, mse_100, r2_100 = calculate_metrics(
    test["연평균기온"],
    test_prediction_100
)


# =========================================================
# ⑤ 50년 / 100년 회귀선 기울기 비교
# =========================================================
st.subheader("③ 50년 학습과 100년 학습의 회귀선 비교")

slope_col1, slope_col2, slope_col3 = st.columns(3)

slope_col1.metric(
    "50년 학습 기울기",
    f"{slope_50:.5f} ℃/년"
)

slope_col2.metric(
    "100년 학습 기울기",
    f"{slope_100:.5f} ℃/년"
)

slope_col3.metric(
    "기울기 차이",
    f"{slope_50 - slope_100:+.5f} ℃/년"
)


st.write(
    f"""
    **50년 회귀식:**  
    연평균기온 = {slope_50:.5f} × (연도 - {BASE_YEAR}) {intercept_50:+.3f}

    **100년 회귀식:**  
    연평균기온 = {slope_100:.5f} × (연도 - {BASE_YEAR}) {intercept_100:+.3f}
    """
)


# =========================================================
# ⑥ 50년 / 100년 회귀선 그래프
# =========================================================
st.subheader("📈 50년·100년 학습 회귀선")

graph_years = np.arange(
    int(annual["연도"].min()),
    2026
)

graph_x = graph_years - BASE_YEAR

graph_prediction_50 = predict_temperature(
    graph_x,
    slope_50,
    intercept_50
)

graph_prediction_100 = predict_temperature(
    graph_x,
    slope_100,
    intercept_100
)

fig_models = go.Figure()

# 실제 데이터
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
        x=graph_years,
        y=graph_prediction_50,
        mode="lines",
        name="1956~2005 학습 회귀선"
    )
)

# 100년 회귀선
fig_models.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_prediction_100,
        mode="lines",
        name="1906~2005 학습 회귀선"
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
# ⑦ 테스트 데이터 예측 성능
# =========================================================
st.subheader("④ 공통 테스트 데이터 예측 성능")

st.write(
    "두 모델 모두 **2006~2025년**을 예측하고, "
    "같은 테스트 데이터를 기준으로 MAE, MSE, R²를 계산합니다."
)


performance = pd.DataFrame({
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
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    performance.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 50년 / 100년 성능 카드
# =========================================================
st.subheader("📊 50년 vs 100년 예측 성능")

a1, a2 = st.columns(2)

with a1:
    st.markdown("### 최근 50년 학습")
    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )
    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )
    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )

with a2:
    st.markdown("### 최근 100년 학습")
    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )
    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )
    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# =========================================================
# ⑧ 실제값과 예측값 비교
# =========================================================
st.subheader("🔎 2006~2025년 실제 기온과 예측 기온")

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

# 50년 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test_prediction_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

# 100년 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test_prediction_100,
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
# ⑨ 어느 모델이 더 잘 예측했는지 자동 비교
# =========================================================
st.subheader("💡 예측 성능 비교 결과")

if mae_50 < mae_100:
    mae_result = "50년 학습 모델"
elif mae_100 < mae_50:
    mae_result = "100년 학습 모델"
else:
    mae_result = "두 모델 동일"


if mse_50 < mse_100:
    mse_result = "50년 학습 모델"
elif mse_100 < mse_50:
    mse_result = "100년 학습 모델"
else:
    mse_result = "두 모델 동일"


if r2_50 > r2_100:
    r2_result = "50년 학습 모델"
elif r2_100 > r2_50:
    r2_result = "100년 학습 모델"
else:
    r2_result = "두 모델 동일"


result_table = pd.DataFrame({
    "평가 기준": [
        "기울기",
        "MAE",
        "MSE",
        "R²"
    ],
    "50년 학습": [
        f"{slope_50:.5f} ℃/년",
        f"{mae_50:.3f} ℃",
        f"{mse_50:.3f}",
        f"{r2_50:.3f}"
    ],
    "100년 학습": [
        f"{slope_100:.5f} ℃/년",
        f"{mae_100:.3f} ℃",
        f"{mse_100:.3f}",
        f"{r2_100:.3f}"
    ],
    "비교": [
        "기울기 자체는 좋고 나쁨을 판단하지 않음",
        mae_result,
        mse_result,
        r2_result
    ]
})

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MAE와 MSE는 작을수록 좋고, R²는 클수록 좋습니다."
)


# =========================================================
# ⑩ 연도별 기온 예측
# =========================================================
st.subheader("⑤ 연도별 기온 예측")

st.write(
    "1900년부터 2100년까지 원하는 연도를 선택할 수 있습니다."
)

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 전체 데이터 회귀모델을 이용한 예측
selected_x = selected_year - BASE_YEAR

selected_prediction = predict_temperature(
    selected_x,
    full_slope,
    full_intercept
)


# =========================================================
# 예측값 표시
# =========================================================
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{selected_prediction:.2f} ℃"
)


# =========================================================
# 실제 데이터 범위 밖 안내
# =========================================================
if (
    selected_year < int(annual["연도"].min())
    or selected_year > int(annual["연도"].max())
):
    st.warning(
        f"{selected_year}년은 실제 회귀에 사용한 데이터 범위 "
        f"({int(annual['연도'].min())}~"
        f"{int(annual['연도'].max())}년) 밖입니다. "
        "회귀선을 연장하여 계산한 추정값입니다."
    )
