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
MIN_DAYS = 300


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온 + 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수 300일 미만인 연도 제외
    annual = annual[
        annual["관측일수"] >= MIN_DAYS
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

    return (
        slope * x + intercept
    )


# =========================================================
# 평가 지표
# =========================================================
def evaluate(y_true, y_pred):

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
    "서울의 연도별 평균기온을 이용하여 선형회귀 모델을 만들고, "
    "과거 데이터를 학습한 모델이 최근 20년의 기온을 "
    "얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 데이터 정보
# =========================================================
st.subheader("📋 분석 데이터")

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "사용 연도 수",
    f"{len(annual)}개"
)

d2.metric(
    "시작 연도",
    f"{int(annual['연도'].min())}년"
)

d3.metric(
    "끝 연도",
    f"{int(annual['연도'].max())}년"
)

d4.metric(
    "최소 관측일",
    "300일"
)

st.caption(
    "2025년 이후 데이터와 관측일수가 300일 미만인 연도는 제외했습니다."
)


# =========================================================
# ① 전체 데이터 회귀모델
# =========================================================
st.header("① 전체 데이터로 만든 회귀모델")

st.write(
    "전체 연평균기온 데이터를 이용해 선형회귀 모델을 만들었습니다. "
    "회귀모델의 독립변수는 **1908년부터 지난 연수**입니다."
)


# 전체 데이터 회귀
x_all = annual["지난연수"]
y_all = annual["연평균기온"]

full_slope, full_intercept = make_regression(
    x_all,
    y_all
)

full_pred = predict(
    x_all,
    full_slope,
    full_intercept
)

full_mae, full_mse, full_r2 = evaluate(
    y_all,
    full_pred
)

full_corr = np.corrcoef(
    x_all,
    y_all
)[0, 1]

annual["전체회귀예측"] = full_pred


# =========================================================
# 전체 모델 핵심 지표
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "연간 기온 변화",
    f"{full_slope:.2f} ℃/년"
)

c2.metric(
    "상관계수 r",
    f"{full_corr:.3f}"
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


st.caption(
    "위 MAE·MSE·R²는 전체 데이터를 회귀선에 적용해 계산한 값입니다. "
    "아래의 50년·100년 모델 평가는 별도의 테스트 데이터로 계산합니다."
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
# 전체 회귀식
# =========================================================
st.write(
    f"**전체 회귀식:** "
    f"연평균기온 = "
    f"{full_slope:.5f} × (연도 - {BASE_YEAR}) "
    f"{full_intercept:+.3f}"
)


# =========================================================
# ② 훈련 / 테스트 데이터 분리
# =========================================================
st.header("② 훈련 데이터와 테스트 데이터 분리")

st.write(
    "최근 50년과 최근 100년의 과거 데이터를 각각 훈련에 사용하고, "
    "두 모델 모두 **2006~2025년을 공통 테스트 데이터**로 사용합니다."
)


# 50년 훈련
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# 100년 훈련
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# 공통 테스트
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


split_df = pd.DataFrame({
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
    "사용 연도 수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    split_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ③ 50년 / 100년 모델 학습
# =========================================================

# ---------------------------------------------------------
# 50년 모델
# ---------------------------------------------------------
slope_50, intercept_50 = make_regression(
    train_50["지난연수"],
    train_50["연평균기온"]
)

pred_50 = predict(
    test["지난연수"],
    slope_50,
    intercept_50
)

mae_50, mse_50, r2_50 = evaluate(
    test["연평균기온"],
    pred_50
)


# ---------------------------------------------------------
# 100년 모델
# ---------------------------------------------------------
slope_100, intercept_100 = make_regression(
    train_100["지난연수"],
    train_100["연평균기온"]
)

pred_100 = predict(
    test["지난연수"],
    slope_100,
    intercept_100
)

mae_100, mse_100, r2_100 = evaluate(
    test["연평균기온"],
    pred_100
)


# =========================================================
# ③ 50년 vs 100년 비교
# =========================================================
st.header("③ 50년 학습과 100년 학습 비교")

st.write(
    "두 모델의 회귀선 기울기와 **동일한 2006~2025년 테스트 데이터에 대한 "
    "예측 성능**을 비교합니다."
)


comparison = pd.DataFrame({
    "평가 항목": [
        "회귀선 기울기",
        "MAE",
        "MSE",
        "R²"
    ],
    "최근 50년 학습": [
        f"{slope_50:.5f} ℃/년",
        f"{mae_50:.3f} ℃",
        f"{mse_50:.3f}",
        f"{r2_50:.3f}"
    ],
    "최근 100년 학습": [
        f"{slope_100:.5f} ℃/년",
        f"{mae_100:.3f} ℃",
        f"{mse_100:.3f}",
        f"{r2_100:.3f}"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 성능 비교 카드
# =========================================================
st.subheader("📊 테스트 데이터 예측 성능")

col50, col100 = st.columns(2)

with col50:

    st.markdown("### 🟦 최근 50년 학습")

    st.metric(
        "회귀선 기울기",
        f"{slope_50:.5f} ℃/년"
    )

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


with col100:

    st.markdown("### 🟩 최근 100년 학습")

    st.metric(
        "회귀선 기울기",
        f"{slope_100:.5f} ℃/년"
    )

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


st.caption(
    "MAE와 MSE는 작을수록 좋고, R²는 클수록 좋습니다."
)


# =========================================================
# 50년 / 100년 회귀선 비교 그래프
# =========================================================
st.subheader("📈 50년·100년 학습 회귀선 비교")

graph_years = np.arange(
    int(annual["연도"].min()),
    2026
)

graph_x = (
    graph_years - BASE_YEAR
)

line_50 = predict(
    graph_x,
    slope_50,
    intercept_50
)

line_100 = predict(
    graph_x,
    slope_100,
    intercept_100
)

fig_compare = go.Figure()

# 실제 데이터
fig_compare.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온"
    )
)

# 50년 회귀선
fig_compare.add_trace(
    go.Scatter(
        x=graph_years,
        y=line_50,
        mode="lines",
        name="1956~2005 학습 회귀선"
    )
)

# 100년 회귀선
fig_compare.add_trace(
    go.Scatter(
        x=graph_years,
        y=line_100,
        mode="lines",
        name="1906~2005 학습 회귀선"
    )
)

# 테스트 기간
fig_compare.add_vrect(
    x0=2006,
    x1=2025,
    opacity=0.12,
    line_width=0,
    annotation_text="공통 테스트 기간"
)

fig_compare.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

fig_compare.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# =========================================================
# ④ 실제값 vs 예측값
# =========================================================
st.header("④ 2006~2025년 실제값과 예측값")

st.write(
    "학습에 사용하지 않은 2006~2025년의 실제 연평균기온과 "
    "50년·100년 학습 모델의 예측값을 비교합니다."
)

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

# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

# 100년 모델
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
# ⑤ 최종 비교 결과
# =========================================================
st.header("⑤ 모델 비교 결과")

if mae_50 < mae_100:
    mae_winner = "최근 50년 학습"
elif mae_100 < mae_50:
    mae_winner = "최근 100년 학습"
else:
    mae_winner = "동일"


if mse_50 < mse_100:
    mse_winner = "최근 50년 학습"
elif mse_100 < mse_50:
    mse_winner = "최근 100년 학습"
else:
    mse_winner = "동일"


if r2_50 > r2_100:
    r2_winner = "최근 50년 학습"
elif r2_100 > r2_50:
    r2_winner = "최근 100년 학습"
else:
    r2_winner = "동일"


result_df = pd.DataFrame({
    "평가 기준": [
        "MAE",
        "MSE",
        "R²"
    ],
    "결과": [
        mae_winner,
        mse_winner,
        r2_winner
    ],
    "판단 기준": [
        "작을수록 좋음",
        "작을수록 좋음",
        "클수록 좋음"
    ]
})

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 자동 해석
# =========================================================
if (
    mae_50 < mae_100
    and mse_50 < mse_100
    and r2_50 > r2_100
):
    overall_result = (
        "최근 50년 학습 모델이 세 가지 평가 지표에서 "
        "모두 더 좋은 성능을 보였습니다."
    )

elif (
    mae_100 < mae_50
    and mse_100 < mse_50
    and r2_100 > r2_50
):
    overall_result = (
        "최근 100년 학습 모델이 세 가지 평가 지표에서 "
        "모두 더 좋은 성능을 보였습니다."
    )

else:
    overall_result = (
        "평가 지표에 따라 더 좋은 모델이 달라집니다. "
        "MAE·MSE는 낮은 값, R²는 높은 값을 중심으로 비교해야 합니다."
    )


st.info(
    f"""
    **분석 결과**

    50년 학습 회귀선의 기울기는 **{slope_50:.5f} ℃/년**,
    100년 학습 회귀선의 기울기는 **{slope_100:.5f} ℃/년**입니다.

    따라서 두 학습 기간은 장기적인 기온 상승 추세를
    서로 다르게 나타낼 수 있습니다.

    {overall_result}
    """
)


# =========================================================
# ⑥ 연도별 기온 예측
# =========================================================
st.header("⑥ 연도별 기온 예측")

st.write(
    "전체 데이터를 이용해 만든 회귀모델을 기준으로 "
    "1900~2100년의 예상 연평균기온을 확인할 수 있습니다."
)

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 1908년부터 지난 연수
selected_x = (
    selected_year - BASE_YEAR
)

selected_prediction = predict(
    selected_x,
    full_slope,
    full_intercept
)


# =========================================================
# 예측값
# =========================================================
st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{selected_prediction:.2f} ℃"
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
        "따라서 회귀선을 연장한 추정값입니다."
    )
