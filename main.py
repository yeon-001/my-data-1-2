import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==================================================
# 기본 설정
# ==================================================
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


# ==================================================
# 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
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

    # 관측일수가 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_DAYS
    ].copy()

    annual["지난연수"] = (
        annual["연도"] - BASE_YEAR
    )

    annual = annual.sort_values("연도")

    return annual


annual = load_data()


# ==================================================
# 선형회귀 함수
# numpy만 사용
# ==================================================
def make_linear_regression(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # y = slope * x + intercept
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


# ==================================================
# 예측 함수
# ==================================================
def predict_temperature(
    x,
    slope,
    intercept
):

    return slope * np.asarray(x) + intercept


# ==================================================
# 평가 지표 함수
# ==================================================
def evaluate_model(
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

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# ==================================================
# 전체 데이터 회귀
# ==================================================
x_all = annual["지난연수"]
y_all = annual["연평균기온"]

full_slope, full_intercept = make_linear_regression(
    x_all,
    y_all
)

annual["전체회귀예측"] = predict_temperature(
    x_all,
    full_slope,
    full_intercept
)


# ==================================================
# 제목
# ==================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 이용해 선형회귀 모델을 만들고, "
    "과거 데이터를 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 비교합니다."
)


# ==================================================
# 전체 데이터 정보
# ==================================================
st.subheader("📊 전체 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "사용한 연도 수",
    f"{len(annual)}개"
)

col2.metric(
    "시작 연도",
    f"{int(annual['연도'].min())}년"
)

col3.metric(
    "끝 연도",
    f"{int(annual['연도'].max())}년"
)

col4.metric(
    "전체 회귀 기울기",
    f"{full_slope:.4f} ℃/년"
)

st.caption(
    "2025년 이후 데이터와 연평균기온 관측일수가 "
    "300일 미만인 연도는 제외했습니다."
)


# ==================================================
# 전체 데이터 산점도 + 회귀선
# ==================================================
st.subheader("📈 전체 연도별 평균기온과 회귀선")

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃"
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


# ==================================================
# 학습 / 테스트 데이터
# ==================================================

# 최근 50년 학습
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# 최근 100년 학습
# 실제 데이터가 1908년부터라면
# 1906~1907년은 자동으로 제외됨
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# ==================================================
# 50년 모델
# ==================================================
slope_50, intercept_50 = make_linear_regression(
    train_50["지난연수"],
    train_50["연평균기온"]
)

pred_50 = predict_temperature(
    test["지난연수"],
    slope_50,
    intercept_50
)


# ==================================================
# 100년 모델
# ==================================================
slope_100, intercept_100 = make_linear_regression(
    train_100["지난연수"],
    train_100["연평균기온"]
)

pred_100 = predict_temperature(
    test["지난연수"],
    slope_100,
    intercept_100
)


# ==================================================
# 모델 평가
# ==================================================
mae_50, mse_50, r2_50 = evaluate_model(
    test["연평균기온"],
    pred_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    test["연평균기온"],
    pred_100
)


# ==================================================
# 학습 / 테스트 데이터 설명
# ==================================================
st.subheader("🧪 학습 데이터와 테스트 데이터")

st.write(
    "두 모델 모두 동일한 2006~2025년 데이터를 테스트 데이터로 사용합니다."
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "50년 모델 학습",
    f"{int(train_50['연도'].min())}~"
    f"{int(train_50['연도'].max())}년"
)

c2.metric(
    "100년 모델 학습",
    f"{int(train_100['연도'].min())}~"
    f"{int(train_100['연도'].max())}년"
)

c3.metric(
    "공통 테스트",
    f"{int(test['연도'].min())}~"
    f"{int(test['연도'].max())}년"
)

st.write(
    f"""
    - 최근 50년 학습 데이터: **{len(train_50)}개 연도**
    - 최근 100년 학습 데이터: **{len(train_100)}개 연도**
    - 공통 테스트 데이터: **{len(test)}개 연도**
    """
)


# ==================================================
# 기울기 비교
# ==================================================
st.subheader("📐 회귀선 기울기 비교")

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


# ==================================================
# 회귀선 비교 그래프
# ==================================================
st.subheader("📈 50년 학습 회귀선과 100년 학습 회귀선")

line_years = np.arange(
    int(annual["연도"].min()),
    2026
)

line_x = line_years - BASE_YEAR

line_pred_50 = predict_temperature(
    line_x,
    slope_50,
    intercept_50
)

line_pred_100 = predict_temperature(
    line_x,
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
        x=line_years,
        y=line_pred_50,
        mode="lines",
        name="1956~2005 학습 회귀선"
    )
)

# 100년 회귀선
fig_models.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred_100,
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


# ==================================================
# 성능 비교
# ==================================================
st.subheader("📊 테스트 데이터 예측 성능 비교")

comparison = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습",
            "최근 100년 학습"
        ],
        "학습 기간": [
            "1956~2005",
            "1906~2005"
        ],
        "학습 연도 수": [
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
    }
)

st.dataframe(
    comparison.style.format(
        {
            "기울기 (℃/년)": "{:.5f}",
            "MAE (℃)": "{:.4f}",
            "MSE (℃²)": "{:.4f}",
            "R²": "{:.4f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 테스트 실제값 vs 예측값
# ==================================================
st.subheader("🔎 2006~2025년 실제 기온과 예측 기온")

fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온"
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

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


# ==================================================
# 자동 비교 해석
# ==================================================
st.subheader("💡 모델 성능 해석")

if mae_50 < mae_100:
    mae_result = "50년 모델의 MAE가 더 작습니다."
elif mae_50 > mae_100:
    mae_result = "100년 모델의 MAE가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."


if mse_50 < mse_100:
    mse_result = "50년 모델의 MSE가 더 작습니다."
elif mse_50 > mse_100:
    mse_result = "100년 모델의 MSE가 더 작습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."


if r2_50 > r2_100:
    r2_result = "50년 모델의 R²가 더 높습니다."
elif r2_50 < r2_100:
    r2_result = "100년 모델의 R²가 더 높습니다."
else:
    r2_result = "두 모델의 R²가 같습니다."


st.write(
    f"""
    **MAE:** {mae_result}

    **MSE:** {mse_result}

    **R²:** {r2_result}

    50년 모델의 기울기는 **{slope_50:.5f} ℃/년**, 
    100년 모델의 기울기는 **{slope_100:.5f} ℃/년**입니다.
    """
)

st.caption(
    "MAE와 MSE는 작을수록 좋으며, R²는 높을수록 좋습니다."
)


# ==================================================
# 기존 연도 예측 기능
# ==================================================
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - BASE_YEAR

predicted_temperature = predict_temperature(
    selected_x,
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
        f"{selected_year}년은 실제 회귀 데이터 범위 "
        f"({int(annual['연도'].min())}~"
        f"{int(annual['연도'].max())}) 밖이므로 "
        "회귀선을 연장한 추정값입니다."
    )
