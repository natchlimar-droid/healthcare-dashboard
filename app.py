# =========================================================
# 1) IMPORTS
# =========================================================
import os
import io
import zipfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# หากใช้โมเดล Holt-Winters
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =========================================================
# 2) STREAMLIT CONFIGURATION
# ต้องอยู่ก่อนคำสั่ง Streamlit อื่น
# =========================================================
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3) CONSTANTS / THEME
# =========================================================
COLORS = {
    "primary": "#1F77B4",
    "success": "#2CA02C",
    "warning": "#FF7F0E",
    "danger": "#D62728",
    "neutral": "#6C757D",
    "background": "#F7F9FC",
}

DATA_FILE_CANDIDATES = [
    "visits_cleaned.csv",
    "visits_cleaned.zip",
    "visits_with_monthly_count.csv",
    "visits_with_monthly_count.zip",
]


# =========================================================
# 4) CSS / STYLING
# =========================================================
def apply_custom_css():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.7rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5) DATA UTILITIES
# =========================================================
def find_data_file(file_candidates, base_dir=None):
    """ค้นหาไฟล์ข้อมูลจากรายชื่อไฟล์ที่กำหนด"""
    if base_dir is None:
        base_dir = os.getcwd()

    for filename in file_candidates:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            return file_path

    return None


def read_csv_or_zip(file_path):
    """อ่าน CSV หรือ ZIP ที่ภายในมี CSV"""
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            csv_files = [
                name for name in zip_ref.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError("ไม่พบไฟล์ CSV ภายใน ZIP")

            with zip_ref.open(csv_files[0]) as csv_file:
                return pd.read_csv(csv_file)

    return pd.read_csv(file_path)


# =========================================================
# 6) DATA LOADING & PREPROCESSING
# =========================================================
@st.cache_data(show_spinner="กำลังโหลดข้อมูล...")
def load_data():
    """โหลดข้อมูลผู้รับบริการและสร้างข้อมูลรายเดือนสำหรับ Forecast"""
    main_file = find_data_file(DATA_FILE_CANDIDATES)

    if main_file is None:
        raise FileNotFoundError(
            "ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบว่าไฟล์ CSV หรือ ZIP อยู่ในโฟลเดอร์เดียวกับแอป"
        )

    df = read_csv_or_zip(main_file)
    df.columns = df.columns.str.strip()

    # ตรวจหาคอลัมน์วันที่
    date_candidates = ["date", "visit_date", "วันที่รับบริการ", "service_date"]
    date_col = next((col for col in date_candidates if col in df.columns), None)

    has_date = date_col is not None
    if has_date:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        df["visit_date"] = df[date_col]

    # ตรวจหาคอลัมน์อายุ
    age_candidates = ["age", "อายุ", "patient_age"]
    age_col = next((col for col in age_candidates if col in df.columns), None)

    has_age = age_col is not None
    if has_age:
        df["age"] = pd.to_numeric(df[age_col], errors="coerce")

    # คำนวณ BMI หากมีข้อมูลส่วนสูงและน้ำหนัก
    weight_candidates = ["weight", "น้ำหนัก", "weight_kg"]
    height_candidates = ["height", "ส่วนสูง", "height_cm"]

    weight_col = next((col for col in weight_candidates if col in df.columns), None)
    height_col = next((col for col in height_candidates if col in df.columns), None)

    if weight_col and height_col:
        weight = pd.to_numeric(df[weight_col], errors="coerce")
        height_m = pd.to_numeric(df[height_col], errors="coerce") / 100

        df["bmi"] = np.where(
            height_m > 0,
            weight / (height_m ** 2),
            np.nan,
        )

    # สร้างข้อมูลจำนวนผู้รับบริการรายเดือน
    if has_date:
        monthly = (
            df.dropna(subset=["visit_date"])
            .assign(month=lambda x: x["visit_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")
            .size()
            .reset_index(name="sales")
            .rename(columns={"month": "date"})
            .sort_values("date")
        )
    else:
        monthly = pd.DataFrame(columns=["date", "sales"])

    return df, monthly, has_date, has_age


@st.cache_data(show_spinner=False)
def process_patient_data(dataframe):
    """ประมวลผลข้อมูลผู้ป่วยและกำหนดระดับความสำคัญ"""
    df_proc = dataframe.copy()

    if "visit_date" in df_proc.columns:
        latest_date = df_proc["visit_date"].max()
        df_proc["days_since_last_visit"] = (
            latest_date - df_proc["visit_date"]
        ).dt.days
    else:
        df_proc["days_since_last_visit"] = np.nan

    # ตัวอย่างการกำหนด Priority
    conditions = [
        df_proc["days_since_last_visit"] > 365,
        df_proc["days_since_last_visit"] > 180,
    ]
    choices = [
        "P1-Urgent",
        "P2-Warning",
    ]

    df_proc["priority_status"] = np.select(
        conditions,
        choices,
        default="P3-Normal",
    )

    return df_proc


# =========================================================
# 7) FORECASTING
# =========================================================
def run_forecasting_model(
    df_hist,
    horizon=6,
    m_type="add",
    is_damped=False,
    z_val=1.96,
    target_pct=0.10,
):
    """พยากรณ์จำนวนผู้รับบริการด้วย Holt-Winters Exponential Smoothing"""
    if not STATSMODELS_AVAILABLE:
        return None

    if df_hist.empty or len(df_hist) < 6:
        return None

    df_hist = df_hist.copy().sort_values("date")
    ts = pd.Series(
        df_hist["sales"].values,
        index=pd.to_datetime(df_hist["date"]),
    )

    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            damped_trend=is_damped,
            seasonal=None,
        ).fit(optimized=True)

        fitted_series = model.fittedvalues
        forecast = model.forecast(horizon)

        residuals = ts - fitted_series
        res_std = residuals.std()

        future_dates = pd.date_range(
            start=ts.index.max() + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast.values,
        })

        forecast_df["lower_bound"] = (
            forecast_df["forecast"] - z_val * res_std
        ).clip(lower=0)

        forecast_df["upper_bound"] = (
            forecast_df["forecast"] + z_val * res_std
        )

        forecast_df["target"] = (
            forecast_df["forecast"] * (1 + target_pct)
        )

        return {
            "model": model,
            "fitted": fitted_series,
            "forecast_df": forecast_df,
            "res_std": res_std,
        }

    except Exception as error:
        st.warning(f"ไม่สามารถสร้างโมเดลพยากรณ์ได้: {error}")
        return None


# =========================================================
# 8) DASHBOARD COMPONENTS / FRAGMENTS
# =========================================================
@st.fragment
def render_action_panel(dv):
    st.subheader("Action Panel")

    if "priority_status" not in dv.columns:
        st.info("ยังไม่มีข้อมูลลำดับความสำคัญ")
        return

    urgent_df = dv[dv["priority_status"] == "P1-Urgent"]

    st.metric("ผู้ป่วยกลุ่มเร่งด่วน", len(urgent_df))

    if urgent_df.empty:
        st.success("ไม่พบผู้ป่วยกลุ่มเร่งด่วน")
    else:
        st.dataframe(urgent_df, use_container_width=True)


@st.fragment
def render_patient_profile(avail_df, summary_pts, dv, sel_idx):
    st.subheader("Patient Profile")

    if avail_df.empty:
        st.info("ไม่พบข้อมูลผู้ป่วย")
        return

    selected_row = avail_df.iloc[sel_idx]
    st.json(selected_row.to_dict())


@st.fragment
def render_disease_center(dv, df, selected_tab_key, active_config, search_term):
    st.subheader(f"Disease Command Center: {selected_tab_key}")

    filtered_df = dv.copy()

    if search_term:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask.any(axis=1)]

    st.dataframe(filtered_df, use_container_width=True)


@st.fragment
def render_other_packages_dashboard(dv, df, search_term):
    st.subheader("AI Health Data Architect")

    if search_term:
        mask = dv.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        )
        dv = dv[mask.any(axis=1)]

    st.dataframe(dv, use_container_width=True)


def render_general_dashboard(dv, monthly_df, forecast_results):
    st.title("🏥 Healthcare Analytics Dashboard")

    total_visits = len(dv)
    urgent_count = int((dv["priority_status"] == "P1-Urgent").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("จำนวนรายการทั้งหมด", f"{total_visits:,}")
    col2.metric("ผู้ป่วยเร่งด่วน", f"{urgent_count:,}")

    if not monthly_df.empty:
        col3.metric(
            "เดือนล่าสุด",
            f"{int(monthly_df.iloc[-1]['sales']):,}",
        )

        fig = px.line(
            monthly_df,
            x="date",
            y="sales",
            markers=True,
            title="จำนวนผู้รับบริการรายเดือน",
        )

        if forecast_results is not None:
            forecast_df = forecast_results["forecast_df"]

            fig.add_scatter(
                x=forecast_df["date"],
                y=forecast_df["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash"),
            )

        st.plotly_chart(fig, use_container_width=True)

    render_action_panel(dv)


# =========================================================
# 9) MAIN APPLICATION
# =========================================================
def main():
    apply_custom_css()

    try:
        df, monthly_df, has_date, has_age = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"เกิดข้อผิดพลาดระหว่างโหลดข้อมูล: {error}")
        st.stop()

    dv = process_patient_data(df)

    # Sidebar ควรมีเฉพาะ controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")

        page = st.radio(
            "เลือกหน้า Dashboard",
            options=[
                "General Dashboard",
                "Disease Center",
                "Other Packages",
            ],
        )

        search_term = st.text_input("ค้นหาข้อมูลผู้ป่วย")

        forecast_horizon = st.slider(
            "ระยะเวลาพยากรณ์ (เดือน)",
            min_value=1,
            max_value=24,
            value=6,
        )

        target_pct = st.slider(
            "เป้าหมายการเติบโต (%)",
            min_value=0,
            max_value=100,
            value=10,
        ) / 100

    # ห้ามใส่ st.stop() ตรงนี้
    # เพราะจะทำให้ส่วน Dashboard ด้านล่างไม่ทำงาน

    forecast_results = run_forecasting_model(
        df_hist=monthly_df,
        horizon=forecast_horizon,
        target_pct=target_pct,
    )

    if page == "General Dashboard":
        render_general_dashboard(
            dv=dv,
            monthly_df=monthly_df,
            forecast_results=forecast_results,
        )

    elif page == "Disease Center":
        active_config = {
            "is_general": False,
            "is_other_packages": False,
        }

        render_disease_center(
            dv=dv,
            df=df,
            selected_tab_key="Disease Center",
            active_config=active_config,
            search_term=search_term,
        )

    elif page == "Other Packages":
        render_other_packages_dashboard(
            dv=dv,
            df=df,
            search_term=search_term,
        )


if __name__ == "__main__":
    main()
