import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------
# Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1a6fbf; }
    h2 { border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Data Loading
# -------------------------------------------------------
@st.cache_data
def load_data():
    try:
        visits = pd.read_csv("visits_cleaned.csv", encoding="utf-8-sig", low_memory=False)
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ visits_cleaned.csv — กรุณารัน clean_visits.py ก่อน")
        st.stop()

    try:
        monthly = pd.read_csv("monthly_visit_summary.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        monthly = None

    try:
        visits_monthly = pd.read_csv("visits_with_monthly_count.csv", encoding="utf-8-sig", low_memory=False)
    except FileNotFoundError:
        visits_monthly = visits.copy()

    # ทำความสะอาดชื่อคอลัมน์ (ตัด trailing whitespace)
    visits.columns = visits.columns.str.strip()

    # แปลงประเภท
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors="coerce")
    visits["year_month"] = visits["visit_date"].dt.to_period("M").astype(str)

    num_cols = ["age_at_visit", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]
    for col in num_cols:
        if col in visits.columns:
            visits[col] = pd.to_numeric(visits[col], errors="coerce")

    # Fallback: สร้าง systolic_bp / diastolic_bp จาก bp_raw ถ้าคอลัมน์ว่างทั้งหมด
    if "bp_raw" in visits.columns:
        if "systolic_bp" not in visits.columns or visits["systolic_bp"].isna().all():
            bp_split = visits["bp_raw"].astype(str).str.replace(",", "", regex=False).str.extract(
                r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$"
            )
            visits["systolic_bp"] = pd.to_numeric(bp_split[0], errors="coerce")
            visits["diastolic_bp"] = pd.to_numeric(bp_split[1], errors="coerce")
            # range check
            visits.loc[~visits["systolic_bp"].between(60, 250), "systolic_bp"] = float("nan")
            visits.loc[~visits["diastolic_bp"].between(30, 150), "diastolic_bp"] = float("nan")

    if monthly is not None:
        monthly["visit_count"] = pd.to_numeric(monthly["visit_count"], errors="coerce")

    return visits, monthly, visits_monthly


visits_raw, monthly_df, visits_monthly_df = load_data()

# -------------------------------------------------------
# Sidebar — Filters
# -------------------------------------------------------
st.sidebar.header("🔍 ตัวกรองข้อมูล")

all_months = sorted(visits_raw["year_month"].dropna().unique().tolist())
sel_months = st.sidebar.multiselect(
    "เดือน (year_month)", all_months, default=all_months, key="months"
)

all_diseases = sorted(visits_raw["disease_group"].dropna().unique().tolist())
sel_diseases = st.sidebar.multiselect(
    "กลุ่มโรค", all_diseases, default=all_diseases, key="diseases"
)

all_clinics = sorted(visits_raw["clinic_name"].dropna().unique().tolist())
sel_clinics = st.sidebar.multiselect(
    "คลินิก", all_clinics, default=all_clinics, key="clinics"
)

gender_options = visits_raw["gender"].dropna().unique().tolist()
sel_genders = st.sidebar.multiselect(
    "เพศ", gender_options, default=gender_options, key="genders"
)

# Apply filters (guard: ถ้า multiselect ว่าง ให้ใช้ทั้งหมด)
months_filter = sel_months if sel_months else all_months
diseases_filter = sel_diseases if sel_diseases else all_diseases
clinics_filter = sel_clinics if sel_clinics else all_clinics
genders_filter = sel_genders if sel_genders else gender_options

df = visits_raw[
    visits_raw["year_month"].isin(months_filter) &
    visits_raw["disease_group"].isin(diseases_filter) &
    visits_raw["clinic_name"].isin(clinics_filter) &
    visits_raw["gender"].isin(genders_filter)
].copy()

if df.empty:
    st.warning("ไม่มีข้อมูลที่ตรงกับตัวกรองที่เลือก — กรุณาปรับตัวกรอง")
    st.stop()

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.title("🏥 Healthcare Analysis Dashboard")
st.caption(f"ข้อมูล {len(df):,} Visit | {df['patient_id'].nunique():,} ผู้รับบริการ | {len(sel_months)} เดือน")

# -------------------------------------------------------
# Section 1 — KPI Metrics
# -------------------------------------------------------
st.markdown("## 📈 ภาพรวม")

k1, k2, k3, k4, k5 = st.columns(5)

total_visits = len(df)
unique_patients = df["patient_id"].nunique()
avg_visit = round(total_visits / unique_patients, 2) if unique_patients > 0 else 0
bp_available = df["systolic_bp"].notna().sum()
bp_pct = round(bp_available / total_visits * 100, 1) if total_visits > 0 else 0
multi_visit = df.groupby(["patient_id", "year_month"])["visit_id"].count()
returning = (multi_visit > 1).sum()

k1.metric("จำนวน Visit", f"{total_visits:,}")
k2.metric("ผู้รับบริการ", f"{unique_patients:,}")
k3.metric("เฉลี่ย Visit/คน", f"{avg_visit}")
k4.metric("ข้อมูล BP (%)", f"{bp_pct}%")
k5.metric("มาซ้ำในเดือน", f"{returning}")

st.divider()

# -------------------------------------------------------
# Section 2 — โรค + เพศ + คลินิก
# -------------------------------------------------------
st.markdown("## 🩺 กลุ่มโรคและคลินิก")

col_a, col_b = st.columns([2, 1])

with col_a:
    disease_counts = df["disease_group"].value_counts().reset_index()
    disease_counts.columns = ["กลุ่มโรค", "จำนวน Visit"]
    fig_disease = px.bar(
        disease_counts, x="จำนวน Visit", y="กลุ่มโรค",
        orientation="h", title="จำนวน Visit แยกกลุ่มโรค",
        color="จำนวน Visit", color_continuous_scale="Blues",
        text="จำนวน Visit"
    )
    fig_disease.update_traces(textposition="outside")
    fig_disease.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
    st.plotly_chart(fig_disease, use_container_width=True)

with col_b:
    gender_counts = df["gender"].value_counts().reset_index()
    gender_counts.columns = ["เพศ", "จำนวน"]
    fig_gender = px.pie(
        gender_counts, values="จำนวน", names="เพศ",
        title="สัดส่วนเพศ",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_gender.update_layout(height=350)
    st.plotly_chart(fig_gender, use_container_width=True)

# Top 10 คลินิก
clinic_counts = df["clinic_name"].value_counts().head(10).reset_index()
clinic_counts.columns = ["คลินิก", "จำนวน Visit"]
fig_clinic = px.bar(
    clinic_counts, x="คลินิก", y="จำนวน Visit",
    title="Top 10 คลินิกที่มี Visit มากที่สุด",
    color="จำนวน Visit", color_continuous_scale="Teal",
    text="จำนวน Visit"
)
fig_clinic.update_traces(textposition="outside")
fig_clinic.update_layout(
    xaxis_tickangle=-30, showlegend=False,
    coloraxis_showscale=False, height=380
)
st.plotly_chart(fig_clinic, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Section 3 — Vital Signs
# -------------------------------------------------------
st.markdown("## 💉 Vital Signs")

col_bp1, col_bp2 = st.columns(2)

with col_bp1:
    bp_data = df["systolic_bp"].dropna()
    if not bp_data.empty:
        fig_sbp = px.histogram(
            df, x="systolic_bp", nbins=30,
            title="การกระจาย Systolic BP",
            labels={"systolic_bp": "Systolic BP (mmHg)"},
            color_discrete_sequence=["#e06c75"]
        )
        fig_sbp.add_vline(x=140, line_dash="dash", line_color="red",
                          annotation_text="เกณฑ์ HT (140)", annotation_position="top right")
        fig_sbp.update_layout(height=320)
        st.plotly_chart(fig_sbp, use_container_width=True)
    else:
        st.info("ไม่มีข้อมูล Systolic BP")

with col_bp2:
    dbp_data = df["diastolic_bp"].dropna()
    if not dbp_data.empty:
        fig_dbp = px.histogram(
            df, x="diastolic_bp", nbins=30,
            title="การกระจาย Diastolic BP",
            labels={"diastolic_bp": "Diastolic BP (mmHg)"},
            color_discrete_sequence=["#61afef"]
        )
        fig_dbp.add_vline(x=90, line_dash="dash", line_color="red",
                          annotation_text="เกณฑ์ HT (90)", annotation_position="top right")
        fig_dbp.update_layout(height=320)
        st.plotly_chart(fig_dbp, use_container_width=True)
    else:
        st.info("ไม่มีข้อมูล Diastolic BP")

# Box plot BP แยกกลุ่มโรค
bp_cols = ["systolic_bp", "diastolic_bp"]
has_bp = df[bp_cols].notna().any(axis=1)
if has_bp.sum() > 0:
    bp_melt = df[has_bp][["disease_group", "systolic_bp", "diastolic_bp"]].melt(
        id_vars="disease_group", var_name="ประเภท", value_name="mmHg"
    ).dropna()
    fig_box = px.box(
        bp_melt, x="disease_group", y="mmHg", color="ประเภท",
        title="BP แยกตามกลุ่มโรค",
        labels={"disease_group": "กลุ่มโรค"},
        color_discrete_map={"systolic_bp": "#e06c75", "diastolic_bp": "#61afef"}
    )
    fig_box.update_layout(height=380, xaxis_tickangle=-15)
    st.plotly_chart(fig_box, use_container_width=True)

# Scatter BMI vs อายุ
has_bmi = df[["bmi", "age_at_visit"]].notna().all(axis=1)
if has_bmi.sum() > 0:
    fig_scatter = px.scatter(
        df[has_bmi], x="age_at_visit", y="bmi",
        color="disease_group", title="BMI vs อายุ แยกกลุ่มโรค",
        labels={"age_at_visit": "อายุ (ปี)", "bmi": "BMI"},
        opacity=0.7, height=380
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Section 4 — Monthly Trends
# -------------------------------------------------------
st.markdown("## 📅 แนวโน้มรายเดือน")

if monthly_df is not None and not monthly_df.empty:
    # กรองเดือนตาม sidebar
    monthly_filtered = monthly_df[monthly_df["year_month"].isin(sel_months)].copy()

    # Line chart: Visit ต่อเดือน
    monthly_agg = (
        monthly_filtered.groupby("year_month")
        .agg(total_visit=("visit_count", "sum"), unique_patients=("patient_id", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )
    fig_trend = px.line(
        monthly_agg, x="year_month", y="total_visit",
        title="จำนวน Visit รายเดือน",
        markers=True, text="total_visit",
        labels={"year_month": "เดือน", "total_visit": "จำนวน Visit"}
    )
    fig_trend.update_traces(textposition="top center", line_color="#1a6fbf", marker_size=8)
    fig_trend.update_layout(height=320)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_ret1, col_ret2 = st.columns(2)

    with col_ret1:
        # ผู้รับบริการใหม่ vs กลับมา
        type_agg = monthly_filtered.copy()
        type_agg["ประเภท"] = type_agg["visit_count"].apply(
            lambda x: "มาซ้ำในเดือน (≥2)" if x >= 2 else "มาครั้งเดียว"
        )
        type_summary = (
            type_agg.groupby(["year_month", "ประเภท"])
            .size()
            .reset_index(name="จำนวนผู้รับบริการ")
            .sort_values("year_month")
        )
        fig_type = px.bar(
            type_summary, x="year_month", y="จำนวนผู้รับบริการ",
            color="ประเภท", barmode="stack",
            title="ผู้รับบริการมาครั้งเดียว vs มาซ้ำ",
            labels={"year_month": "เดือน"},
            color_discrete_map={
                "มาครั้งเดียว": "#98c379",
                "มาซ้ำในเดือน (≥2)": "#e06c75"
            }
        )
        fig_type.update_layout(height=340)
        st.plotly_chart(fig_type, use_container_width=True)

    with col_ret2:
        # Top patients ที่มาบ่อย
        top_patients = (
            monthly_filtered[monthly_filtered["visit_count"] >= 2]
            .sort_values(["visit_count", "year_month"], ascending=[False, True])
            .head(15)[["patient_id", "year_month", "visit_count", "first_visit", "last_visit"]]
            .rename(columns={
                "patient_id": "Patient ID",
                "year_month": "เดือน",
                "visit_count": "ครั้ง",
                "first_visit": "วันแรก",
                "last_visit": "วันสุดท้าย",
            })
        )
        st.markdown("**Patient ที่มาซ้ำในเดือนเดียว**")
        if not top_patients.empty:
            st.dataframe(top_patients, use_container_width=True, hide_index=True, height=300)
        else:
            st.info("ไม่มี patient ที่มาซ้ำในเดือนเดียวภายใต้ filter ที่เลือก")

else:
    # Fallback ถ้าไม่มี monthly_visit_summary.csv
    monthly_fallback = (
        df.groupby("year_month")
        .agg(total_visit=("visit_id", "count"), unique_patients=("patient_id", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )
    fig_trend = px.line(
        monthly_fallback, x="year_month", y="total_visit",
        title="จำนวน Visit รายเดือน", markers=True,
        labels={"year_month": "เดือน", "total_visit": "จำนวน Visit"}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Section 5 — Raw Data Table
# -------------------------------------------------------
with st.expander("📋 ดูข้อมูลดิบ (visits_cleaned)", expanded=False):
    show_cols = [
        "visit_date", "visit_id", "patient_id", "gender", "age_at_visit",
        "clinic_name", "diagnosis_text", "disease_group",
        "systolic_bp", "diastolic_bp", "bmi"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols].sort_values("visit_date", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"แสดง {len(df):,} แถว")