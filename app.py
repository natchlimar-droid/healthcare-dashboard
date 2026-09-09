# app.py
"""Refactored Streamlit dashboard for Clinical Command Center.
- Uses @st.cache_data for data loading and @st.cache_resource for ML model.
- Splits logic into clear helper functions.
- Improves performance by caching heavy aggregations.
- Keeps original visual style while simplifying CSS.
- Ready to be deployed to Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ---------------------------------------------------------------
# Design Tokens (Tailwind‑like colors) – kept simple for Streamlit
# ---------------------------------------------------------------
INK = "#0B1B2B"
BG = "#F4F7F5"
SURFACE = "#FFFFFF"
BORDER = "#DCE3E0"
TEAL = "#0E5C56"
TEAL_LIGHT = "#12857A"
SAGE = "#4F7C5B"
AMBER = "#C87F0A"
RED = "#B3261E"
MUTED = "#5B6B6B"

DISEASE_COLORS = {
    "ความดันโลหิตสูง": TEAL,
    "เบาหวาน": "#2F6FB5",
    "ไขมันในเลือดสูง": AMBER,
    "อื่น ๆ": "#9AA6A0",
}
BP_COLORS = {
    "ปกติ (<120)": SAGE,
    "เฝ้าระวัง (120-139)": AMBER,
    "สูง (≥140)": RED,
    "ไม่มีข้อมูล": "#C7CFCC",
}

# ---------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------
st.set_page_config(page_title="Clinical Command Center", page_icon="🏥", layout="wide")
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
        html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {INK}; }}
        .stApp {{ background-color: {BG}; }}
        .block-container {{ padding-top: 1.1rem !important; padding-bottom: 1.5rem !important; max-width: 98% !important; }}
        div[data-testid="stMetric"] {{ background: {SURFACE}; border-radius: 10px; padding: 14px 18px; border: 1px solid {BORDER}; }}
        div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important; font-weight: 600 !important; color: {INK}; }}
        div[data-testid="stMetricLabel"] {{ font-size: 0.8rem !important; font-weight: 600 !important; color: {MUTED} !important; }}
        .header-bar {{ border-bottom: 1px solid {BORDER}; padding-bottom: 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; }}
        .header-title {{ font-size: 1.3rem; font-weight: 700; color: {INK}; }}
        .header-sub {{ font-size: 0.83rem; color: {MUTED}; margin-top: 2px; }}
        .asof-chip {{ background: {TEAL}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
        .quality-strip {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
        .quality-chip {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {AMBER}; border-radius: 8px; padding: 7px 12px; font-size: 0.74rem; color: {INK}; }}
        .quality-chip b {{ font-family: 'IBM Plex Mono', monospace; color: {AMBER}; }}
        .panel-title {{ font-size: 0.88rem; font-weight: 700; color: {INK}; margin-bottom: 2px; }}
        .panel-sub {{ font-size: 0.72rem; color: {MUTED}; margin-bottom: 8px; }}
        .action-card {{ background: {SURFACE}; border: 1px solid {BORDER}; padding: 10px 12px; border-radius: 8px; }}
        .action-card b {{ font-size: 0.8rem; color: {INK}; }}
        .action-card span {{ font-size: 0.74rem; color: {MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
MIN_SAMPLE_FOR_DELTA = 5
MIN_SAMPLE_FOR_TREND = 5
SUNBURST_TOP_N = 5
DATA_PATH = Path(__file__).parent / "visits_cleaned.csv"

# ---------------------------------------------------------------
# Data loading – cached once per session
# ---------------------------------------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None, False, False
    df = pd.read_csv(DATA_PATH)

    # ---- Date handling ----
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
        valid = df["visit_date"].notna() & (df["visit_date"].dt.year >= 2000)
        has_date = valid.any()
        if has_date:
            df.loc[~valid, "visit_date"] = pd.NaT
            df["visit_day"] = df["visit_date"].dt.normalize()
            df["week_label"] = df["visit_date"].dt.strftime("%d %b")
        else:
            df["visit_day"] = pd.NaT
            df["week_label"] = "ไม่ระบุ"
    else:
        df["visit_day"] = pd.NaT
        df["week_label"] = "ไม่ระบุ"
        has_date = False

    # ---- Blood pressure ----
    if "bp_raw" in df.columns:
        bp_split = df["bp_raw"].str.split(" / ", expand=True)
        systolic = pd.to_numeric(bp_split[0], errors="coerce")
        diastolic = (
            pd.to_numeric(bp_split[1], errors="coerce")
            if bp_split.shape[1] > 1
            else pd.Series(np.nan, index=df.index)
        )
        zero_junk = (systolic == 0) & (diastolic == 0)
        df["systolic"] = systolic.mask(zero_junk)
        df["diastolic"] = diastolic.mask(zero_junk)
    else:
        df["systolic"] = np.nan
        df["diastolic"] = np.nan

    # ---- Gender / BMI ----
    df["gender"] = df.get("gender", pd.Series(["ไม่ระบุ"] * len(df)))
    df["gender"] = df["gender"].fillna("ไม่ระบุ")
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        valid_bmi = df["bmi"].dropna()
        df["bmi_imputed"] = df["bmi"].isna()
        median = valid_bmi.median() if not valid_bmi.empty else 22.0
        df["bmi"] = df["bmi"].fillna(median)
    else:
        df["bmi"] = 22.0
        df["bmi_imputed"] = True

    # ---- Age ----
    if "age_at_visit" in df.columns and df["age_at_visit"].notna().any():
        df["age_at_visit"] = pd.to_numeric(df["age_at_visit"], errors="coerce")
        median_age = df["age_at_visit"].dropna().median() or 35.0
        df["age_at_visit"] = df["age_at_visit"].fillna(median_age)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(
            df["age_at_visit"],
            bins=[0, 25, 45, 60, 120],
            labels=["<25 ปี", "25-45 ปี", "46-60 ปี", ">60 ปี"]
        ).astype(str).replace("nan", "ไม่ระบุ")
        df["pyramid_group"] = pd.cut(
            df["age_at_visit"],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
            right=False,
            labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        ).astype(str)
        has_age = True
    else:
        df["age_group"] = "ไม่ระบุ"
        df["pyramid_group"] = "ไม่ระบุ"
        df["age_at_visit"] = 0
        df["is_adult"] = True
        has_age = False

    # ---- BP level ----
    bp_cat = pd.cut(df["systolic"], bins=[-1, 120, 139, 300],
                    labels=["ปกติ (<120)", "เฝ้าระวัง (120-139)", "สูง (≥140)"]).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat

    # ---- Critical risk ----
    df["critical_risk"] = ((df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140))).astype(int)

    # ---- Disease group default ----
    df["disease_group"] = df.get("disease_group", "ทั่วไป")

    # ---- Diagnosis cleaning ----
    def clean_diag(x):
        if pd.isna(x):
            return "ไม่ระบุ"
        s = str(x).strip()
        return "ไม่ระบุ" if s in (":", "", "-") else s
    df["diagnosis_clean"] = df.get("diagnosis_text", pd.Series(["ไม่ระบุ"] * len(df))).apply(clean_diag)

    # ---- Clinic name ----
    df["clinic_name"] = df.get("clinic_name", "ไม่ระบุ")
    return df, has_date, has_age

# ---------------------------------------------------------------
# Model – cached after first training
# ---------------------------------------------------------------
@st.cache_resource
def get_scoring_engine(df):
    if df.empty or len(df) < 10:
        return None
    feats = ["visits", "bmi", "systolic", "gender_code"]
    X = df[feats]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

# ---------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------
df, has_date, has_age = load_dataset()
if df is None:
    st.error("⚠️ ไม่พบไฟล์ `visits_cleaned.csv` กรุณาตรวจสอบการเชื่อมต่อข้อมูล")
    st.stop()

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Filter Scope")
    disease_sel = st.multiselect(
        "กลุ่มโรค",
        sorted(df["disease_group"].unique()),
        default=sorted(df["disease_group"].unique()),
    )
    gender_sel = st.multiselect(
        "เพศ",
        sorted(df["gender"].unique()),
        default=sorted(df["gender"].unique()),
    )
    if has_date:
        min_date, max_date = df["visit_date"].dropna().min().date(), df["visit_date"].dropna().max().date()
        date_filter = st.date_input("ช่วงวันที่บันทึก", (min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_filter = None
    chronic_scope = st.checkbox("เฉพาะคลินิกกลุ่มโรคเรื้อรัง/คัดกรองสุขภาพ", value=False)
    st.markdown("---")

# ---------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------
mask = df["disease_group"].isin(disease_sel) & df["gender"].isin(gender_sel)
if date_filter and isinstance(date_filter, (list, tuple)):
    start_d, end_d = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    mask &= df["visit_date"].between(start_d, end_d) | df["visit_date"].isna()
if chronic_scope:
    chronic_clinics = [
        "อายุรกรรมChronic -S(โครงการสสส.)",
        "Refillยา (Drug Chronic Clinic)",
        "ตรวจสุขภาพผู้ประกันตน-โครงการ",
        "ตรวจสุขภาพ ฉีดวัคซีน",
        "ตรวจสุขภาพบริษัท",
        "อายุรกรรมโรคหัวใจ-S",
    ]
    mask &= df["clinic_name"].isin(chronic_clinics)

df_view = df[mask]
if df_view.empty:
    st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก")
    st.stop()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
as_of = df["visit_date"].max()
as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"
st.markdown(
    f"""
    <div class='header-bar'>
        <div>
            <div class='header-title'>🏥 Clinical & Growth Command Center</div>
            <div class='header-sub'>ภาพรวมตัวชี้วัดสุขภาพ การจัดการกลุ่มเสี่ยง และโอกาสขยายผลสู่แพ็กจ์ตรวจสุขภาพเชิงป้องกัน</div>
        </div>
        <span class='asof-chip'>ข้อมูลล่าสุด {as_of_str} · {df['clinic_name'].nunique()} คลินิก</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Quality chips (single‑day share, missing BP, pediatric, uncoded)
# ---------------------------------------------------------------
single_day_share = 0.0
if has_date:
    valid_days = df["visit_day"].dropna()
    if not valid_days.empty:
        single_day_share = (valid_days == valid_days.mode()[0]).mean() * 100
missing_bp_share = df["systolic"].isna().mean() * 100
pediatric_share = (~df["is_adult"]).mean() * 100
uncoded_share = (
    (df["disease_group"] == "อื่น ๆ").mean()
    if "อื่น ๆ" in df["disease_group"].unique()
    else 0
)
st.markdown(
    f"""
    <div class='quality-strip'>
        <div class='quality-chip'>📅 <b>{single_day_share:.0f}%</b> ของบันทึกทั้งหมดกระจุกอยู่วันเดียว ({df['visit_day'].mode()[0].strftime('%d %b') if has_date and not df['visit_day'].dropna().empty else '-'}) — กราฟรายวันด้านล่างจึงไม่สะท้อนแนวโน้มจริง</div>
        <div class='quality-chip'>🩺 <b>{missing_bp_share:.0f}%</b> ของเคสไม่มีค่าความดันโลหิต</div>
        <div class='quality-chip'>🧒 <b>{pediatric_share:.1f}%</b> เป็นผู้ป่วยเด็กอายุ &lt;18 ปี</div>
        <div class='quality-chip'>🏷️ <b>{uncoded_share:.0f}%</b> ของเคสอยู่ในกลุ่ม "อื่น ๆ"</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# KPI Row (cached aggregation)
# ---------------------------------------------------------------
@st.cache_data
def compute_kpis(df_view):
    total_cases = len(df_view)
    unique_pts = df_view["patient_id"].nunique() if "patient_id" in df_view.columns else total_cases
    risk_pts = (
        df_view[df_view["critical_risk"] == 1]["patient_id"].nunique()
        if "patient_id" in df_view.columns
        else 0
    )
    risk_pct = (risk_pts / unique_pts * 100) if unique_pts else 0
    avg_bp = df_view["systolic"].mean()
    bp_measured_pct = df_view["systolic"].notna().mean() * 100
    return total_cases, unique_pts, risk_pts, risk_pct, avg_bp, bp_measured_pct

total_v, unique_pts, risk_pts, risk_pct, avg_bp, bp_measured_pct = compute_kpis(df_view)

# Period‑over‑Period deltas (same logic as original)
delta_visits_pct = delta_patients_pct = delta_risk_pct = None
compare_label = None
delta_insufficient = False
if has_date and date_filter and isinstance(date_filter, (list, tuple)):
    base_mask = df["disease_group"].isin(disease_sel) & df["gender"].isin(gender_sel)
    if chronic_scope:
        base_mask &= df["clinic_name"].isin(chronic_clinics)
    cur_start, cur_end = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    period_len = cur_end - cur_start
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - period_len
    curr = df[base_mask & df["visit_date"].between(cur_start, cur_end)]
    prev = df[base_mask & df["visit_date"].between(prev_start, prev_end)]
    compare_label = f"เทียบ {period_len.days + 1} วันก่อนหน้า"
    if len(prev) >= MIN_SAMPLE_FOR_DELTA:
        delta_visits_pct = (len(curr) - len(prev)) / len(prev) * 100
        if "patient_id" in df.columns:
            prev_pts = prev["patient_id"].nunique()
            cur_pts = curr["patient_id"].nunique()
            if prev_pts >= MIN_SAMPLE_FOR_DELTA:
                delta_patients_pct = (cur_pts - prev_pts) / prev_pts * 100
                prev_risk = prev[prev["critical_risk"] == 1]["patient_id"].nunique()
                cur_risk = curr[curr["critical_risk"] == 1]["patient_id"].nunique()
                prev_risk_pct = (prev_risk / prev_pts) * 100
                cur_risk_pct = (cur_risk / cur_pts) * 100 if cur_pts else 0
                delta_risk_pct = cur_risk_pct - prev_risk_pct
    else:
        delta_insufficient = True

k1, k2, k3, k4, k5 = st.columns(5)

# KPI 1 – Total cases
k1.metric(
    "จำนวนเคสทั้งหมด",
    f"{total_v:,}",
    delta=f"{delta_visits_pct:+.1f}% {compare_label}" if delta_visits_pct is not None else None,
    help="เคสรับบริการทั้งหมดในช่วงที่เลือก",
)

# KPI 2 – Unique patients
k2.metric(
    "ผู้ป่วยที่ Active",
    f"{unique_pts:,}",
    delta=f"{delta_patients_pct:+.1f}% {compare_label}" if delta_patients_pct is not None else None,
    help="คนไข้รายบุคคลในช่วงที่เลือก",
)

# KPI 3 – Avg systolic
k3.metric(
    "Systolic เฉลี่ย",
    f"{avg_bp:.1f} mmHg" if pd.notna(avg_bp) else "N/A",
    help=f"คำนวณจากเคสที่มีข้อมูลจริง ({bp_measured_pct:.0f}% ของเคสในมุมมองนี้)",
)

# KPI 4 – Critical risk pct
k4.metric(
    "กลุ่มเสี่ยงวิกฤต",
    f"{risk_pct:.1f}%",
    delta=f"{delta_risk_pct:+.1f} pp {compare_label}" if delta_risk_pct is not None else f"{risk_pts:,} คน",
    delta_color="inverse",
    help=f"{risk_pts:,} คน อายุ≥18 ปี มี BMI≥25 และ Systolic≥140",
)

# KPI 5 – Lead‑score count (computed later after model training)
high_lead_count = 0
est_pipeline_value = 0
if "patient_id" in df.columns:
    summary_pts = (
        df.groupby("patient_id")
        .agg(
            visits=("visit_id", "count") if "visit_id" in df.columns else ("patient_id", "count"),
            bmi=("bmi", "mean"),
            systolic=("systolic", "mean"),
            gender_code=("gender_code", "first"),
        )
        .round(1)
    )
    summary_pts["systolic"] = summary_pts["systolic"].fillna(df["systolic"].median())
    summary_pts["target"] = ((summary_pts["visits"] >= 3) | (summary_pts["systolic"] >= 135)).astype(int)
    clf = get_scoring_engine(summary_pts)
    if clf is not None:
        probs = clf.predict_proba(summary_pts[["visits", "bmi", "systolic", "gender_code"]])[:, 1]
        summary_pts["lead_score"] = (probs * 100).astype(int)
        high_lead_count = (summary_pts["lead_score"] >= 60).sum()
        est_pipeline_value = high_lead_count * 3000
k5.metric("โอกาสขายแพ็กจ์", f"{high_lead_count:,} ราย", help=f"มูลค่าประมาณการ ~{est_pipeline_value:,.0f} บาท")

if delta_insufficient:
    st.caption(f"ℹ️ ไม่แสดง % เปลี่ยนแปลงเทียบช่วงก่อนหน้า เพราะข้อมูลน้อยกว่า {MIN_SAMPLE_FOR_DELTA} เคส")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sunburst + BP pie (main visual row)
# ---------------------------------------------------------------
h1, h2 = st.columns([1.7, 1])
with h1:
    st.markdown("<div class='panel-title'>🔬 โครงสร้างการวินิจฉัยจริง (กลุ่มโรค → รหัสวินิจฉัย)</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-sub'>{uncoded_share:.0f}% ของเคสอยู่ใน \"อื่น ๆ\" — ดูผัง Sunburst (Top {SUNBURST_TOP_N} รายการต่อกลุ่ม)</div>",
        unsafe_allow_html=True,
    )
    frames = []
    for grp, sub in df_view.groupby("disease_group"):
        cnts = sub["diagnosis_clean"].value_counts()
        top = cnts.head(SUNBURST_TOP_N)
        rest = cnts.iloc[SUNBURST_TOP_N:].sum()
        for label, cnt in top.items():
            short = label if len(label) <= 28 else label[:26] + "…"
            frames.append({"disease_group": grp, "diagnosis": short, "count": int(cnt)})
        if rest > 0:
            frames.append({"disease_group": grp, "diagnosis": "อื่นๆ ในกลุ่มนี้", "count": int(rest)})
    sun_df = pd.DataFrame(frames)
    if not sun_df.empty:
        fig_sun = px.sunburst(
            sun_df,
            path=["disease_group", "diagnosis"],
            values="count",
            color="disease_group",
            color_discrete_map=DISEASE_COLORS,
        )
        fig_sun.update_layout(margin=dict(l=0, r=0, t=6, b=6), height=420, paper_bgcolor="rgba(0,0,0,0)")
        fig_sun.update_traces(textfont_size=11, insidetextorientation="radial")
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.caption("ไม่มีข้อมูลตามตัวกรองที่เลือก")
with h2:
    st.markdown("<div class='panel-title'>🩺 สัดส่วนระดับความดันโลหิต</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-sub'>รวมหมวด \"ไม่มีข้อมูล\" แยกออกมาชัดเจน ({missing_bp_share:.0f}% ไม่เคยบันทึก)</div>",
        unsafe_allow_html=True,
    )
    bp_counts = df_view["bp_level"].value_counts().reset_index()
    bp_counts.columns = ["Level", "Count"]
    fig_pie = px.pie(
        bp_counts,
        names="Level",
        values="Count",
        hole=0.55,
        color="Level",
        color_discrete_map=BP_COLORS,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=9)),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    if risk_pts > 0 and "patient_id" in df_view.columns:
        high_risk = df_view[df_view["critical_risk"] == 1][["patient_id", "disease_group", "bmi", "systolic", "diastolic"]].drop_duplicates("patient_id")
        high_risk = high_risk.sort_values(["systolic", "bmi"], ascending=False).head(4)
        high_risk.columns = ["ID", "กลุ่มโรค", "BMI", "Sys", "Dia"]
        st.dataframe(high_risk, use_container_width=True, hide_index=True, height=165)
        csv_bytes = high_risk.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Export ทั้งหมด", csv_bytes, file_name="critical_risk_patients.csv", mime="text/csv", use_container_width=True
        )
    else:
        st.success("✅ ไม่พบคนไข้ในเกณฑ์ความเสี่ยงวิกฤต")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Bottom three panels: Pyramid, Top clinics, BMI‑BP scatter
# ---------------------------------------------------------------
p1, p2, p3 = st.columns([1, 1.1, 1.2])
with p1:
    st.markdown("<div class='panel-title'>👥 พีระมิดประชากรผู้ป่วย</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-sub'>อายุ x เพศ</div>", unsafe_allow_html=True)
    order = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    pyr = df_view.groupby(["pyramid_group", "gender"]).size().reset_index(name="count")
    male = pyr[pyr["gender"] == "ช"].set_index("pyramid_group")["count"]
    female = pyr[pyr["gender"] == "ญ"].set_index("pyramid_group")["count"]
    male_vals = [int(male.get(l, 0)) for l in order]
    female_vals = [int(female.get(l, 0)) for l in order]
    fig_pyr = go.Figure()
    fig_pyr.add_trace(
        go.Bar(
            y=order,
            x=[-v for v in male_vals],
            name="ชาย",
            orientation="h",
            marker_color=TEAL,
            customdata=male_vals,
            hovertemplate="ชาย %{y}: %{customdata} คน<extra></extra>",
        )
    )
    fig_pyr.add_trace(
        go.Bar(
            y=order,
            x=female_vals,
            name="หญิง",
            orientation="h",
            marker_color="#D97AA0",
            customdata=female_vals,
            hovertemplate="หญิง %{y}: %{customdata} คน<extra></extra>",
        )
    )
    max_v = max(male_vals + female_vals) if (male_vals + female_vals) else 1
    fig_pyr.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        xaxis=dict(
            tickvals=[-max_v, -max_v / 2, 0, max_v / 2, max_v],
            ticktext=[str(int(max_v)), str(int(max_v / 2)), "0", str(int(max_v / 2)), str(int(max_v))],
            title=None,
            gridcolor="#F1F5F9",
        ),
    )
    st.plotly_chart(fig_pyr, use_container_width=True)

with p2:
    st.markdown("<div class='panel-title'>🏢 คลินิกที่มีผู้ป่วยมากที่สุด</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-sub'>Top 10 จาก 69 คลินิก</div>", unsafe_allow_html=True)
    top_clinics = df_view["clinic_name"].value_counts().head(10).reset_index()
    top_clinics.columns = ["คลินิก", "จำนวนเคส"]
    top_clinics["label"] = top_clinics["คลินิก"].apply(lambda x: x if len(x) <= 26 else x[:24] + "…")
    fig_clinic = px.bar(
        top_clinics.sort_values("จำนวนเคส"),
        x="จำนวนเคส",
        y="label",
        orientation="h",
        color="จำนวนเคส",
        color_continuous_scale=[[0, "#CFE3DF"], [1, TEAL]],
    )
    fig_clinic.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    fig_clinic.update_yaxes(title=None)
    st.plotly_chart(fig_clinic, use_container_width=True)

with p3:
    st.markdown("<div class='panel-title'>⚖️ BMI กับความดันโลหิต (รายเคส)</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-sub'>เฉพาะผู้ใหญ่อายุ≥18 ปี ที่มีค่าความดันจริง — ขนาดจุด = ความถี่การมาใช้บริการ</div>", unsafe_allow_html=True)
    scatter_df = df_view[df_view["is_adult"] & df_view["systolic"].notna() & ~df_view["bmi_imputed"]].copy()
    if "patient_id" in scatter_df.columns:
        scatter_df["visit_freq"] = scatter_df.groupby("patient_id")["patient_id"].transform("count")
    else:
        scatter_df["visit_freq"] = 1
    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x="bmi",
            y="systolic",
            color="disease_group",
            size="visit_freq",
            color_discrete_map=DISEASE_COLORS,
            hover_data={"patient_id": True} if "patient_id" in scatter_df.columns else None,
        )
        fig_scatter.add_hline(y=140, line_dash="dot", line_color=RED, opacity=0.5)
        fig_scatter.add_vline(x=25, line_dash="dot", line_color=RED, opacity=0.5)
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=330,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=8)),
        )
        fig_scatter.update_xaxes(title="BMI", gridcolor="#F1F5F9")
        fig_scatter.update_yaxes(title="Systolic (mmHg)", gridcolor="#F1F5F9")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.caption("ไม่มีข้อมูลเพียงพอตามตัวกรองที่เลือก")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Trend row + Lead score card + Action cards
# ---------------------------------------------------------------
t1, t2, t3 = st.columns([1.3, 1.1, 1.6])
with t1:
    st.markdown("<div class='panel-title'>📈 ปริมาณเคสรายวัน</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='panel-sub'>⚠️ {single_day_share:.0f}% กระจุกวันเดียว — ควรระวังการตีความ</div>", unsafe_allow_html=True)
    if has_date:
        weekly = (
            df_view.dropna(subset=["visit_date"])
            .groupby(["visit_day", "disease_group"])
            .size()
            .reset_index(name="count")
            .sort_values("visit_day")
        )
        weekly["week_label"] = weekly["visit_day"].dt.strftime("%d %b")
        day_order = weekly["week_label"].drop_duplicates().tolist()
        fig_trend = px.bar(
            weekly,
            x="week_label",
            y="count",
            color="disease_group",
            category_orders={"week_label": day_order},
            color_discrete_map=DISEASE_COLORS,
            barmode="stack",
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=210,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        fig_trend.update_xaxes(title=None, showgrid=False)
        fig_trend.update_yaxes(title=None, gridcolor="#F1F5F9")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("ไม่มีข้อมูลวันที่")

    st.markdown("<div class='panel-title' style='margin-top:4px;'>📉 แนวโน้มกลุ่มเสี่ยงวิกฤต</div>", unsafe_allow_html=True)
    if has_date:
        daily = (
            df_view.dropna(subset=["visit_date"])
            .groupby("visit_day")
            .apply(
                lambda g: pd.Series(
                    {
                        "total": g["patient_id"].nunique() if "patient_id" in g.columns else len(g),
                        "risk": g[g["critical_risk"] == 1]["patient_id"].nunique()
                        if "patient_id" in g.columns
                        else 0,
                    }
                )
            )
            .reset_index()
        )
        daily = daily.sort_values("visit_day")
        daily["risk_pct"] = np.where(
            daily["total"] >= MIN_SAMPLE_FOR_TREND,
            np.where(daily["total"] > 0, daily["risk"] / daily["total"] * 100, 0),
            np.nan,
        )
        daily["label"] = daily["visit_day"].dt.strftime("%d %b")
        fig_rt = px.line(daily, x="label", y="risk_pct", markers=True)
        fig_rt.update_traces(line_color=RED, connectgaps=False)
        fig_rt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=150,
            margin=dict(l=0, r=0, t=6, b=0),
        )
        fig_rt.update_xaxes(title=None, showgrid=False, categoryorder="array", categoryarray=daily["label"].tolist())
        fig_rt.update_yaxes(title="%")
        st.plotly_chart(fig_rt, use_container_width=True)
        low_days = (daily["total"] < MIN_SAMPLE_FOR_TREND).sum()
        if low_days > 0:
            st.caption(f"⚠️ มี {low_days} วันที่ข้อมูลน้อยกว่า {MIN_SAMPLE_FOR_TREND} คน")
    else:
        st.caption("ไม่มีข้อมูลวันที่")

with t2:
    st.markdown("<div class='panel-title'>🎯 โอกาสนำเสนอแพ็กเกจตรวจสุขภาพ</div>", unsafe_allow_html=True)
    if clf is not None and "patient_id" in df_view.columns:
        avail = [p for p in summary_pts.index if p in df_view["patient_id"].values]
        if avail:
            sel_pid = st.selectbox("เลือก Patient ID", avail[:40], label_visibility="collapsed")
            score = summary_pts.loc[sel_pid, "lead_score"]
            card_bg = "#ECFDF5" if score >= 60 else BG
            card_border = "#A7F3D0" if score >= 60 else BORDER
            text_color = "#065F46" if score >= 60 else MUTED
            badge = "🔥 High Priority: เสนอโปรแกรมตรวจคัดกรอง" if score >= 60 else "🌱 General: ติดตามผลรอบปกติ"
            st.markdown(
                f"""
                <div style='background:{card_bg}; border:1px solid {card_border}; border-radius:8px; padding:10px; margin-top:2px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:0.75rem; font-weight:600; color:{text_color};'>Conversion score</span>
                        <span style='font-size:1.3rem; font-weight:700; color:{text_color}; font-family:'IBM Plex Mono',monospace;'>{score}%</span>
                    </div>
                    <div style='font-size:0.75rem; font-weight:500; color:{text_color}; margin-top:2px;'>{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style='font-size:0.73rem; color:{MUTED}; margin-top:6px;'>
                    💼 กลุ่มเป้าหมายคะแนนสูงรวม: <b>{high_lead_count:,} ราย</b><br>
                    ประมาณการมูลค่า: <b>~{est_pipeline_value:,.0f} บาท</b>
                    <span style='font-size:0.68rem;'>( {high_lead_count:,} × 3,000 บาท/แพ็กจ์ )</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("⚠️ คะแนนอิงกฎ (visits≥3 หรือ systolic≥135) ไม่ใช่พยากรณ์สถิติ")
        else:
            st.caption("ไม่มีผู้ป่วยที่ตรงกับกรองนี้")
    else:
        st.caption("ไม่มีข้อมูลโมเดล")

with t3:
    st.markdown("<div class='panel-title'>⚡ สรุปทิศทางเพื่อการตัดสินใจ</div>", unsafe_allow_html=True)
    scope_note = "เฉพาะคลินิกโรคเรื้อรัง/คัดกรอง" if chronic_scope else "ทุกคลินิก (รวม ER/ทันตกรรม/ล้างไต)"
    st.markdown(
        f"""
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:4px;'>
            <div class='action-card' style='border-left:4px solid {INK};'>
                <b>1. มาตรการเชิงคลินิก</b><br>
                ผู้ป่วยเสี่ยงสูง {risk_pts:,} ราย (อายุ≥18, BP≥140, BMI≥25) ควรส่งต่อ Fast‑track ตรวจหัวใจและหลอดเลือด
            </div>
            <div class='action-card' style='border-left:4px solid {TEAL};'>
                <b>2. โอกาสสร้างรายได้</b><br>
                คนไข้ {high_lead_count:,} รายพร้อมรับการเสนอขาย ({scope_note}) แนะนำเปิดแคมเปญ Preventive Care Package
            </div>
            <div class='action-card' style='border-left:4px solid {AMBER};'>
                <b>3. คุณภาพข้อมูล</b><br>
                {missing_bp_share:.0f}% ของเคสไม่มีค่าความดัน แนะนำเพิ่มการบันทึก vital signs ให้ครบใน EMR ก่อนใช้ตัวเลขนี้ตัดสินใจเชิงลึก
            </div>
            <div class='action-card' style='border-left:4px solid #9AA6A0;'>
                <b>4. การจัดหมวดโรค</b><br>
                {uncoded_share:.0f}% ยังอยู่ใน "อื่น ๆ" แนะนำขยาย taxonomy disease_group เพื่อวิเคราะห์เชิงลึกต่อได้
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# End of app
