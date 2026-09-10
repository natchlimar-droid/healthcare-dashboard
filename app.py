import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time

# optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# ============================================================
# Design tokens
# ============================================================
INK    = "#0B1B2B"
BG     = "#F7F9F8"
SURFACE = "#FFFFFF"
TEAL   = "#0E5C56"
SAGE   = "#4F7C5B"
AMBER  = "#C87F0A"
RED    = "#B3261E"
MUTED  = "#5B6B6B"

DISEASE_COLORS = {
    "ความดันโลหิตสูง":  TEAL,
    "เบาหวาน":          "#2F6FB5",
    "ไขมันในเลือดสูง":  AMBER,
    "อื่น ๆ":           "#9AA6A0",
}
BP_COLORS = {
    "ปกติ (<120)":         SAGE,
    "เฝ้าระวัง (120-139)": AMBER,
    "สูง (≥140)":          RED,
    "ไม่มีข้อมูล":         "#C7CFCC",
}

MIN_SAMPLE = 5
SUNBURST_TOP_N = 5

# ============================================================
# Page config + CSS
# ============================================================
st.set_page_config(
    page_title="Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {INK}; }}
.stApp {{ background-color: {BG}; }}
.block-container {{ padding-top: 1.1rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }}

div[data-testid="stMetric"] {{
    background: {SURFACE}; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(11,27,43,0.07);
}}
div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem !important;
    font-weight: 600 !important; color: {INK};
}}
div[data-testid="stMetricLabel"] {{
    font-size: 0.8rem !important; font-weight: 600 !important; color: {MUTED} !important;
}}

.header-bar {{ padding-bottom: 12px; margin-bottom: 14px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; }}
.header-title {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
.header-sub {{ font-size: 0.83rem; color: {MUTED}; margin-top: 2px; }}
.asof-chip {{ background: {TEAL}; color: white; padding: 5px 14px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}

.pulse-card {{ background: {SURFACE}; border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 2px 14px rgba(11,27,43,0.08);
    display: flex; align-items: center; gap: 16px; height: 100%; }}
.pulse-dot {{ width: 54px; height: 54px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 1.05rem; color: white; flex-shrink: 0; }}
.pulse-label {{ font-size: 0.78rem; color: {MUTED}; font-weight: 600; }}
.pulse-status {{ font-size: 1.05rem; font-weight: 700; }}

.meter-wrap {{ margin-bottom: 10px; }}
.meter-top {{ display: flex; justify-content: space-between;
    font-size: 0.76rem; color: {MUTED}; margin-bottom: 4px; }}
.meter-track {{ background: #E7ECEA; border-radius: 20px; height: 8px; overflow: hidden; }}
.meter-fill {{ height: 100%; border-radius: 20px; }}
.panel-title {{ font-size: 0.9rem; font-weight: 700; color: {INK}; margin-bottom: 2px; }}
.panel-sub {{ font-size: 0.72rem; color: {MUTED}; margin-bottom: 10px; }}
.drill-banner {{ background: #EAF3F1; border-radius: 10px; padding: 8px 14px;
    font-size: 0.78rem; color: {TEAL}; margin-bottom: 8px; font-weight: 600; }}
.action-btn button {{ background: {TEAL} !important; color: white !important; border: none !important;
    border-radius: 10px !important; box-shadow: 0 2px 8px rgba(14,92,86,.25) !important; font-weight: 600 !important; }}
.action-btn-secondary button {{ background: {SURFACE} !important; color: {TEAL} !important;
    border: 1.5px solid {TEAL} !important; border-radius: 10px !important; font-weight: 600 !important; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    # --- visits_cleaned.csv (required) ---
    if not os.path.exists("visits_cleaned.csv"):
        return None, None, None, False, False

    df = pd.read_csv("visits_cleaned.csv", encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()          # ตัด trailing whitespace ออกจากชื่อคอลัมน์

    # วันที่
    has_date = False
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
        valid = df["visit_date"].notna() & (df["visit_date"].dt.year >= 2000)
        has_date = valid.any()
        df.loc[~valid, "visit_date"] = pd.NaT
    df["year_month"] = df["visit_date"].dt.to_period("M").astype(str) if has_date else "ไม่ระบุ"
    df["visit_day"] = df["visit_date"].dt.normalize() if has_date else pd.NaT

    # ตัวเลข
    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # BP — ใช้ systolic_bp/diastolic_bp ก่อน ถ้าว่างทั้งหมดค่อย parse bp_raw
    for col in ["systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bp_raw" in df.columns:
        need_fallback = ("systolic_bp" not in df.columns) or df["systolic_bp"].isna().all()
        if need_fallback:
            bp_clean = df["bp_raw"].astype(str).str.replace(",", "", regex=False)
            bp_split = bp_clean.str.extract(r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$")
            sys_ = pd.to_numeric(bp_split[0], errors="coerce")
            dia_ = pd.to_numeric(bp_split[1], errors="coerce")
            # mask 0/0 และ out-of-range
            sys_[~sys_.between(60, 250)] = np.nan
            dia_[~dia_.between(30, 150)] = np.nan
            df["systolic_bp"]  = sys_
            df["diastolic_bp"] = dia_

    df.rename(columns={"systolic_bp": "systolic", "diastolic_bp": "diastolic"}, inplace=True)

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("ไม่ระบุ")
    else:
        df["gender"] = "ไม่ระบุ"
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)

    # BMI imputation flag
    df["bmi_imputed"] = df["bmi"].isna() if "bmi" in df.columns else True
    if "bmi" in df.columns:
        med_bmi = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(med_bmi if pd.notna(med_bmi) else 22.0)
    else:
        df["bmi"] = 22.0

    # อายุ
    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()
    if has_age:
        med_age = df["age_at_visit"].median()
        df["age_at_visit"] = df["age_at_visit"].fillna(med_age if pd.notna(med_age) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(
            df["age_at_visit"], bins=[0,29,39,49,59,120],
            labels=["<30 ปี","30-40 ปี","40-50 ปี","50-60 ปี",">60 ปี"]
        ).astype(str).replace("nan","ไม่ระบุ")
        df["pyramid_group"] = pd.cut(
            df["age_at_visit"],
            bins=[0,10,20,30,40,50,60,70,80,120], right=False,
            labels=["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
        ).astype(str)
    else:
        df["is_adult"]      = True
        df["age_group"]     = "ไม่ระบุ"
        df["pyramid_group"] = "ไม่ระบุ"

    # BP category
    bp_cat = pd.cut(
        df["systolic"], bins=[-1,120,139,300],
        labels=["ปกติ (<120)","เฝ้าระวัง (120-139)","สูง (≥140)"]
    ).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat

    # Critical risk flag (ผู้ใหญ่ BMI≥25 + systolic≥140)
    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    # diagnosis clean
    def _clean(x):
        if pd.isna(x): return "ไม่ระบุ"
        s = str(x).strip()
        return "ไม่ระบุ" if s in (":", "", "-") else s
    df["diagnosis_clean"] = df["diagnosis_text"].apply(_clean) if "diagnosis_text" in df.columns else "ไม่ระบุ"

    if "disease_group" not in df.columns:
        df["disease_group"] = "ทั่วไป"
    if "clinic_name" not in df.columns:
        df["clinic_name"] = "ไม่ระบุ"

    # --- monthly_visit_summary.csv (optional) ---
    try:
        monthly = pd.read_csv("monthly_visit_summary.csv", encoding="utf-8-sig")
        monthly["visit_count"] = pd.to_numeric(monthly["visit_count"], errors="coerce")
    except FileNotFoundError:
        monthly = None

    return df, monthly, has_date, has_age


result = load_data()
if result[0] is None:
    st.error("⚠️ ไม่พบไฟล์ `visits_cleaned.csv` — กรุณารัน `clean_visits.py` ก่อน")
    st.stop()

df, monthly_df, has_date, has_age = result


# ============================================================
# Lead scoring (optional sklearn)
# ============================================================
@st.cache_resource
def build_scorer(data):
    if not SKLEARN_AVAILABLE or len(data) < 10:
        return None
    feats = ["visits","bmi","systolic","gender_code"]
    X, y = data[feats], data["target"]
    return RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42).fit(X, y)

summary_pts = high_lead_count = est_pipeline = clf = None
if "patient_id" in df.columns:
    agg_kwargs = {"bmi":("bmi","mean"), "systolic":("systolic","mean"), "gender_code":("gender_code","first"), "age_at_visit":("age_at_visit","max")}
    if "visit_id" in df.columns:
        agg_kwargs["visits"] = ("visit_id","count")
    else:
        agg_kwargs["visits"] = ("patient_id","count")
    summary_pts = df.groupby("patient_id").agg(**agg_kwargs).round(1)
    summary_pts["systolic"] = summary_pts["systolic"].fillna(df["systolic"].median())
    summary_pts["target"] = ((summary_pts["visits"] >= 3) | (summary_pts["systolic"] >= 135)).astype(int)
    clf = build_scorer(summary_pts)
    if clf is not None:
        probs = clf.predict_proba(summary_pts[["visits","bmi","systolic","gender_code"]])[:,1]
        summary_pts["lead_score"] = (probs * 100).astype(int)
        high_lead_count = int((summary_pts["lead_score"] >= 60).sum())
        est_pipeline = high_lead_count * 3000
    else:
        high_lead_count = est_pipeline = 0


# ============================================================
# Sidebar filters
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filter Scope")
    all_diseases = sorted(df["disease_group"].unique())
    disease_sel  = st.multiselect("กลุ่มโรค",  all_diseases,  default=all_diseases)
    all_genders  = sorted(df["gender"].unique())
    gender_sel   = st.multiselect("เพศ",        all_genders,   default=all_genders)

    all_months  = sorted(df["year_month"].dropna().unique())
    month_sel   = st.multiselect("เดือน",       all_months,    default=all_months)

    all_clinics = sorted(df["clinic_name"].dropna().unique())
    clinic_sel  = st.multiselect("คลินิก",      all_clinics,   default=all_clinics)

    if has_date:
        valid_dates = df["visit_date"].dropna()
        if not valid_dates.empty:
            import datetime
            d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
            
            # ขยายกรอบเวลาให้ครอบคลุม 1/5/2026 - 30/6/2026
            cal_min = min(d_min, datetime.date(2026, 5, 1))
            cal_max = max(d_max, datetime.date(2026, 6, 30))
            
            date_filter = st.date_input("ช่วงวันที่", (d_min, d_max), min_value=cal_min, max_value=cal_max)
        else:
            date_filter = None
    else:
        date_filter = None

# Apply filters (guard: ว่าง → ใช้ทั้งหมด)
mask = (
    df["disease_group"].isin(disease_sel  or all_diseases) &
    df["gender"].isin(gender_sel          or all_genders) &
    df["year_month"].isin(month_sel       or all_months) &
    df["clinic_name"].isin(clinic_sel     or all_clinics)
)
if date_filter and isinstance(date_filter, (list,tuple)) and len(date_filter) == 2:
    mask &= df["visit_date"].between(pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])) | df["visit_date"].isna()

dv = df[mask].copy()
if dv.empty:
    st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก — กรุณาปรับตัวกรอง")
    st.stop()


# ============================================================
# Period-over-Period Delta
# ============================================================
delta_v = delta_p = delta_r = compare_label = None
if has_date and date_filter and isinstance(date_filter,(list,tuple)) and len(date_filter)==2:
    base = df["disease_group"].isin(disease_sel or all_diseases) & df["gender"].isin(gender_sel or all_genders)
    cur_s, cur_e = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    span = cur_e - cur_s
    prev_e, prev_s = cur_s - pd.Timedelta(days=1), cur_s - span - pd.Timedelta(days=1)
    cur_d  = df[base & df["visit_date"].between(cur_s, cur_e)]
    prev_d = df[base & df["visit_date"].between(prev_s, prev_e)]
    compare_label = f"เทียบ {span.days+1} วันก่อนหน้า"
    if len(prev_d) >= MIN_SAMPLE:
        delta_v = (len(cur_d) - len(prev_d)) / len(prev_d) * 100
        if "patient_id" in df.columns:
            pp, cp = prev_d["patient_id"].nunique(), cur_d["patient_id"].nunique()
            if pp >= MIN_SAMPLE:
                delta_p = (cp - pp) / pp * 100
                pr = prev_d[prev_d["critical_risk"]==1]["patient_id"].nunique()
                cr = cur_d[cur_d["critical_risk"]==1]["patient_id"].nunique()
                delta_r = (cr/cp*100 if cp else 0) - (pr/pp*100 if pp else 0)


# ============================================================
# Header
# ============================================================
as_of     = df["visit_date"].max()
as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"

st.markdown(f"""
<div class="header-bar">
  <div>
    <div class="header-title">🏥 Clinical Command Center</div>
    <div class="header-sub">ภาพรวมตัวชี้วัดสุขภาพ · การจัดการกลุ่มเสี่ยง · โอกาสขยายผลแพ็กเกจตรวจสุขภาพเชิงป้องกัน</div>
  </div>
  <span class="asof-chip">ข้อมูลล่าสุด {as_of_str} · {df['clinic_name'].nunique()} คลินิก</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Section 1 — Pulse + KPIs
# ============================================================
total_v   = len(dv)
uniq_pts  = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v
risk_pts  = dv[dv["critical_risk"]==1]["patient_id"].nunique() if "patient_id" in dv.columns else 0
risk_pct  = risk_pts / uniq_pts * 100 if uniq_pts else 0
avg_bp    = dv["systolic"].mean()
bp_ok_pct = dv["systolic"].notna().mean() * 100
miss_bp   = dv["systolic"].isna().mean() * 100
uncoded   = (dv["disease_group"]=="อื่น ๆ").mean() * 100 if "อื่น ๆ" in dv["disease_group"].unique() else 0

health_score = round((max(0, 100 - risk_pct*2) + bp_ok_pct) / 2)
if health_score >= 80:   health_status, health_color = "ปกติดี",       SAGE
elif health_score >= 60: health_status, health_color = "เฝ้าระวัง",    AMBER
else:                    health_status, health_color = "ต้องดำเนินการ", RED

pulse_col, kpi_col = st.columns([1, 3])
with pulse_col:
    st.markdown(f"""
    <div class="pulse-card">
      <div class="pulse-dot" style="background:{health_color};">{health_score}</div>
      <div>
        <div class="pulse-label">Health Score</div>
        <div class="pulse-status" style="color:{health_color};">{health_status}</div>
        <div style="font-size:0.68rem;color:{MUTED};margin-top:3px;">คุมความเสี่ยง + ข้อมูลครบถ้วน</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col:
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("จำนวนเคส",        f"{total_v:,}",
              delta=f"{delta_v:+.1f}% {compare_label}" if delta_v is not None else None)
    k2.metric("ผู้รับบริการ",     f"{uniq_pts:,}",
              delta=f"{delta_p:+.1f}% {compare_label}" if delta_p is not None else None)
    k3.metric("Systolic เฉลี่ย", f"{avg_bp:.1f} mmHg" if pd.notna(avg_bp) else "N/A",
              help=f"คำนวณจากเคสที่มีค่าจริง ({bp_ok_pct:.0f}%)")
    k4.metric("กลุ่มเสี่ยงวิกฤต", f"{risk_pct:.1f}%",
              delta=f"{delta_r:+.1f} pp {compare_label}" if delta_r is not None else f"{risk_pts:,} คน",
              delta_color="inverse")

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ============================================================
# Section 2 — Data Quality Indicator
# ============================================================
def quality_meter(label, pct, note):
    c = SAGE if pct >= 80 else (AMBER if pct >= 50 else RED)
    st.markdown(f"""
    <div class="meter-wrap">
      <div class="meter-top">
        <span>{label}</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-weight:600;color:{c};">{pct:.0f}%</span>
      </div>
      <div class="meter-track"><div class="meter-fill" style="width:{max(pct,2):.0f}%;background:{c};"></div></div>
      <div style="font-size:0.68rem;color:{MUTED};margin-top:2px;">{note}</div>
    </div>""", unsafe_allow_html=True)

with st.container(key="card_dq"):
    st.markdown('<div class="panel-title">📊 Data Quality Indicator</div>', unsafe_allow_html=True)
    q1,q2,q3,q4 = st.columns(4)
    with q1: quality_meter("ข้อมูลความดันครบถ้วน",    100-miss_bp,  f"{miss_bp:.0f}% ว่าง — fallback จาก bp_raw")
    with q2: quality_meter("ครอบคลุมกลุ่มผู้ใหญ่",  100-(~dv["is_adult"]).mean()*100, f"{(~dv['is_adult']).mean()*100:.1f}% เป็นเด็ก")
    with q3: quality_meter("จัดหมวดโรคสำเร็จ",        100-uncoded,  f"{uncoded:.0f}% ยังอยู่ใน 'อื่น ๆ'")
    with q4: quality_meter("มีข้อมูล BMI จริง",       100-dv["bmi_imputed"].mean()*100, "ที่เหลือ imputed ด้วยค่ากลาง")

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ============================================================
# Section 3 — Sunburst + Command Action Panel
# ============================================================
def build_sunburst(data, top_n=SUNBURST_TOP_N):
    rows, top_map = [], {}
    for grp, sub in data.groupby("disease_group"):
        counts = sub["diagnosis_clean"].value_counts()
        top = counts.head(top_n)
        rest = counts.iloc[top_n:].sum()
        top_map[grp] = list(top.index)
        for lb,cnt in top.items():
            rows.append({"disease_group":grp,"diagnosis":lb,"count":int(cnt)})
        if rest > 0:
            rows.append({"disease_group":grp,"diagnosis":"อื่นๆ ในกลุ่มนี้","count":int(rest)})
    return pd.DataFrame(rows), top_map

hero_l, hero_r = st.columns([1.7,1])

with hero_l:
    with st.container(key="card_sunburst"):
        st.markdown('<div class="panel-title">🔬 โครงสร้างการวินิจฉัย — คลิกเพื่อเจาะลึก</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-sub">{uncoded:.0f}% ของเคสอยู่ใน "อื่น ๆ" — คลิกวงในดูกลุ่มโรค วงนอกดูรหัสวินิจฉัย</div>', unsafe_allow_html=True)

        sun_df, top_map = build_sunburst(dv)
        if not sun_df.empty:
            fig_sun = px.sunburst(sun_df, path=["disease_group","diagnosis"], values="count",
                                   color="disease_group", color_discrete_map=DISEASE_COLORS)
            fig_sun.update_layout(margin=dict(l=0,r=0,t=6,b=6), height=420, paper_bgcolor="rgba(0,0,0,0)")
            fig_sun.update_traces(textfont_size=11, insidetextorientation="radial")

            click = st.plotly_chart(fig_sun, use_container_width=True,
                                     on_select="rerun", selection_mode="points", key="sun_click")
            sel_label = sel_group = None
            if click and click.selection and click.selection.get("points"):
                pt = click.selection["points"][0]
                sel_label = pt.get("label")
                sel_group = pt.get("parent") or sel_label

            if sel_label:
                if sel_label in DISEASE_COLORS:
                    drill = dv[dv["disease_group"]==sel_label]
                    n = drill["patient_id"].nunique() if "patient_id" in drill.columns else len(drill)
                    st.markdown(f'<div class="drill-banner">🔍 กลุ่ม: {sel_label} · {n} ราย</div>', unsafe_allow_html=True)
                elif sel_label == "อื่นๆ ในกลุ่มนี้":
                    rest_lb = top_map.get(sel_group,[])
                    drill = dv[(dv["disease_group"]==sel_group) & (~dv["diagnosis_clean"].isin(rest_lb))]
                    st.markdown(f'<div class="drill-banner">🔍 อื่นๆ ในกลุ่ม {sel_group} · {len(drill)} เคส</div>', unsafe_allow_html=True)
                else:
                    drill = dv[dv["diagnosis_clean"]==sel_label]
                    st.markdown(f'<div class="drill-banner">🔍 {sel_label} · {len(drill)} เคส</div>', unsafe_allow_html=True)

                show_c = [c for c in ["patient_id","clinic_name","age_at_visit","gender","bmi","systolic"] if c in drill.columns]
                st.dataframe(drill[show_c].head(10), use_container_width=True, hide_index=True, height=180)
            else:
                st.caption("👆 คลิกส่วนใดของผังเพื่อดูรายละเอียด")

with hero_r:
    with st.container(key="card_bp_donut"):
        st.markdown('<div class="panel-title">🩺 ระดับความดันโลหิต</div>', unsafe_allow_html=True)
        bp_pie = dv["bp_level"].value_counts().reset_index()
        bp_pie.columns = ["Level","Count"]
        fig_pie = px.pie(bp_pie, names="Level", values="Count", hole=0.55,
                          color="Level", color_discrete_map=BP_COLORS)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230,
                               margin=dict(l=0,r=0,t=10,b=0),
                               legend=dict(orientation="h",yanchor="bottom",y=-0.3,font=dict(size=9)))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    with st.container(key="card_action"):
        st.markdown('<div class="panel-title">⚡ Command Action Panel</div>', unsafe_allow_html=True)
        high_risk = dv[dv["critical_risk"]==1]
        if "patient_id" in high_risk.columns:
            high_risk = high_risk[["patient_id","disease_group","bmi","systolic","diastolic"]]\
                .drop_duplicates("patient_id").sort_values(["systolic","bmi"],ascending=False)

        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            gen = st.button("🔔 Alert List", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with ac2:
            st.markdown('<div class="action-btn-secondary">', unsafe_allow_html=True)
            sched = st.button("📅 Outreach", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if gen:
            if not high_risk.empty:
                st.success(f"✅ {len(high_risk)} รายการ")
                st.download_button("📥 ดาวน์โหลด CSV",
                                    high_risk.to_csv(index=False).encode("utf-8-sig"),
                                    "alert_list.csv", "text/csv", use_container_width=True)
            else:
                st.info("ไม่มีกลุ่มเสี่ยงวิกฤต")

        if "outreach_q" not in st.session_state:
            st.session_state["outreach_q"] = []
        if sched:
            st.session_state["show_sched"] = True
        if st.session_state.get("show_sched"):
            with st.form("sched_form"):
                ids = st.multiselect("เลือก Patient",
                    high_risk["patient_id"].tolist() if not high_risk.empty else [],
                    default=(high_risk["patient_id"].tolist()[:5] if not high_risk.empty else []))
                d = st.date_input("วันที่นัด")
                note = st.text_area("บันทึก")
                if st.form_submit_button("ยืนยัน"):
                    st.session_state["outreach_q"].append({"วัน":str(d),"ราย":len(ids),"บันทึก":note})
                    st.session_state["show_sched"] = False
                    st.success(f"✅ กำหนดการ {len(ids)} ราย")
        if st.session_state["outreach_q"]:
            st.dataframe(pd.DataFrame(st.session_state["outreach_q"]),
                          use_container_width=True, hide_index=True)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ============================================================
# Section 4 — Risk Table
# ============================================================
with st.container(key="card_risktable"):
    st.markdown('<div class="panel-title">🚨 รายชื่อผู้ป่วยกลุ่มเสี่ยงสูง</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">ไล่สีตามความรุนแรงของ Systolic BP</div>', unsafe_allow_html=True)
    if not high_risk.empty:
        disp = high_risk.copy()
        disp.columns = [c[:4]+"…" if len(c)>6 else c for c in disp.columns]
        disp.columns = ["ID","กลุ่มโรค","BMI","Sys","Dia"]

        if AGGRID_AVAILABLE:
            row_js = JsCode("""function(p){
                if(p.data.Sys>=180) return{'backgroundColor':'#F3A9A5','color':'#5A0E0E'};
                if(p.data.Sys>=160) return{'backgroundColor':'#F8C6C0'};
                if(p.data.Sys>=140) return{'backgroundColor':'#FBE1DE'};
            }""")
            gb = GridOptionsBuilder.from_dataframe(disp)
            gb.configure_pagination(paginationPageSize=8)
            gb.configure_grid_options(getRowStyle=row_js)
            AgGrid(disp, gridOptions=gb.build(), allow_unsafe_jscode=True,
                   fit_columns_on_grid_load=True, height=260, theme="alpine")
        else:
            def hl(row):
                if row["Sys"] >= 180: return ["background-color:#F3A9A5"]*5
                if row["Sys"] >= 160: return ["background-color:#F8C6C0"]*5
                if row["Sys"] >= 140: return ["background-color:#FBE1DE"]*5
                return [""]*5
            st.dataframe(disp.style.apply(hl,axis=1), use_container_width=True, hide_index=True, height=260)

        st.download_button(f"📥 Export ({len(high_risk)} ราย)",
                            high_risk.to_csv(index=False).encode("utf-8-sig"),
                            "critical_risk.csv","text/csv",use_container_width=True)
    else:
        st.success("✅ ไม่พบคนไข้ในเกณฑ์ความเสี่ยงวิกฤต")

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ============================================================
# Section 5 — Population Pyramid + Clinic + BMI-BP Scatter
# ============================================================
p2,p3 = st.columns([1,1])

with p2:
    with st.container(key="card_clinic"):
        st.markdown('<div class="panel-title">🏢 Top 10 คลินิก</div>', unsafe_allow_html=True)
        cl = dv["clinic_name"].value_counts().head(10).reset_index()
        cl.columns = ["คลินิก","n"]
        cl["lbl"] = cl["คลินิก"].apply(lambda x: x[:24]+"…" if len(x)>26 else x)
        fig_cl = px.bar(cl.sort_values("n"), x="n", y="lbl", orientation="h",
                         color="n", color_continuous_scale=[[0,"#CFE3DF"],[1,TEAL]])
        fig_cl.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              height=310,margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False)
        fig_cl.update_yaxes(title=None)
        fig_cl.update_xaxes(title="จำนวนเคส")
        st.plotly_chart(fig_cl, use_container_width=True)

with p3:
    with st.container(key="card_scatter"):
        st.markdown('<div class="panel-title">⚖️ BMI vs Systolic BP</div>', unsafe_allow_html=True)
        sc = dv[dv["is_adult"] & dv["systolic"].notna() & ~dv["bmi_imputed"]].copy()
        if "patient_id" in sc.columns:
            sc["freq"] = sc.groupby("patient_id")["patient_id"].transform("count")
        else:
            sc["freq"] = 1
        if not sc.empty:
            fig_sc = px.scatter(sc, x="bmi", y="systolic", color="disease_group",
                                 size="freq", color_discrete_map=DISEASE_COLORS, opacity=0.7)
            fig_sc.add_hline(y=140, line_dash="dot", line_color=RED, opacity=0.5)
            fig_sc.add_vline(x=25,  line_dash="dot", line_color=RED, opacity=0.5)
            fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                  height=310,margin=dict(l=0,r=0,t=10,b=0),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=8)))
            fig_sc.update_xaxes(title="BMI",   gridcolor="#F1F5F9")
            fig_sc.update_yaxes(title="Systolic", gridcolor="#F1F5F9")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.caption("ไม่มีข้อมูลเพียงพอ")

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ============================================================
# Section 6 — Smart Package Recommender & Clinical Health Trends
# ============================================================

def recommend_package(row):
    """ประเมินแพ็กเกจจากช่วงอายุและเพศ รวมถึงการคัดกรองพิเศษ"""
    age = row.get("age_at_visit", 35)
    gender_code = row.get("gender_code", 0.5)
    
    # 1. Base Package
    if age >= 50:
        pkg_name = "Longevity Package"
        base_price = 8000
    elif age >= 30:
        pkg_name = "Advanced Package"
        base_price = 5500
    else:
        pkg_name = "Essential Package"
        base_price = 3000
        
    # 2. Special Screening
    screenings = []
    add_on_price = 0
    
    # หญิง (gender_code=1) อายุ >= 40 แนะนำ Mammogram
    if gender_code > 0.5 and age >= 40:
        screenings.append("Mammogram")
        add_on_price += 2000
        
    # ชาย (gender_code=0) อายุ >= 50 แนะนำ PSA
    if gender_code < 0.5 and age >= 50:
        screenings.append("PSA (มะเร็งต่อมลูกหมาก)")
        add_on_price += 2000
        
    return pkg_name, base_price, screenings, add_on_price


def analyze_patient_risk(row):
    """
    คำนวณ Health Score (0-100%) จากข้อมูลความดัน (BP), น้ำหนัก (BMI) และความถี่ในการพบแพทย์
    และเพิ่มคำแนะนำแพ็กเกจ
    """
    score = 100
    reasons = []
    
    sys_val = row.get("systolic", 0)
    bmi_val = row.get("bmi", 22)
    visits_val = row.get("visits", 1)
    
    # หักคะแนนความดัน
    if sys_val >= 160:
        score -= 40
        reasons.append(f"<span style='background:#FBE1DE; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันวิกฤต ({sys_val:.0f})</span>")
    elif sys_val >= 140:
        score -= 25
        reasons.append(f"<span style='background:#F8C6C0; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันสูง ({sys_val:.0f})</span>")
    elif sys_val >= 130:
        score -= 10
        reasons.append(f"<span style='background:#FEF0C7; color:#B54708; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 เฝ้าระวังความดัน ({sys_val:.0f})</span>")
        
    # หักคะแนน BMI
    if bmi_val >= 30:
        score -= 20
        reasons.append(f"<span style='background:#FBE1DE; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 โรคอ้วน ({bmi_val:.1f})</span>")
    elif bmi_val >= 25:
        score -= 10
        reasons.append(f"<span style='background:#FEF0C7; color:#B54708; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 น้ำหนักเกิน ({bmi_val:.1f})</span>")
        
    # หักคะแนนความถี่ (บ่งบอกถึงปัญหาสุขภาพเรื้อรัง)
    if visits_val >= 5:
        score -= 15
        reasons.append(f"<span style='background:#E0F2FE; color:#0369A1; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มารพ. บ่อยผิดปกติ ({visits_val:.0f} ครั้ง)</span>")
    elif visits_val >= 3:
        score -= 5
        reasons.append(f"<span style='background:#F1F5F9; color:#475569; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มีประวัติมาซ้ำ ({visits_val:.0f} ครั้ง)</span>")
        
    # แนะนำแพ็กเกจและเพิ่ม Insight
    pkg_name, base_price, screenings, add_on_price = recommend_package(row)
    
    if screenings:
        score -= 5
        for sc in screenings:
            reasons.append(f"<span style='background:#F3E8FF; color:#7E22CE; border:1px solid #D8B4FE; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🎗️ แนะนำ {sc}</span>")

    if not reasons:
        reasons.append(f"<span style='background:#ECFDF5; color:#065F46; padding:4px 8px; border-radius:12px; font-size:0.7rem; display:inline-block; margin-bottom:4px;'>✅ สุขภาพอยู่ในเกณฑ์ปกติ</span>")
        
    return max(0, score), "".join(reasons), pkg_name, base_price + add_on_price, screenings


col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.markdown('<div class="panel-title">📈 Clinical Health Trend by Age</div>', unsafe_allow_html=True)
    c_tab1, c_tab2, c_tab3 = st.tabs(["📊 Age-Risk Stacked Bar", "🎯 Conversion Donut", "👥 Population Pyramid"])
    
    with c_tab1:
        # Age-Risk Stacked Bar
        risk_age = dv.groupby(["age_group", "critical_risk"]).size().reset_index(name="n")
        risk_age["Risk Level"] = risk_age["critical_risk"].map({0: "ปกติ/เฝ้าระวัง", 1: "High Risk"})
        if not risk_age.empty:
            fig_bar = px.bar(risk_age, x="age_group", y="n", color="Risk Level", barmode="stack",
                             color_discrete_map={"ปกติ/เฝ้าระวัง": SAGE, "High Risk": RED},
                             labels={"age_group": "ช่วงอายุ", "n": "จำนวนคนไข้"})
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=260, margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig_bar.update_xaxes(categoryorder="array", categoryarray=["<30 ปี","30-40 ปี","40-50 ปี","50-60 ปี",">60 ปี"], gridcolor="#F1F5F9")
            fig_bar.update_yaxes(gridcolor="#F1F5F9")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูล Risk แยกตามอายุ")
            
    with c_tab2:
        # Conversion Potential Donut
        if summary_pts is not None and "age_at_visit" in summary_pts.columns:
            def est_pkg(age):
                if age >= 50: return "Longevity (>50)"
                elif age >= 30: return "Advanced (30-50)"
                return "Essential (<30)"
            summary_pts["pkg_type"] = summary_pts["age_at_visit"].apply(est_pkg)
            pkg_counts = summary_pts["pkg_type"].value_counts().reset_index()
            pkg_counts.columns = ["Package", "Count"]
            
            fig_don = px.pie(pkg_counts, names="Package", values="Count", hole=0.55,
                             color="Package", color_discrete_map={
                                 "Longevity (>50)": AMBER,
                                 "Advanced (30-50)": TEAL,
                                 "Essential (<30)": "#2F6FB5"
                             })
            fig_don.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                                  margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.0))
            st.plotly_chart(fig_don, use_container_width=True)
            
            # Text summary
            st.markdown(f"""
            <div style="text-align:center; font-size:0.8rem; color:{MUTED}; margin-top:-10px;">
                มูลค่าคาดการณ์ (Base): <span style="color:{TEAL}; font-weight:700;">฿ {(pkg_counts[pkg_counts["Package"]=="Longevity (>50)"]["Count"].sum() * 8000 + pkg_counts[pkg_counts["Package"]=="Advanced (30-50)"]["Count"].sum() * 5500 + pkg_counts[pkg_counts["Package"]=="Essential (<30)"]["Count"].sum() * 3000):,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ไม่มีข้อมูลผู้ป่วยที่สรุปได้")
            
    with c_tab3:
        # Population Pyramid
        age_order = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
        pyr = dv.groupby(["pyramid_group","gender"]).size().reset_index(name="n")
        m_s = pyr[pyr["gender"]=="ช"].set_index("pyramid_group")["n"]
        f_s = pyr[pyr["gender"]=="ญ"].set_index("pyramid_group")["n"]
        mv = [int(m_s.get(l,0)) for l in age_order]
        fv = [int(f_s.get(l,0)) for l in age_order]
        maxv = max(mv+fv) or 1
        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(y=age_order,x=[-v for v in mv],name="ชาย",orientation="h",
                                  marker_color=TEAL,customdata=mv,
                                  hovertemplate="ชาย %{y}: %{customdata}<extra></extra>"))
        fig_pyr.add_trace(go.Bar(y=age_order,x=fv,name="หญิง",orientation="h",
                                  marker_color="#D97AA0",customdata=fv,
                                  hovertemplate="หญิง %{y}: %{customdata}<extra></extra>"))
        fig_pyr.update_layout(barmode="overlay",paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",height=260,
                               margin=dict(l=0,r=0,t=10,b=0),
                               legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=9)),
                               xaxis=dict(tickvals=[-maxv,-maxv//2,0,maxv//2,maxv],
                                          ticktext=[str(maxv),str(maxv//2),"0",str(maxv//2),str(maxv)],
                                          gridcolor="#F1F5F9"))
        st.plotly_chart(fig_pyr, use_container_width=True)

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 เลือกผู้ป่วยเพื่อประเมิน Package (Search & Select)</div>', unsafe_allow_html=True)
    
    if summary_pts is not None and "patient_id" in dv.columns:
        avail_df = summary_pts[summary_pts.index.isin(dv["patient_id"].values)].reset_index()
        if not avail_df.empty:
            selection = st.dataframe(
                avail_df[["patient_id", "age_at_visit", "visits", "bmi", "systolic"]].rename(
                    columns={"patient_id": "Patient ID", "age_at_visit": "Age", "visits": "Visits", "bmi": "BMI", "systolic": "Systolic"}
                ),
                use_container_width=True,
                hide_index=True,
                height=220,
                on_select="rerun",
                selection_mode="single-row"
            )
            sel_idx = selection.selection.rows
        else:
            sel_idx = []
    else:
        avail_df = None
        sel_idx = []

with col_right:
    st.markdown('<div class="panel-title">📋 Patient Profile & Recommendation</div>', unsafe_allow_html=True)
    
    if sel_idx and avail_df is not None:
        sel_pid = avail_df.iloc[sel_idx[0]]["patient_id"]
        pt_data = summary_pts.loc[sel_pid]
        
        score, reasons_html, pkg_name, total_price, screenings = analyze_patient_risk(pt_data)
        gender_icon = "👩" if pt_data["gender_code"] > 0.5 else "👨"
        
        if score <= 60: c_tx, badge = "#B3261E", "🚨 High Risk"
        elif score <= 80: c_tx, badge = "#B54708", "⚠️ Medium Risk"
        else: c_tx, badge = "#065F46", "🌱 Low Risk"
            
        st.markdown(f"""
        <div style="background:{SURFACE}; border:1px solid #E2E8F0; border-radius:12px; padding:18px; margin-top:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="font-size:2.4rem; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:50%; width:54px; height:54px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">{gender_icon}</div>
                <div>
                    <div style="font-weight:700; color:{INK}; font-size:1.15rem;">{sel_pid}</div>
                    <div style="font-size:0.75rem; color:{MUTED}; margin-bottom:2px;">อายุ: {pt_data['age_at_visit']:.0f} ปี</div>
                    <div style="font-size:0.75rem; font-weight:600; color:{c_tx}; background:#F8FAFC; padding:2px 6px; border-radius:6px; display:inline-block;">{badge}</div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.7rem; color:{MUTED}; font-weight:600;">Health Score</div>
                <div style="font-size:1.6rem; font-weight:700; color:{c_tx}; font-family:'IBM Plex Mono',monospace; line-height:1.1;">{score}%</div>
            </div>
          </div>
          
          <div style="background:#F0F9FF; border-left:4px solid #0284C7; padding:10px 12px; border-radius:6px; margin-bottom:14px;">
              <div style="font-size:0.7rem; color:#0284C7; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:2px;">💎 Recommended Package</div>
              <div style="font-size:1.05rem; font-weight:700; color:{INK};">{pkg_name}</div>
              <div style="font-size:0.85rem; font-weight:600; color:{MUTED}; font-family:'IBM Plex Mono',monospace;">Est. ฿ {total_price:,.0f}</div>
          </div>
          
          <div style="font-size:0.78rem; font-weight:600; color:{INK}; margin-bottom:8px;">💡 AI Analysis Insights:</div>
          <div style="margin-bottom:12px; line-height:1.6;">
            {reasons_html}
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="action-btn" style="margin-top:14px;">', unsafe_allow_html=True)
        if st.button("✨ Generate Personalized Proposal", use_container_width=True):
            with st.spinner("กำลังเชื่อมต่อระบบ CRM/LINE API..."):
                import time
                time.sleep(1)
                try:
                    payload = {
                        "patient_id": sel_pid,
                        "health_score": score,
                        "risk_level": badge,
                        "recommended_package": pkg_name,
                        "special_screening": screenings,
                        "estimated_price": total_price
                    }
                    st.success(f"✅ ส่งข้อมูลให้ระบบเรียบร้อย! (Webhook API Mockup)\n\n**Package:** {pkg_name}\n**Add-on:** {', '.join(screenings) if screenings else 'ไม่มี'}")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:12px; padding:30px 16px; text-align:center; color:{MUTED};">
            <div style="font-size:2rem; margin-bottom:10px;">👈</div>
            <div style="font-size:0.9rem; font-weight:600;">คลิกเลือกผู้ป่วยจากตารางด้านซ้าย</div>
            <div style="font-size:0.8rem; margin-top:4px;">เพื่อดู Health Score และการแนะนำแพ็กเกจที่เหมาะสม</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ============================================================
# Section 7 — Raw Data
# ============================================================
with st.expander("📋 ดูข้อมูลดิบ (visits_cleaned)", expanded=False):
    show_c = [c for c in ["visit_date","visit_id","patient_id","gender","age_at_visit",
                            "clinic_name","diagnosis_clean","disease_group",
                            "systolic","diastolic","bmi"] if c in dv.columns]
    st.dataframe(dv[show_c].sort_values("visit_date",ascending=False) if "visit_date" in dv.columns else dv[show_c],
                  use_container_width=True, hide_index=True)
    st.caption(f"แสดง {len(dv):,} แถว")
