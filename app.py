import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# ============================================================
# 1. ตั้งค่าหน้าเว็บและ Design Tokens
#    (ค่าพื้นฐานยังอยู่ใน .streamlit/config.toml ที่แนบมาด้วย — ส่วนที่ config.toml
#    ทำไม่ได้ เช่น ฟอนต์ Google Font, shadow, badge ต้องยังคง inject CSS เพิ่มที่นี่)
# ============================================================
INK = "#0B1B2B"
BG = "#F7F9F8"
SURFACE = "#FFFFFF"
TEAL = "#0E5C56"
SAGE = "#4F7C5B"
AMBER = "#C87F0A"
RED = "#B3261E"
MUTED = "#5B6B6B"

DISEASE_COLORS = {
    'ความดันโลหิตสูง': TEAL,
    'เบาหวาน': "#2F6FB5",
    'ไขมันในเลือดสูง': AMBER,
    'อื่น ๆ': "#9AA6A0",
}
BP_COLORS = {
    'ปกติ (<120)': SAGE,
    'เฝ้าระวัง (120-139)': AMBER,
    'สูง (≥140)': RED,
    'ไม่มีข้อมูล': "#C7CFCC",
}

st.set_page_config(page_title="Clinical Command Center", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {INK}; }}
    .stApp {{ background-color: {BG}; }}
    .block-container {{ padding-top: 1.1rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }}

    /* Floating card look: shadow แทน border หนาๆ ตามที่ขอ
       ใช้ st.container(key="card_...") แทนการเปิด/ปิด <div> คนละ st.markdown() —
       เพราะ Streamlit render แต่ละ st.markdown() เป็นก้อนแยกกัน เปิดปิด div ข้าม call
       จะเจอปัญหากล่องขาว/ว่างเปล่า (เบราว์เซอร์ auto-close tag ที่ไม่ครบในแต่ละก้อน) */
    div[data-testid="stMetric"], .agrid-wrap, [class*="st-key-card_"] {{
        background: {SURFACE}; border-radius: 14px; padding: 16px 18px;
        border: none; box-shadow: 0 2px 10px rgba(11,27,43,0.06);
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem !important;
        font-weight: 600 !important; color: {INK};
    }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.8rem !important; font-weight: 600 !important; color: {MUTED} !important; }}

    .header-bar {{ padding-bottom: 12px; margin-bottom: 14px; display: flex;
        justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; }}
    .header-title {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
    .header-sub {{ font-size: 0.83rem; color: {MUTED}; margin-top: 2px; }}
    .asof-chip {{ background: {TEAL}; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}

    /* Pulse: health-score badge ใหญ่ มองปราดเดียวรู้สถานะ */
    .pulse-card {{ background: {SURFACE}; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 2px 14px rgba(11,27,43,0.08); display: flex; align-items: center; gap: 16px; height: 100%; }}
    .pulse-dot {{ width: 54px; height: 54px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1.05rem; color: white; flex-shrink: 0; }}
    .pulse-label {{ font-size: 0.78rem; color: {MUTED}; font-weight: 600; }}
    .pulse-status {{ font-size: 1.05rem; font-weight: 700; }}

    /* Data Quality meters */
    .meter-wrap {{ margin-bottom: 10px; }}
    .meter-top {{ display: flex; justify-content: space-between; font-size: 0.76rem; color: {MUTED}; margin-bottom: 4px; }}
    .meter-track {{ background: #E7ECEA; border-radius: 20px; height: 8px; overflow: hidden; }}
    .meter-fill {{ height: 100%; border-radius: 20px; }}

    .panel-title {{ font-size: 0.9rem; font-weight: 700; color: {INK}; margin-bottom: 2px; }}
    .panel-sub {{ font-size: 0.72rem; color: {MUTED}; margin-bottom: 10px; }}

    .action-btn button {{ background: {TEAL} !important; color: white !important; border: none !important;
        border-radius: 10px !important; box-shadow: 0 2px 8px rgba(14,92,86,0.25) !important; font-weight: 600 !important; }}
    .action-btn-secondary button {{ background: {SURFACE} !important; color: {TEAL} !important;
        border: 1.5px solid {TEAL} !important; border-radius: 10px !important; font-weight: 600 !important; }}

    .drill-banner {{ background: #EAF3F1; border-radius: 10px; padding: 8px 14px; font-size: 0.78rem;
        color: {TEAL}; margin-bottom: 8px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

MIN_SAMPLE_FOR_DELTA = 5
MIN_SAMPLE_FOR_TREND = 5
SUNBURST_TOP_N = 5

# ============================================================
# 2. Pipeline เตรียมข้อมูล
# ============================================================
@st.cache_data
def load_clean_dataset():
    path = "visits_cleaned.csv"
    if not os.path.exists(path):
        return None, False, False

    df = pd.read_csv(path)

    has_date = False
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        valid_date_mask = df['visit_date'].notna() & (df['visit_date'].dt.year >= 2000)
        has_date = valid_date_mask.any()
        if has_date:
            df.loc[~valid_date_mask, 'visit_date'] = pd.NaT
            df['visit_day'] = df['visit_date'].dt.normalize()
            df['week_label'] = df['visit_date'].dt.strftime('%d %b')
        else:
            df['visit_day'] = pd.NaT
            df['week_label'] = 'ไม่ระบุ'
    else:
        df['visit_day'] = pd.NaT
        df['week_label'] = 'ไม่ระบุ'

    # ความดันโลหิต: คอลัมน์ systolic_bp/diastolic_bp ต้นฉบับว่างเปล่า 100% ใช้ bp_raw แทน
    # "0.00 / 0.00" คือค่ายังไม่ได้วัด ต้อง mask เป็นค่าว่างเช่นเดียวกับ NaN
    if 'bp_raw' in df.columns:
        bp_split = df['bp_raw'].str.split(' / ', expand=True)
        systolic = pd.to_numeric(bp_split[0], errors='coerce')
        diastolic = pd.to_numeric(bp_split[1], errors='coerce') if bp_split.shape[1] > 1 else pd.Series(np.nan, index=df.index)
        zero_junk = (systolic == 0) & (diastolic == 0)
        df['systolic'] = systolic.mask(zero_junk)
        df['diastolic'] = diastolic.mask(zero_junk)
    else:
        df['systolic'] = np.nan
        df['diastolic'] = np.nan

    df['gender'] = df['gender'].fillna('ไม่ระบุ') if 'gender' in df.columns else 'ไม่ระบุ'
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        valid_bmi = df['bmi'].dropna()
        df['bmi_imputed'] = df['bmi'].isna()
        df['bmi'] = df['bmi'].fillna(valid_bmi.median() if not valid_bmi.empty else 22.0)
    else:
        df['bmi'] = 22.0
        df['bmi_imputed'] = True

    has_age = 'age_at_visit' in df.columns and df['age_at_visit'].notna().any()
    if has_age:
        df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce')
        valid_age = df['age_at_visit'].dropna()
        df['age_at_visit'] = df['age_at_visit'].fillna(valid_age.median() if not valid_age.empty else 35.0)
        df['is_adult'] = df['age_at_visit'] >= 18
        df['age_group'] = pd.cut(
            df['age_at_visit'], bins=[0, 25, 45, 60, 120],
            labels=['<25 ปี', '25-45 ปี', '46-60 ปี', '>60 ปี']
        ).astype(str).replace('nan', 'ไม่ระบุ')
        df['pyramid_group'] = pd.cut(
            df['age_at_visit'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False,
            labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        ).astype(str)
    else:
        df['age_group'] = 'ไม่ระบุ'
        df['pyramid_group'] = 'ไม่ระบุ'
        df['age_at_visit'] = 0
        df['is_adult'] = True

    bp_cat = pd.cut(df['systolic'], bins=[-1, 120, 139, 300], labels=['ปกติ (<120)', 'เฝ้าระวัง (120-139)', 'สูง (≥140)']).astype(object)
    bp_cat[df['systolic'].isna()] = 'ไม่มีข้อมูล'
    df['bp_level'] = bp_cat

    # เกณฑ์ BMI ผู้ใหญ่ใช้เฉพาะอายุ ≥18 ปี (เด็กใช้เกณฑ์ต่างชุด ไม่ควรปนกัน)
    df['critical_risk'] = (df['is_adult'] & (df['bmi'] >= 25) & (df['systolic'] >= 140)).astype(int)

    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ทั่วไป'

    def clean_diag(x):
        if pd.isna(x):
            return 'ไม่ระบุ'
        x = str(x).strip()
        return 'ไม่ระบุ' if x in (':', '', '-') else x
    df['diagnosis_clean'] = df['diagnosis_text'].apply(clean_diag) if 'diagnosis_text' in df.columns else 'ไม่ระบุ'

    if 'clinic_name' not in df.columns:
        df['clinic_name'] = 'ไม่ระบุ'

    return df, has_date, has_age

df, has_date, has_age = load_clean_dataset()

if df is None:
    st.error("⚠️ ไม่พบไฟล์ `visits_cleaned.csv` กรุณาตรวจสอบการเชื่อมต่อข้อมูล")
    st.stop()

# ============================================================
# 3. โมเดลประเมิน Lead Conversion
# ============================================================
@st.cache_resource
def get_scoring_engine(data_df):
    if data_df.empty or len(data_df) < 10:
        return None
    feats = ['visits', 'bmi', 'systolic', 'gender_code']
    X = data_df[feats]
    y = data_df['target']
    return RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42).fit(X, y)

if 'patient_id' in df.columns:
    summary_pts = df.groupby('patient_id').agg(
        visits=('visit_id', 'count') if 'visit_id' in df.columns else ('patient_id', 'count'),
        bmi=('bmi', 'mean'),
        systolic=('systolic', 'mean'),
        gender_code=('gender_code', 'first')
    ).round(1)
    summary_pts['systolic'] = summary_pts['systolic'].fillna(df['systolic'].median())
    summary_pts['target'] = ((summary_pts['visits'] >= 3) | (summary_pts['systolic'] >= 135)).astype(int)
    clf = get_scoring_engine(summary_pts)

    if clf is not None:
        probs = clf.predict_proba(summary_pts[['visits', 'bmi', 'systolic', 'gender_code']])[:, 1]
        summary_pts['lead_score'] = (probs * 100).astype(int)
        high_lead_count = (summary_pts['lead_score'] >= 60).sum()
        est_pipeline_value = high_lead_count * 3000
    else:
        high_lead_count, est_pipeline_value = 0, 0
else:
    clf, summary_pts = None, None
    high_lead_count, est_pipeline_value = 0, 0

# ============================================================
# 4. แถบตัวกรอง Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filter Scope")
    disease_sel = st.multiselect("กลุ่มโรค", sorted(df['disease_group'].unique()), default=sorted(df['disease_group'].unique()))
    gender_sel = st.multiselect("เพศ", sorted(df['gender'].unique()), default=sorted(df['gender'].unique()))

    if has_date:
        valid_dates = df['visit_date'].dropna()
        if not valid_dates.empty:
            d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
            date_filter = st.date_input("ช่วงวันที่บันทึก", (d_min, d_max), min_value=d_min, max_value=d_max)
        else:
            date_filter = None
    else:
        date_filter = None

    st.markdown("---")
    chronic_scope = st.checkbox(
        "เฉพาะคลินิกกลุ่มโรคเรื้อรัง/คัดกรองสุขภาพ", value=False,
        help="กรองเฉพาะคลินิกที่เกี่ยวข้องกับแพ็กเกจตรวจสุขภาพเชิงป้องกัน (~34% ของเคสทั้งหมด) ตัดคลินิก ER/ทันตกรรม/ล้างไต ออก"
    )

CHRONIC_RELEVANT_CLINICS = [
    'อายุรกรรมChronic -S(โครงการสสส.)', 'Refillยา (Drug Chronic Clinic)',
    'ตรวจสุขภาพผู้ประกันตน-โครงการ', 'ตรวจสุขภาพ ฉีดวัคซีน',
    'ตรวจสุขภาพบริษัท', 'อายุรกรรมโรคหัวใจ-S'
]

mask = df['disease_group'].isin(disease_sel) & df['gender'].isin(gender_sel)
if date_filter and isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    start_d, end_d = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    mask &= df['visit_date'].between(start_d, end_d) | df['visit_date'].isna()
if chronic_scope:
    mask &= df['clinic_name'].isin(CHRONIC_RELEVANT_CLINICS)

df_view = df[mask]

if df_view.empty:
    st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก")
    st.stop()

# ============================================================
# 4b. Period-over-Period Delta
# ============================================================
delta_visits_pct = delta_patients_pct = delta_risk_pct = None
compare_label = None
delta_insufficient_sample = False

if has_date and date_filter and isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    base_mask = df['disease_group'].isin(disease_sel) & df['gender'].isin(gender_sel)
    if chronic_scope:
        base_mask &= df['clinic_name'].isin(CHRONIC_RELEVANT_CLINICS)

    cur_start, cur_end = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    period_len = cur_end - cur_start
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - period_len

    curr_period = df[base_mask & df['visit_date'].between(cur_start, cur_end)]
    prev_period = df[base_mask & df['visit_date'].between(prev_start, prev_end)]
    compare_label = f"เทียบ {period_len.days + 1} วันก่อนหน้า"

    if len(prev_period) >= MIN_SAMPLE_FOR_DELTA:
        delta_visits_pct = (len(curr_period) - len(prev_period)) / len(prev_period) * 100
        if 'patient_id' in df.columns:
            prev_pts = prev_period['patient_id'].nunique()
            curr_pts = curr_period['patient_id'].nunique()
            if prev_pts >= MIN_SAMPLE_FOR_DELTA:
                delta_patients_pct = (curr_pts - prev_pts) / prev_pts * 100
                prev_risk = prev_period[prev_period['critical_risk'] == 1]['patient_id'].nunique()
                curr_risk = curr_period[curr_period['critical_risk'] == 1]['patient_id'].nunique()
                prev_risk_pct = (prev_risk / prev_pts * 100)
                curr_risk_pct = (curr_risk / curr_pts * 100) if curr_pts else 0
                delta_risk_pct = curr_risk_pct - prev_risk_pct
    else:
        delta_insufficient_sample = True

# ============================================================
# 5. Header
# ============================================================
as_of = df['visit_date'].max()
as_of_str = as_of.strftime('%d %b %Y') if pd.notna(as_of) else "ไม่ระบุ"

st.markdown(f"""
<div class="header-bar">
    <div>
        <div class="header-title">🏥 Clinical Command Center</div>
        <div class="header-sub">ภาพรวมตัวชี้วัดสุขภาพ การจัดการกลุ่มเสี่ยง และโอกาสขยายผลสู่แพ็กเกจตรวจสุขภาพเชิงป้องกัน</div>
    </div>
    <span class="asof-chip">ข้อมูลล่าสุด {as_of_str} · {df['clinic_name'].nunique()} คลินิก</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 6. Level 1 — The Pulse: Health Score + KPI
# ============================================================
total_v = len(df_view)
unique_pts = df_view['patient_id'].nunique() if 'patient_id' in df_view.columns else total_v
risk_pts = df_view[df_view['critical_risk'] == 1]['patient_id'].nunique() if 'patient_id' in df_view.columns else 0
risk_pct = (risk_pts / unique_pts * 100) if unique_pts else 0
avg_bp = df_view['systolic'].mean()
bp_measured_pct = df_view['systolic'].notna().mean() * 100

single_day_share = 0.0
if has_date:
    valid_days = df['visit_day'].dropna()
    if not valid_days.empty:
        single_day_share = (valid_days == valid_days.mode()[0]).mean() * 100
missing_bp_share = df['systolic'].isna().mean() * 100
pediatric_share = (~df['is_adult']).mean() * 100
uncoded_share = (df['disease_group'] == 'อื่น ๆ').mean() * 100 if 'อื่น ๆ' in df['disease_group'].unique() else 0

# สูตร Health Score โปร่งใส: เฉลี่ยของ (คุมความเสี่ยงได้ดีแค่ไหน) กับ (ข้อมูลความดันครบแค่ไหน)
# ไม่ใช่ AI black-box — คำนวณจากตัวเลขที่เห็นในหน้านี้ตรงๆ
risk_control_score = max(0, 100 - risk_pct * 2)
health_score = round((risk_control_score + bp_measured_pct) / 2)
if health_score >= 80:
    health_status, health_color = "ปกติดี", SAGE
elif health_score >= 60:
    health_status, health_color = "เฝ้าระวัง", AMBER
else:
    health_status, health_color = "ต้องดำเนินการ", RED

pulse_col, kpi_col = st.columns([1, 3])
with pulse_col:
    st.markdown(f"""
    <div class="pulse-card">
        <div class="pulse-dot" style="background:{health_color};">{health_score}</div>
        <div>
            <div class="pulse-label">Health Score (คุมความเสี่ยง + ข้อมูลครบถ้วน เฉลี่ย 2 ด้าน)</div>
            <div class="pulse-status" style="color:{health_color};">{health_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("จำนวนเคสทั้งหมด", f"{total_v:,}",
        delta=f"{delta_visits_pct:+.1f}% {compare_label}" if delta_visits_pct is not None else None,
        help="เคสรับบริการทั้งหมดในช่วงที่เลือก")
    k2.metric("ผู้ป่วยที่ Active", f"{unique_pts:,}",
        delta=f"{delta_patients_pct:+.1f}% {compare_label}" if delta_patients_pct is not None else None,
        help="คนไข้รายบุคคลในช่วงที่เลือก")
    k3.metric("Systolic เฉลี่ย", f"{avg_bp:.1f} mmHg" if pd.notna(avg_bp) else "N/A",
        help=f"คำนวณจากเคสที่มีข้อมูลจริงเท่านั้น ({bp_measured_pct:.0f}%)")
    k4.metric("กลุ่มเสี่ยงวิกฤต", f"{risk_pct:.1f}%",
        delta=f"{delta_risk_pct:+.1f} pp {compare_label}" if delta_risk_pct is not None else f"{risk_pts:,} คน",
        delta_color="inverse", help=f"{risk_pts:,} คน อายุ≥18 ปี มี BMI≥25 และ Systolic≥140")

if delta_insufficient_sample:
    st.caption(f"ℹ️ ไม่แสดง % เปลี่ยนแปลงเทียบช่วงก่อนหน้า เพราะช่วงก่อนหน้ามีข้อมูลน้อยกว่า {MIN_SAMPLE_FOR_DELTA} เคส")

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ============================================================
# 7. Data Quality Indicator (แถบ meter แทนข้อความเตือน)
# ============================================================
def quality_meter(label, pct, note):
    color = SAGE if pct >= 80 else (AMBER if pct >= 50 else RED)
    st.markdown(f"""
    <div class="meter-wrap">
        <div class="meter-top"><span>{label}</span><span style="font-family:'IBM Plex Mono',monospace; font-weight:600; color:{color};">{pct:.0f}%</span></div>
        <div class="meter-track"><div class="meter-fill" style="width:{max(pct,2):.0f}%; background:{color};"></div></div>
        <div style="font-size:0.68rem; color:{MUTED}; margin-top:2px;">{note}</div>
    </div>
    """, unsafe_allow_html=True)

with st.container(key="card_dq"):
    st.markdown('<div class="panel-title">📊 Data Quality Indicator</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">สถานะความสมบูรณ์ของข้อมูลที่ใช้คำนวณตัวเลขในหน้านี้ — ดูก่อนเชื่อตัวเลข 100%</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        quality_meter("ข้อมูลความดันครบถ้วน", 100 - missing_bp_share, "systolic_bp/diastolic_bp ต้นฉบับว่างเปล่า ใช้ bp_raw แทน")
    with q2:
        quality_meter("การกระจายของวันบันทึก", 100 - single_day_share, f"{single_day_share:.0f}% กระจุกวันเดียว ใช้ระวังตีความเทรนด์")
    with q3:
        quality_meter("ครอบคลุมกลุ่มผู้ใหญ่", 100 - pediatric_share, f"{pediatric_share:.1f}% เป็นเด็ก ไม่รวมเกณฑ์ BMI ผู้ใหญ่")
    with q4:
        quality_meter("การจัดหมวดโรคสำเร็จ", 100 - uncoded_share, f"{uncoded_share:.0f}% ยังอยู่ในหมวด 'อื่น ๆ' ดูราย diagnosis ในผัง Sunburst")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ============================================================
# 8. Hero Row (F-Pattern ซ้ายบน) — Sunburst Drill-down | Command Action Panel
# ============================================================
hero_l, hero_r = st.columns([1.7, 1])

def build_sunburst_data(data, top_n=SUNBURST_TOP_N):
    frames = []
    top_labels_by_group = {}
    for grp, sub in data.groupby('disease_group'):
        counts = sub['diagnosis_clean'].value_counts()
        top = counts.head(top_n)
        rest = counts.iloc[top_n:].sum()
        top_labels_by_group[grp] = list(top.index)
        for label, cnt in top.items():
            frames.append({'disease_group': grp, 'diagnosis': label, 'count': int(cnt)})
        if rest > 0:
            frames.append({'disease_group': grp, 'diagnosis': 'อื่นๆ ในกลุ่มนี้', 'count': int(rest)})
    return pd.DataFrame(frames), top_labels_by_group

with hero_l:
    with st.container(key="card_sunburst"):
        st.markdown('<div class="panel-title">🔬 โครงสร้างการวินิจฉัยจริง — คลิกเพื่อเจาะลึก</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-sub">{uncoded_share:.0f}% ของเคสอยู่ใน "อื่น ๆ" — คลิกวงในเพื่อดูกลุ่มโรค หรือวงนอกเพื่อดูรหัสวินิจฉัยที่แท้จริง</div>', unsafe_allow_html=True)

        sun_df, top_labels_by_group = build_sunburst_data(df_view)
        if not sun_df.empty:
            fig_sun = px.sunburst(sun_df, path=['disease_group', 'diagnosis'], values='count',
                                   color='disease_group', color_discrete_map=DISEASE_COLORS)
            fig_sun.update_layout(margin=dict(l=0, r=0, t=6, b=6), height=420, paper_bgcolor='rgba(0,0,0,0)')
            fig_sun.update_traces(textfont_size=11, insidetextorientation='radial')

            click = st.plotly_chart(fig_sun, use_container_width=True, on_select="rerun",
                                     selection_mode="points", key="sunburst_click")

            selected_label, selected_group = None, None
            if click and click.selection and click.selection.get("points"):
                pt = click.selection["points"][0]
                selected_label = pt.get("label")
                selected_group = pt.get("parent") or selected_label

            if selected_label:
                if selected_label in DISEASE_COLORS:  # คลิกวงใน (disease_group)
                    drill_df = df_view[df_view['disease_group'] == selected_label]
                    st.markdown(f'<div class="drill-banner">🔍 กำลังดูกลุ่ม: {selected_label} · {drill_df["patient_id"].nunique() if "patient_id" in drill_df.columns else len(drill_df)} ราย</div>', unsafe_allow_html=True)
                elif selected_label == 'อื่นๆ ในกลุ่มนี้':
                    rest_labels = top_labels_by_group.get(selected_group, [])
                    drill_df = df_view[(df_view['disease_group'] == selected_group) & (~df_view['diagnosis_clean'].isin(rest_labels))]
                    st.markdown(f'<div class="drill-banner">🔍 กำลังดู: รหัสวินิจฉัยอื่นๆ ในกลุ่ม {selected_group} · {len(drill_df)} เคส</div>', unsafe_allow_html=True)
                else:  # คลิกรหัสวินิจฉัยเฉพาะ
                    drill_df = df_view[df_view['diagnosis_clean'] == selected_label]
                    st.markdown(f'<div class="drill-banner">🔍 กำลังดู: {selected_label} · {len(drill_df)} เคส</div>', unsafe_allow_html=True)

                show_cols = [c for c in ['patient_id', 'clinic_name', 'age_at_visit', 'gender', 'bmi', 'systolic'] if c in drill_df.columns]
                st.dataframe(drill_df[show_cols].head(10), use_container_width=True, hide_index=True, height=180)
            else:
                st.caption("👆 ยังไม่ได้เลือก — คลิกส่วนใดส่วนหนึ่งของผังด้านบนเพื่อดูรายละเอียด")
        else:
            st.caption("ไม่มีข้อมูลตามตัวกรองที่เลือก")

with hero_r:
    with st.container(key="card_action"):
        st.markdown('<div class="panel-title">⚡ Command Action Panel</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">ปุ่มด้านล่างเป็นการทำงานจริงในหน้านี้ (บันทึกชั่วคราวระหว่างเซสชัน) — ยังไม่เชื่อมระบบ EMR/CRM จริง</div>', unsafe_allow_html=True)

        if 'patient_id' in df_view.columns:
            high_risk_full = df_view[df_view['critical_risk'] == 1][[
                'patient_id', 'disease_group', 'bmi', 'systolic', 'diastolic'
            ]].drop_duplicates('patient_id').sort_values(['systolic', 'bmi'], ascending=False)
        else:
            high_risk_full = pd.DataFrame()

        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            gen_clicked = st.button("🔔 Generate Alert List", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with ac2:
            st.markdown('<div class="action-btn-secondary">', unsafe_allow_html=True)
            sched_clicked = st.button("📅 Schedule Outreach", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if gen_clicked:
            if not high_risk_full.empty:
                st.success(f"✅ สร้างรายชื่อแจ้งเตือนแล้ว {len(high_risk_full)} ราย")
                csv_data = high_risk_full.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 ดาวน์โหลด Alert List (CSV)", csv_data, "critical_risk_alert_list.csv",
                                    "text/csv", use_container_width=True)
            else:
                st.info("ไม่มีผู้ป่วยกลุ่มเสี่ยงวิกฤตในตัวกรองปัจจุบัน")

        if 'outreach_queue' not in st.session_state:
            st.session_state['outreach_queue'] = []

        if sched_clicked:
            st.session_state['show_outreach_form'] = True

        if st.session_state.get('show_outreach_form'):
            with st.form("outreach_form"):
                pick_ids = st.multiselect("เลือกผู้ป่วย", high_risk_full['patient_id'].tolist() if not high_risk_full.empty else [],
                                           default=high_risk_full['patient_id'].tolist()[:5] if not high_risk_full.empty else [])
                outreach_date = st.date_input("วันที่นัดติดต่อ")
                note = st.text_area("บันทึกเพิ่มเติม", placeholder="เช่น โทรนัด Fast-track ตรวจหัวใจ")
                submitted = st.form_submit_button("ยืนยันกำหนดการ")
                if submitted:
                    st.session_state['outreach_queue'].append({
                        'วันที่นัด': str(outreach_date), 'จำนวนราย': len(pick_ids), 'บันทึก': note
                    })
                    st.session_state['show_outreach_form'] = False
                    st.success(f"✅ กำหนดการติดต่อ {len(pick_ids)} รายเรียบร้อย")

        if st.session_state['outreach_queue']:
            st.markdown('<div class="panel-title" style="margin-top:8px; font-size:0.8rem;">คิวที่กำหนดไว้ (session นี้)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state['outreach_queue']), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.container(key="card_bp_pie"):
        st.markdown('<div class="panel-title">🩺 สัดส่วนระดับความดันโลหิต</div>', unsafe_allow_html=True)
        bp_data = df_view['bp_level'].value_counts().reset_index()
        bp_data.columns = ['Level', 'Count']
        fig_pie = px.pie(bp_data, names='Level', values='Count', hole=0.55, color='Level', color_discrete_map=BP_COLORS)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=230, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=-0.3, font=dict(size=9)))
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ============================================================
# 9. รายชื่อกลุ่มเสี่ยงสูง — AgGrid conditional formatting (fallback: pandas Styler)
# ============================================================
with st.container(key="card_risktable"):
    st.markdown('<div class="panel-title">🚨 รายชื่อผู้ป่วยกลุ่มเสี่ยงสูงต้องเฝ้าระวัง</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">ไล่สีแดงตามความรุนแรงของ Systolic — ยิ่งเข้มยิ่งเร่งด่วน</div>', unsafe_allow_html=True)

    if not high_risk_full.empty:
        disp = high_risk_full.copy()
        disp.columns = ['ID', 'กลุ่มโรค', 'BMI', 'Sys', 'Dia']

        if AGGRID_AVAILABLE:
            row_style_jscode = JsCode("""
            function(params) {
                if (params.data.Sys >= 180) { return {'backgroundColor': '#F3A9A5', 'color': '#5A0E0E'}; }
                if (params.data.Sys >= 160) { return {'backgroundColor': '#F8C6C0'}; }
                if (params.data.Sys >= 140) { return {'backgroundColor': '#FBE1DE'}; }
                return {};
            }
            """)
            gb = GridOptionsBuilder.from_dataframe(disp)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=8)
            gb.configure_default_column(sortable=True, filter=True, resizable=True)
            gb.configure_grid_options(getRowStyle=row_style_jscode)
            grid_options = gb.build()
            AgGrid(disp, gridOptions=grid_options, allow_unsafe_jscode=True,
                   fit_columns_on_grid_load=True, height=260, theme='alpine')
        else:
            st.caption("ℹ️ ติดตั้ง `streamlit-aggrid` (เพิ่มใน requirements.txt) เพื่อได้ตารางแบบ sort/filter ได้เต็มรูปแบบ — ตอนนี้แสดงแบบไล่สีพื้นฐานแทน")
            def highlight_severity(row):
                if row['Sys'] >= 180:
                    return ['background-color: #F3A9A5'] * len(row)
                elif row['Sys'] >= 160:
                    return ['background-color: #F8C6C0'] * len(row)
                elif row['Sys'] >= 140:
                    return ['background-color: #FBE1DE'] * len(row)
                return [''] * len(row)
            st.dataframe(disp.style.apply(highlight_severity, axis=1), use_container_width=True, hide_index=True, height=260)

        csv_data = high_risk_full.to_csv(index=False).encode('utf-8-sig')
        st.download_button(f"📥 Export ทั้งหมด ({len(high_risk_full)} รายการ)", csv_data,
                            "critical_risk_patients.csv", "text/csv", use_container_width=True)
    else:
        st.success("✅ ไม่พบคนไข้ในเกณฑ์ความเสี่ยงวิกฤต")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ============================================================
# 10. แถวที่ 3 (3-Column): พีระมิดประชากร | คลินิกยอดนิยม | BMI vs ความดัน
# ============================================================
p1, p2, p3 = st.columns([1, 1.1, 1.2])

with p1:
    with st.container(key="card_pyramid"):
        st.markdown('<div class="panel-title">👥 พีระมิดประชากรผู้ป่วย</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">อายุ x เพศ</div>', unsafe_allow_html=True)
        order = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        pyr = df_view.groupby(['pyramid_group', 'gender']).size().reset_index(name='count')
        male = pyr[pyr['gender'] == 'ช'].set_index('pyramid_group')['count']
        female = pyr[pyr['gender'] == 'ญ'].set_index('pyramid_group')['count']
        male_vals = [int(male.get(l, 0)) for l in order]
        female_vals = [int(female.get(l, 0)) for l in order]

        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(y=order, x=[-v for v in male_vals], name='ชาย', orientation='h',
                                  marker_color=TEAL, customdata=male_vals, hovertemplate='ชาย %{y}: %{customdata} คน<extra></extra>'))
        fig_pyr.add_trace(go.Bar(y=order, x=female_vals, name='หญิง', orientation='h',
                                  marker_color="#D97AA0", customdata=female_vals, hovertemplate='หญิง %{y}: %{customdata} คน<extra></extra>'))
        max_v = max(male_vals + female_vals) if (male_vals + female_vals) else 1
        fig_pyr.update_layout(
            barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=310, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
            xaxis=dict(tickvals=[-max_v, -max_v/2, 0, max_v/2, max_v],
                       ticktext=[str(int(max_v)), str(int(max_v/2)), "0", str(int(max_v/2)), str(int(max_v))],
                       title=None, gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_pyr, use_container_width=True)

with p2:
    with st.container(key="card_clinic"):
        st.markdown('<div class="panel-title">🏢 คลินิกที่มีผู้ป่วยมากที่สุด</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Top 10 จาก 69 คลินิก</div>', unsafe_allow_html=True)
        clinic_counts = df_view['clinic_name'].value_counts().head(10).reset_index()
        clinic_counts.columns = ['คลินิก', 'จำนวนเคส']
        clinic_counts['label'] = clinic_counts['คลินิก'].apply(lambda x: x if len(x) <= 26 else x[:24] + '…')
        fig_clinic = px.bar(clinic_counts.sort_values('จำนวนเคส'), x='จำนวนเคส', y='label', orientation='h',
                             color='จำนวนเคส', color_continuous_scale=[[0, '#CFE3DF'], [1, TEAL]])
        fig_clinic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  height=310, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        fig_clinic.update_yaxes(title=None)
        st.plotly_chart(fig_clinic, use_container_width=True)

with p3:
    with st.container(key="card_scatter"):
        st.markdown('<div class="panel-title">⚖️ BMI กับความดันโลหิต (รายเคส)</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">เฉพาะผู้ใหญ่ที่มีค่าความดันจริง — เส้นประ = เกณฑ์เสี่ยง</div>', unsafe_allow_html=True)
        scatter_df = df_view[df_view['is_adult'] & df_view['systolic'].notna() & ~df_view['bmi_imputed']].copy()
        if 'patient_id' in scatter_df.columns:
            scatter_df['visit_freq'] = scatter_df.groupby('patient_id')['patient_id'].transform('count')
        else:
            scatter_df['visit_freq'] = 1
        if not scatter_df.empty:
            fig_scatter = px.scatter(scatter_df, x='bmi', y='systolic', color='disease_group', size='visit_freq',
                                      color_discrete_map=DISEASE_COLORS)
            fig_scatter.add_hline(y=140, line_dash="dot", line_color=RED, opacity=0.5)
            fig_scatter.add_vline(x=25, line_dash="dot", line_color=RED, opacity=0.5)
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       height=310, margin=dict(l=0, r=0, t=10, b=0),
                                       legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=8)))
            fig_scatter.update_xaxes(title="BMI", gridcolor='#F1F5F9')
            fig_scatter.update_yaxes(title="Systolic (mmHg)", gridcolor='#F1F5F9')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.caption("ไม่มีข้อมูลเพียงพอตามตัวกรองที่เลือก")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ============================================================
# 11. แถวที่ 4: แนวโน้มรายวัน + แนวโน้มกลุ่มเสี่ยง | โอกาสแพ็กเกจ
# ============================================================
t1, t2 = st.columns([1.5, 1])

with t1:
    st.markdown('<div class="float-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📈 ปริมาณเคสรายวัน</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-sub">⚠️ {single_day_share:.0f}% กระจุกวันเดียว — น่าจะเป็นวันนำเข้าข้อมูลชุดใหญ่ ไม่ใช่แนวโน้มผู้ป่วยจริง</div>', unsafe_allow_html=True)
    valid_trend_data = df_view.dropna(subset=['visit_date']).copy() if has_date else pd.DataFrame()
    if has_date and not valid_trend_data.empty:
        weekly = valid_trend_data.groupby(['visit_day', 'disease_group']).size().reset_index(name='count')
        weekly = weekly.sort_values('visit_day')
        weekly['week_label'] = weekly['visit_day'].dt.strftime('%d %b')
        day_order = weekly.drop_duplicates('visit_day').sort_values('visit_day')['week_label'].tolist()
        fig_trend = px.bar(weekly, x='week_label', y='count', color='disease_group',
                            category_orders={'week_label': day_order}, color_discrete_map=DISEASE_COLORS, barmode='stack')
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=200,
                                 margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        fig_trend.update_xaxes(title=None, showgrid=False)
        fig_trend.update_yaxes(title=None, gridcolor='#F1F5F9')
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown('<div class="panel-title" style="margin-top:4px;">📉 แนวโน้มกลุ่มเสี่ยงวิกฤต</div>', unsafe_allow_html=True)
        if 'patient_id' in df_view.columns:
            daily = valid_trend_data.groupby('visit_day').apply(
                lambda g: pd.Series({'total': g['patient_id'].nunique(), 'risk': g[g['critical_risk'] == 1]['patient_id'].nunique()})
            ).reset_index()
            daily = daily.sort_values('visit_day')
            low_sample_days = int((daily['total'] < MIN_SAMPLE_FOR_TREND).sum())
            daily['risk_pct'] = np.where(daily['total'] >= MIN_SAMPLE_FOR_TREND,
                                          np.where(daily['total'] > 0, daily['risk'] / daily['total'] * 100, 0), np.nan)
            daily['label'] = daily['visit_day'].dt.strftime('%d %b')
            fig_rt = px.line(daily, x='label', y='risk_pct', markers=True)
            fig_rt.update_traces(line_color=RED, connectgaps=False)
            fig_rt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(l=0, r=0, t=6, b=0))
            fig_rt.update_xaxes(title=None, showgrid=False, categoryorder='array', categoryarray=daily['label'].tolist())
            fig_rt.update_yaxes(title="%", gridcolor='#F1F5F9')
            st.plotly_chart(fig_rt, use_container_width=True)
            if low_sample_days > 0:
                st.caption(f"⚠️ เว้น {low_sample_days} วันที่มีคนไข้น้อยกว่า {MIN_SAMPLE_FOR_TREND} คน")
    else:
        st.caption("ไม่มีข้อมูลวันที่")
    st.markdown('</div>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="float-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎯 โอกาสนำเสนอแพ็กเกจตรวจสุขภาพ</div>', unsafe_allow_html=True)
    if clf is not None and 'patient_id' in df_view.columns:
        avail_pts = [p for p in summary_pts.index if p in df_view['patient_id'].values]
        if avail_pts:
            sel_pid = st.selectbox("เลือก Patient ID:", avail_pts[:40], label_visibility="collapsed")
            score = summary_pts.loc[sel_pid, 'lead_score']
            card_bg = "#ECFDF5" if score >= 60 else BG
            card_border = "#A7F3D0" if score >= 60 else "#E2E8F0"
            text_color = "#065F46" if score >= 60 else MUTED
            badge_text = "🔥 High Priority: เสนอโปรแกรมตรวจคัดกรอง" if score >= 60 else "🌱 General: ติดตามผลรอบปกติ"
            st.markdown(f"""
            <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:10px; padding:10px; margin-top:2px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.75rem; font-weight:600; color:{text_color};">Conversion score</span>
                    <span style="font-size:1.3rem; font-weight:700; color:{text_color}; font-family:'IBM Plex Mono',monospace;">{score}%</span>
                </div>
                <div style="font-size:0.75rem; font-weight:500; color:{text_color}; margin-top:2px;">{badge_text}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.73rem; color:{MUTED}; margin-top:6px;">
                💼 กลุ่มเป้าหมายคะแนนสูงรวม: <b>{high_lead_count:,} ราย</b><br>
                ประมาณการมูลค่า: <b>~{est_pipeline_value:,.0f} บาท</b>
                <span style="font-size:0.68rem;">({high_lead_count:,} × 3,000 บาท/แพ็กเกจ)</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("⚠️ คะแนนอิงกฎ (visits≥3 หรือ systolic≥135) ไม่ใช่ประวัติซื้อจริง ใช้จัดลำดับความสำคัญ ไม่ใช่พยากรณ์ทางสถิติ")
    else:
        st.caption("ไม่มีข้อมูลโมเดล")
    st.markdown('</div>', unsafe_allow_html=True)
