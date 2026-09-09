import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. Figma-Grade Compact Layout CSS (Zero-Scroll Strategy)
# ============================================================
st.set_page_config(
    page_title="Executive Clinical Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0F172A;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0rem !important;
        max-width: 98% !important;
    }
    
    /* Clean Transitions */
    * { transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); }

    /* Compact Bento Card */
    .bento-box {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        height: 100%;
    }
    .bento-box:hover {
        border-color: #CBD5E1;
        box-shadow: 0 8px 16px -4px rgba(15, 23, 42, 0.06);
    }
    
    /* KPI Typography */
    .kpi-title { font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.65rem; font-weight: 700; color: #0F172A; line-height: 1.2; margin: 4px 0; }
    .kpi-sub { font-size: 0.72rem; color: #94A3B8; }
    
    /* Risk Badges */
    .badge { padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.65rem; }
    .badge-red { background: #FEE2E2; color: #DC2626; }
    .badge-green { background: #DCFCE7; color: #16A34A; }

    /* Header text */
    .exec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. Optimized Pipeline
# ============================================================
@st.cache_data
def load_clean_data():
    path = "visits_cleaned.csv"
    if not os.path.exists(path):
        return None, False
        
    df = pd.read_csv(path)

    # Date
    has_date = 'visit_date' in df.columns and df['visit_date'].notna().any()
    if has_date:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df['period'] = df['visit_date'].dt.to_period('M').astype(str)
    else:
        df['period'] = 'ไม่ระบุ'

    # BP split
    if 'bp_raw' in df.columns:
        df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
        bp = df['bp_raw'].str.split(' / ', expand=True)
        df['systolic'] = pd.to_numeric(bp[0], errors='coerce').fillna(0)
    else:
        df['systolic'] = 0

    # BMI & Gender
    df['gender'] = df['gender'].fillna('ไม่ระบุ') if 'gender' in df.columns else 'ไม่ระบุ'
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'] = df['bmi'].fillna(df['bmi'].median() if df['bmi'].notna().any() else 22.0)
    else:
        df['bmi'] = 22.0

    # Categories
    df['bp_level'] = pd.cut(df['systolic'], bins=[-1, 120, 139, 300], labels=['ปกติ (<120)', 'เฝ้าระวัง (120-139)', 'สูง (≥140)'])
    df['critical_risk'] = ((df['bmi'] >= 25) & (df['systolic'] >= 140)).astype(int)
    
    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ทั่วไป'

    return df, has_date

df, has_date = load_clean_data()
if df is None:
    st.error("⚠️ ไม่พบไฟล์ `visits_cleaned.csv`")
    st.stop()

# Scoring Pipeline
@st.cache_resource
def get_scoring_engine(data):
    feats = ['visits', 'bmi', 'systolic', 'gender_code']
    X = data[feats]
    y = data['target']
    return RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X, y)

if 'patient_id' in df.columns:
    summary_pts = df.groupby('patient_id').agg(
        visits=('visit_id', 'count') if 'visit_id' in df.columns else ('patient_id', 'count'),
        bmi=('bmi', 'mean'),
        systolic=('systolic', 'mean'),
        gender_code=('gender_code', 'first')
    ).round(1)
    summary_pts['target'] = ((summary_pts['visits'] >= 3) | (summary_pts['systolic'] >= 135)).astype(int)
    model = get_scoring_engine(summary_pts)
else:
    model, summary_pts = None, None

# ============================================================
# 3. Fast Compact Sidebar Filters
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filter Scope")
    disease_sel = st.multiselect("กลุ่มโรค", sorted(df['disease_group'].unique()), default=sorted(df['disease_group'].unique()))
    gender_sel = st.multiselect("เพศ", sorted(df['gender'].unique()), default=sorted(df['gender'].unique()))

df_view = df[df['disease_group'].isin(disease_sel) & df['gender'].isin(gender_sel)]
if df_view.empty:
    st.warning("ไม่มีข้อมูลตามตัวกรอง")
    st.stop()

# ============================================================
# 4. ONE-PAGE COMMAND DASHBOARD
# ============================================================

# Top Nav Header
st.markdown("""
<div class="exec-header">
    <div>
        <span style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">🏥 Executive Health Command Center</span>
        <span style="font-size: 0.8rem; color: #64748B; margin-left: 10px;">ภาพรวมสุขภาพ ความเสี่ยง และโอกาสแปลงเป็นแพ็กเกจ</span>
    </div>
    <div><span class="badge badge-green">LIVE UPDATED</span></div>
</div>
""", unsafe_allow_html=True)

# --- ROW 1: 4 North Star Metrics ---
total_v = len(df_view)
unique_pts = df_view['patient_id'].nunique() if 'patient_id' in df_view.columns else total_v
risk_pts = df_view[df_view['critical_risk'] == 1]['patient_id'].nunique() if 'patient_id' in df_view.columns else 0
risk_rate = (risk_pts / unique_pts * 100) if unique_pts else 0
avg_bp = df_view[df_view['systolic'] > 0]['systolic'].mean()

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="bento-box"><div class="kpi-title">Total Visits</div><div class="kpi-value">{total_v:,}</div><div class="kpi-sub">เคสรับบริการทั้งหมด</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="bento-box"><div class="kpi-title">Active Patients</div><div class="kpi-value">{unique_pts:,}</div><div class="kpi-sub">จำนวนผู้ป่วยรายบุคคล</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="bento-box"><div class="kpi-title">Average Systolic</div><div class="kpi-value">{avg_bp:.1f} <span style="font-size:0.9rem;font-weight:400;">mmHg</span></div><div class="kpi-sub">ค่าความดันตัวบนเฉลี่ย</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="bento-box"><div class="kpi-title">Critical Risk Pool</div><div class="kpi-value" style="color:#DC2626;">{risk_rate:.1f}%</div><div class="kpi-sub">{risk_pts:,} คน (BMI≥25 & BP≥140)</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# --- ROW 2: Core Analytics (3-Column Bento Grid) ---
c_left, c_mid, c_right = st.columns([1.8, 1.2, 1.4])

with c_left:
    st.markdown('<div class="bento-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.8rem; font-weight:700;'>📈 ปริมาณเคสแยกตามกลุ่มโรค</span>", unsafe_allow_html=True)
    if has_date:
        trend = df_view.groupby(['period', 'disease_group']).size().reset_index(name='count')
        fig1 = px.area(trend, x='period', y='count', color='disease_group', color_discrete_sequence=['#3B82F6','#93C5FD','#CBD5E1'])
    else:
        trend = df_view['disease_group'].value_counts().reset_index()
        fig1 = px.bar(trend, x='count', y='disease_group', orientation='h', color_discrete_sequence=['#3B82F6'])
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    fig1.update_yaxes(gridcolor='#F1F5F9')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_mid:
    st.markdown('<div class="bento-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.8rem; font-weight:700;'>🩺 สัดส่วนความดันโลหิต</span>", unsafe_allow_html=True)
    bp_data = df_view['bp_level'].value_counts().reset_index()
    fig2 = px.pie(
        bp_data, names='bp_level', values='count', hole=0.6,
        color='bp_level',
        color_discrete_map={'ปกติ (<120)': '#10B981', 'เฝ้าระวัง (120-139)': '#F59E0B', 'สูง (≥140)': '#EF4444'}
    )
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=10)))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="bento-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.8rem; font-weight:700;'>🚨 รายชื่อเสี่ยงสูงต้องโทรติดตาม</span>", unsafe_allow_html=True)
    if risk_pts > 0 and 'patient_id' in df_view.columns:
        high_risk = df_view[df_view['critical_risk'] == 1][['patient_id', 'bmi', 'systolic']].drop_duplicates('patient_id').head(6)
        st.dataframe(high_risk, use_container_width=True, hide_index=True, height=190)
    else:
        st.caption("ไม่พบเคสความเสี่ยงวิกฤต")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# --- ROW 3: Predictive Opportunity & Conversion Engine ---
r3_left, r3_right = st.columns([1.5, 2.5])

with r3_left:
    st.markdown('<div class="bento-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.8rem; font-weight:700;'>🎯 ผู้ป่วยกลุ่มเป้าหมาย (แพ็กเกจตรวจ)</span>", unsafe_allow_html=True)
    if model is not None:
        p_list = summary_pts.index.tolist()
        sel_pid = st.selectbox("เลือก Patient ID เพื่อประเมิน:", p_list[:50], label_visibility="collapsed")
        pt = summary_pts.loc[[sel_pid]]
        score = int(model.predict_proba(pt[['visits', 'bmi', 'systolic', 'gender_code']])[0][1] * 100)
        
        status_color = "#10B981" if score >= 60 else "#64748B"
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:10px;">
            <div>
                <div style="font-size:1.8rem; font-weight:700; color:{status_color};">{score}%</div>
                <div style="font-size:0.75rem; color:#64748B;">ความพร้อมซื้อแพ็กเกจ</div>
            </div>
            <div style="font-size:0.8rem; max-width:180px; text-align:right;">
                {"🔥 <b>High Priority</b><br>ส่งต่อทีมโทรเสนอนัด" if score>=60 else "🌱 <b>Standard</b><br>ติดตามผลรอบปกติ"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r3_right:
    st.markdown('<div class="bento-box">', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.8rem; font-weight:700;'>⚡ สรุปทิศทางเพื่อการตัดสินใจของผู้บริหาร (Executive Actions)</span>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:0.82rem; margin-top:6px;">
        <div style="background:#F1F5F9; padding:10px 12px; border-radius:8px; border-left:3px solid #0F172A;">
            <b>1. มาตรการลดความเสี่ยงคลินิก</b><br>
            ผู้ป่วย {risk_pts:,} ราย อยู่ในเกณฑ์ Metabolic Syndrome ควรเปิดช่องทาง Fast-track เข้าพบแพทย์อายุรกรรมเฉพาะทาง
        </div>
        <div style="background:#EFF6FF; padding:10px 12px; border-radius:8px; border-left:3px solid #3B82F6;">
            <b>2. โอกาสสร้างรายได้เชิงรุก</b><br>
            คนไข้ที่มีความดันช่วงเฝ้าระวัง (120-139 mmHg) เหมาะสมกับการเสนอ **Preventive Cardiovascular Package** เพื่อป้องกันการเกิดโรคเรื้อรัง
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
