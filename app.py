import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. ตั้งค่าหน้าเว็บและดีไซน์ UI (Zero Empty Boxes)
# ============================================================
st.set_page_config(
    page_title="Executive Clinical Intelligence",
    page_icon="🏥",
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
        padding-bottom: 1.5rem !important;
        max-width: 98% !important;
    }
    
    /* Clean Metric Card */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #0F172A;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Executive Callout Card */
    .exec-action-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        height: 100%;
    }
    .header-bar {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. ทำความสะอาดข้อมูล (แก้ปัญหาปี 1970 และค่าสูญหาย)
# ============================================================
@st.cache_data
def load_clean_dataset():
    path = "visits_cleaned.csv"
    if not os.path.exists(path):
        return None, False, False
        
    df = pd.read_csv(path)

    # 1. วันที่: ตัดปี 1970 หรือค่าที่เพี้ยนออก
    has_date = False
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        # กรองเฉพาะวันที่สมเหตุสมผล (หลังปี 2000 เป็นต้นมา)
        valid_date_mask = df['visit_date'].notna() & (df['visit_date'].dt.year >= 2000)
        has_date = valid_date_mask.any()
        if has_date:
            df.loc[~valid_date_mask, 'visit_date'] = pd.NaT
            df['period'] = df['visit_date'].dt.strftime('%b %Y')
            df['year_month'] = df['visit_date'].dt.to_period('M').astype(str)
        else:
            df['period'] = 'ไม่ระบุ'
            df['year_month'] = 'ไม่ระบุ'
    else:
        df['period'] = 'ไม่ระบุ'
        df['year_month'] = 'ไม่ระบุ'

    # 2. ความดันโลหิต
    if 'bp_raw' in df.columns:
        df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
        bp_split = df['bp_raw'].str.split(' / ', expand=True)
        df['systolic'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(0)
        df['diastolic'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(0) if bp_split.shape[1] > 1 else 0
    else:
        df['systolic'] = 0
        df['diastolic'] = 0

    # 3. เพศ และ BMI
    df['gender'] = df['gender'].fillna('ไม่ระบุ') if 'gender' in df.columns else 'ไม่ระบุ'
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        valid_bmi = df['bmi'].dropna()
        df['bmi'] = df['bmi'].fillna(valid_bmi.median() if not valid_bmi.empty else 22.0)
    else:
        df['bmi'] = 22.0

    # 4. อายุ
    has_age = 'age_at_visit' in df.columns and df['age_at_visit'].notna().any()
    if has_age:
        df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce')
        valid_age = df['age_at_visit'].dropna()
        df['age_at_visit'] = df['age_at_visit'].fillna(valid_age.median() if not valid_age.empty else 35.0)
        df['age_group'] = pd.cut(
            df['age_at_visit'],
            bins=[0, 25, 45, 60, 120],
            labels=['<25 ปี', '25-45 ปี', '46-60 ปี', '>60 ปี']
        ).astype(str).replace('nan', 'ไม่ระบุ')
    else:
        df['age_group'] = 'ไม่ระบุ'
        df['age_at_visit'] = 0

    # 5. การจัดกลุ่มมาตรฐานทางคลินิก
    df['bp_level'] = pd.cut(
        df['systolic'],
        bins=[-1, 120, 139, 300],
        labels=['ปกติ (<120)', 'เฝ้าระวัง (120-139)', 'สูง (≥140)']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # กลุ่มเสี่ยงวิกฤต (BMI เกิน + ความดันสูง)
    df['critical_risk'] = ((df['bmi'] >= 25) & (df['systolic'] >= 140)).astype(int)

    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ทั่วไป'

    return df, has_date, has_age

df, has_date, has_age = load_clean_dataset()

if df is None:
    st.error("⚠️ ไม่พบไฟล์ข้อมูล `visits_cleaned.csv`")
    st.stop()

# ============================================================
# 3. โมเดลประเมินความสนใจแพ็กเกจ (Predictive Lead Engine)
# ============================================================
@st.cache_resource
def get_scoring_model(data_df):
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
    summary_pts['target'] = ((summary_pts['visits'] >= 3) | (summary_pts['systolic'] >= 135)).astype(int)
    clf = get_scoring_model(summary_pts)
else:
    clf, summary_pts = None, None

# ============================================================
# 4. แถบตัวกรอง (Sidebar Controls)
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

# กรองข้อมูล
mask = df['disease_group'].isin(disease_sel) & df['gender'].isin(gender_sel)
if date_filter and isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    start_d, end_d = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    mask &= df['visit_date'].between(start_d, end_d) | df['visit_date'].isna()

df_view = df[mask]

if df_view.empty:
    st.warning("⚠️ ไม่มีข้อมูลตรงกับตัวกรองที่เลือก")
    st.stop()

# ============================================================
# 5. DASHBOARD หน้าเดียวจบ (One-Page Executive Grid)
# ============================================================

# Header
st.markdown("""
<div class="header-bar">
    <div>
        <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">🏥 Executive Clinical & Growth Command Center</span>
        <span style="font-size: 0.85rem; color: #64748B; margin-left: 12px;">มอนิเตอร์สุขภาพประชากร จัดการกลุ่มเสี่ยง และขยายผลสู่แพ็กเกจการรักษา</span>
    </div>
    <span style="background:#E2E8F0; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; color:#475569;">EXECUTIVE VIEW</span>
</div>
""", unsafe_allow_html=True)

# --- แถวที่ 1: 4 North Star Metrics ---
total_v = len(df_view)
unique_pts = df_view['patient_id'].nunique() if 'patient_id' in df_view.columns else total_v
risk_pts = df_view[df_view['critical_risk'] == 1]['patient_id'].nunique() if 'patient_id' in df_view.columns else 0
risk_pct = (risk_pts / unique_pts * 100) if unique_pts else 0
avg_bp = df_view[df_view['systolic'] > 0]['systolic'].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("TOTAL VISITS", f"{total_v:,}", "เคสรับบริการทั้งหมด")
k2.metric("ACTIVE PATIENTS", f"{unique_pts:,}", "คนไข้รายบุคคล")
k3.metric("AVERAGE SYSTOLIC", f"{avg_bp:.1f} mmHg", "ความดันตัวบนเฉลี่ย")
k4.metric("CRITICAL RISK POOL", f"{risk_pct:.1f}%", f"{risk_pts:,} คน (BMI≥25 & BP≥140)", delta_color="inverse")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# --- แถวที่ 2: การวิเคราะห์เชิงคลินิก (3 คอลัมน์ จัดสัดส่วนชัดเจน) ---
c1, c2, c3 = st.columns([1.6, 1.1, 1.3])

with c1:
    st.markdown("**📈 ปริมาณเคสแยกตามกลุ่มโรค**")
    # หากมีข้อมูลช่วงเดือนที่ถูกต้อง ให้แสดง Time-Series แต่ถ้าไม่มี ให้แสดง Bar Chart แนวโน้มโรคทันที
    valid_trend_data = df_view.dropna(subset=['visit_date']) if has_date else pd.DataFrame()
    
    if has_date and not valid_trend_data.empty:
        trend = valid_trend_data.groupby(['year_month', 'disease_group']).size().reset_index(name='count')
        trend = trend.sort_values('year_month')
        fig_trend = px.area(
            trend, x='year_month', y='count', color='disease_group',
            color_discrete_sequence=['#2563EB', '#60A5FA', '#93C5FD', '#CBD5E1']
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
        )
        fig_trend.update_xaxes(title=None, showgrid=False)
        fig_trend.update_yaxes(title="จำนวนเคส", gridcolor='#F1F5F9')
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        # Fallback: Bar Chart จัดอันดับกลุ่มโรค
        disease_counts = df_view['disease_group'].value_counts().reset_index()
        disease_counts.columns = ['กลุ่มโรค', 'จำนวนเคส']
        fig_dis = px.bar(
            disease_counts, x='จำนวนเคส', y='กลุ่มโรค', orientation='h',
            color='จำนวนเคส', color_continuous_scale='Blues', text_auto=True
        )
        fig_dis.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=240, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
        )
        st.plotly_chart(fig_dis, use_container_width=True)

with c2:
    st.markdown("**🩺 สัดส่วนระดับความดันโลหิต**")
    bp_data = df_view['bp_level'].value_counts().reset_index()
    bp_data.columns = ['Level', 'Count']
    fig_pie = px.pie(
        bp_data, names='Level', values='Count', hole=0.55,
        color='Level',
        color_discrete_map={
            'ปกติ (<120)': '#10B981',
            'เฝ้าระวัง (120-139)': '#F59E0B',
            'สูง (≥140)': '#EF4444'
        }
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=9))
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c3:
    st.markdown("**🚨 รายชื่อผู้ป่วยกลุ่มเสี่ยงสูงต้องเฝ้าระวัง**")
    if risk_pts > 0 and 'patient_id' in df_view.columns:
        high_risk_df = df_view[df_view['critical_risk'] == 1][[
            'patient_id', 'disease_group', 'bmi', 'systolic', 'diastolic'
        ]].drop_duplicates('patient_id').head(6)
        
        high_risk_df.columns = ['ID', 'กลุ่มโรค', 'BMI', 'Sys', 'Dia']
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True, height=230)
    else:
        st.success("✅ ไม่พบคนไข้ในเกณฑ์ความเสี่ยงวิกฤต")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# --- แถวที่ 3: Demographic + โอกาสเสนอแพ็กเกจ + Action Plan ---
r3_c1, r3_c2, r3_c3 = st.columns([1.1, 1.2, 1.7])

with r3_c1:
    st.markdown("**👥 สัดส่วนช่วงอายุผู้รับบริการ**")
    if has_age:
        age_counts = df_view['age_group'].value_counts().reset_index()
        age_counts.columns = ['Age', 'Count']
        fig_age = px.bar(
            age_counts, x='Age', y='Count', color_discrete_sequence=['#475569'], text_auto=True
        )
        fig_age.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=190, margin=dict(l=0, r=0, t=10, b=0)
        )
        fig_age.update_xaxes(title=None, showgrid=False)
        fig_age.update_yaxes(showgrid=False, visible=False)
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.caption("ไม่มีข้อมูลอายุ")

with r3_c2:
    st.markdown("**🎯 ประเมินความพร้อมแพ็กเกจตรวจสุขภาพ**")
    if clf is not None and 'patient_id' in df_view.columns:
        avail_pts = [p for p in summary_pts.index if p in df_view['patient_id'].values]
        if avail_pts:
            selected_pt = st.selectbox("เลือก Patient ID:", avail_pts[:40], label_visibility="collapsed")
            pt_data = summary_pts.loc[[selected_pt]]
            prob = clf.predict_proba(pt_data[['visits', 'bmi', 'systolic', 'gender_code']])[0][1]
            score = int(prob * 100)
            
            card_bg = "#ECFDF5" if score >= 60 else "#F8FAFC"
            card_border = "#A7F3D0" if score >= 60 else "#E2E8F0"
            text_color = "#065F46" if score >= 60 else "#475569"
            badge_text = "🔥 High Priority: ติดต่อเสนอโปรแกรมตรวจสุขภาพ" if score >= 60 else "🌱 General: ส่งบทความดูแลตนเอง"

            st.markdown(f"""
            <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:10px; padding:12px; margin-top:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.75rem; font-weight:600; color:{text_color};">CONVERSION SCORE</span>
                    <span style="font-size:1.4rem; font-weight:700; color:{text_color};">{score}%</span>
                </div>
                <div style="font-size:0.78rem; font-weight:500; color:{text_color}; margin-top:4px;">{badge_text}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("ไม่มีข้อมูลโมเดล")

with r3_c3:
    st.markdown("**⚡ สรุปทิศทางเพื่อการตัดสินใจ (Executive Actions)**")
    st.markdown(f"""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:6px;">
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:10px; border-left:4px solid #0F172A;">
            <b style="font-size:0.8rem; color:#0F172A;">1. มาตรการเชิงคลินิก</b><br>
            <span style="font-size:0.75rem; color:#64748B;">
            พบผู้ป่วยกลุ่มเสี่ยง {risk_pts:,} ราย (ความดัน ≥140 ร่วมกับ BMI เกิน) ควรส่งเข้า Fast-track ตรวจหลอดเลือดหัวใจทันที
            </span>
        </div>
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:10px; border-left:4px solid #2563EB;">
            <b style="font-size:0.8rem; color:#2563EB;">2. โอกาสสร้างรายได้</b><br>
            <span style="font-size:0.75rem; color:#64748B;">
            กลุ่มความดันเฝ้าระวัง (120-139 mmHg) คือเป้าหมายหลักในการนำเสนอ <b>Preventive Wellness Package</b> ประจำปี
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
