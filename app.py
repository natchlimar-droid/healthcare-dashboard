import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. Config & Figma-Inspired CSS Design System
# ============================================================
st.set_page_config(
    page_title="Executive Health Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Typography, Smooth Interactions, Clean Surface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1E293B;
    }
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EDF2F7 100%);
    }
    
    /* Smooth Transitions like Figma */
    * {
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Executive Hero Header */
    .hero-container {
        padding: 24px 0 12px 0;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 400;
    }

    /* Bento Cards */
    .bento-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        margin-bottom: 18px;
    }
    .bento-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 24px -8px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }

    /* Metric Cards */
    .metric-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .badge-blue { background: #EFF6FF; color: #2563EB; }
    .badge-amber { background: #FEF3C7; color: #D97706; }
    .badge-rose { background: #FFE4E6; color: #E11D48; }
    .badge-emerald { background: #D1FAE5; color: #059669; }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 6px;
    }

    /* Action Recommendation Chip */
    .action-box {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #2563EB;
        font-size: 0.9rem;
    }

    /* Clean Streamlit Components */
    div[data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
    }
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: none !important;
        background: transparent !important;
    }
    button[aria-selected="true"] {
        background: #0F172A !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. Data Pipeline
# ============================================================
@st.cache_data
def fetch_and_clean():
    filepath = "visits_cleaned.csv"
    if not os.path.exists(filepath):
        return None, False, False
        
    df = pd.read_csv(filepath)

    # Date parsing
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        has_date = df['visit_date'].notna().any()
    else:
        has_date = False

    if has_date:
        df['period'] = df['visit_date'].dt.to_period('M').astype(str)
        df['month_name'] = df['visit_date'].dt.strftime('%b %Y')
    else:
        df['period'] = 'N/A'
        df['month_name'] = 'N/A'

    # Blood pressure parsing
    if 'bp_raw' in df.columns:
        df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
        bp = df['bp_raw'].str.split(' / ', expand=True)
        df['systolic'] = pd.to_numeric(bp[0], errors='coerce').fillna(0)
        df['diastolic'] = pd.to_numeric(bp[1], errors='coerce').fillna(0) if bp.shape[1] > 1 else 0
    else:
        df['systolic'] = 0
        df['diastolic'] = 0

    # Clean demographics
    df['gender'] = df['gender'].fillna('ไม่ระบุ') if 'gender' in df.columns else 'ไม่ระบุ'
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        valid_bmi = df['bmi'].dropna()
        df['bmi'] = df['bmi'].fillna(valid_bmi.median() if not valid_bmi.empty else 22.0)
    else:
        df['bmi'] = 22.0

    # Clean age
    has_age = 'age_at_visit' in df.columns and df['age_at_visit'].notna().any()
    if has_age:
        df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce')
        valid_age = df['age_at_visit'].dropna()
        df['age_at_visit'] = df['age_at_visit'].fillna(valid_age.median() if not valid_age.empty else 35.0)
        df['age_segment'] = pd.cut(
            df['age_at_visit'],
            bins=[0, 25, 45, 60, 120],
            labels=['Young (<25)', 'Working (25-45)', 'Senior-Mid (46-60)', 'Elderly (60+)']
        ).astype(str).replace('nan', 'ไม่ระบุ')
    else:
        df['age_segment'] = 'ไม่ระบุ'
        df['age_at_visit'] = 0

    # Clinical status definitions
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 23, 25, 30, 150],
        labels=['น้ำหนักน้อย', 'ปกติ', 'เริ่มอ้วน (Overweight)', 'อ้วนระดับ 1', 'อ้วนระดับ 2']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    df['bp_category'] = pd.cut(
        df['systolic'],
        bins=[-1, 120, 139, 300],
        labels=['ปกติ (<120)', 'เฝ้าระวัง (120-139)', 'วิกฤต/สูง (≥140)']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # Executive flags
    df['critical_risk'] = ((df['bmi'] >= 25) & (df['systolic'] >= 140)).astype(int)
    
    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ทั่วไป'

    return df, has_date, has_age

df, has_date, has_age = fetch_and_clean()

if df is None:
    st.error("⚠️ ไม่พบคลังข้อมูล `visits_cleaned.csv` กรุณาตรวจสอบไฟล์")
    st.stop()

# ============================================================
# 3. Lightweight Scoring Model
# ============================================================
@st.cache_resource
def run_model(data_df):
    if data_df.empty or len(data_df) < 8:
        return None
    feats = ['visits', 'bmi', 'systolic', 'gender_code']
    X = data_df[feats]
    y = data_df['target']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=70, max_depth=5, random_state=42).fit(X_tr, y_tr)
    return model

if 'patient_id' in df.columns:
    summary_df = df.groupby('patient_id').agg(
        visits=('visit_id', 'count') if 'visit_id' in df.columns else ('patient_id', 'count'),
        bmi=('bmi', 'mean'),
        systolic=('systolic', 'mean'),
        gender_code=('gender_code', 'first')
    ).round(1)
    summary_df['target'] = ((summary_df['visits'] >= 3) | (summary_df['systolic'] >= 135)).astype(int)
    clf = run_model(summary_df)
else:
    clf = None

# ============================================================
# 4. Minimalist Sidebar Filters
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Data Scope")
    disease_filter = st.multiselect("กลุ่มอาการหลัก", sorted(df['disease_group'].unique()), default=sorted(df['disease_group'].unique()))
    gender_filter = st.multiselect("เพศ", sorted(df['gender'].unique()), default=sorted(df['gender'].unique()))
    
    if has_date:
        d_min, d_max = df['visit_date'].dropna().min().date(), df['visit_date'].dropna().max().date()
        date_pick = st.date_input("ช่วงเวลา", (d_min, d_max))
    else:
        date_pick = None

# Filtering Logic
mask = df['disease_group'].isin(disease_filter) & df['gender'].isin(gender_filter)
if date_pick and len(date_pick) == 2:
    mask &= df['visit_date'].between(pd.Timestamp(date_pick[0]), pd.Timestamp(date_pick[1])) | df['visit_date'].isna()

df_view = df[mask]

if df_view.empty:
    st.warning("⚠️ ไม่มีข้อมูลตรงกับตัวกรองที่เลือก")
    st.stop()

# ============================================================
# 5. Executive UI - Presentation Layout
# ============================================================

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Executive Health Intelligence Brief</div>
    <div class="hero-subtitle">สรุปทิศทางสุขภาพคนไข้ กลุ่มความเสี่ยงที่ต้องติดตาม และโอกาสต่อยอดเชิงกลยุทธ์</div>
</div>
""", unsafe_allow_html=True)

# 4 Key Bento Stat Cards
k1, k2, k3, k4 = st.columns(4)

total_visits = len(df_view)
unique_pts = df_view['patient_id'].nunique() if 'patient_id' in df_view.columns else total_visits
risk_pts = df_view[df_view['critical_risk'] == 1]['patient_id'].nunique() if 'patient_id' in df_view.columns else 0
risk_percent = (risk_pts / unique_pts * 100) if unique_pts else 0
avg_sys = df_view[df_view['systolic'] > 0]['systolic'].mean()

with k1:
    st.markdown(f"""
    <div class="bento-card">
        <span class="metric-badge badge-blue">Volume</span>
        <div class="metric-value">{total_visits:,}</div>
        <div class="metric-label">จำนวนการเข้ารับการตรวจทั้งหมด</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="bento-card">
        <span class="metric-badge badge-emerald">Patients</span>
        <div class="metric-value">{unique_pts:,}</div>
        <div class="metric-label">จำนวนผู้รับบริการรายบุคคล</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="bento-card">
        <span class="metric-badge badge-amber">Vitals</span>
        <div class="metric-value">{avg_sys:.1f} <span style="font-size:1.1rem;font-weight:400;color:#94A3B8;">mmHg</span></div>
        <div class="metric-label">ความดันตัวบนเฉลี่ย (Systolic)</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="bento-card">
        <span class="metric-badge badge-rose">High Risk</span>
        <div class="metric-value">{risk_percent:.1f}<span style="font-size:1.1rem;font-weight:400;color:#94A3B8;">%</span></div>
        <div class="metric-label">คนไข้กลุ่มเสี่ยงสูง ({risk_pts:,} คน)</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Tabs (Figma Segmented Pill Style)
tab1, tab2, tab3 = st.tabs([
    "📊 Strategic Overview", 
    "🚨 Risk Identification", 
    "💼 Revenue & Care Conversion"
])

# ----------------- TAB 1: Strategic Overview -----------------
with tab1:
    c_left, c_right = st.columns([3, 2])
    
    with c_left:
        st.markdown("**ปริมาณการใช้บริการรายเดือนตามกลุ่มโรค**")
        if has_date:
            monthly_data = df_view.groupby(['period', 'disease_group']).size().reset_index(name='count')
            fig_trend = px.area(
                monthly_data, x='period', y='count', color='disease_group',
                color_discrete_sequence=['#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#E2E8F0']
            )
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(gridcolor='#F1F5F9')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            fig_bar = px.bar(
                df_view['disease_group'].value_counts().reset_index(),
                x='count', y='disease_group', orientation='h',
                color_discrete_sequence=['#3B82F6'], text_auto=True
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.markdown("**การกระจายตัวของสถานะความดันโลหิต**")
        bp_pie_data = df_view['bp_category'].value_counts().reset_index()
        bp_pie_data.columns = ['Category', 'Count']
        fig_donut = px.pie(
            bp_pie_data, names='Category', values='Count', hole=0.6,
            color='Category',
            color_discrete_map={
                'ปกติ (<120)': '#10B981',
                'เฝ้าระวัง (120-139)': '#F59E0B',
                'วิกฤต/สูง (≥140)': '#EF4444'
            }
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

# ----------------- TAB 2: Risk Identification -----------------
with tab2:
    st.markdown("**Correlation ระหว่างค่าเฉลี่ย BMI กับระดับความดันโลหิต**")
    
    col_chart, col_action = st.columns([3, 2])
    
    with col_chart:
        bmi_summary = df_view.groupby('bmi_category', observed=False).agg(
            mean_sys=('systolic', 'mean'),
            count=('patient_id', 'nunique') if 'patient_id' in df_view.columns else ('bmi', 'count')
        ).reset_index()
        
        fig_scatter = px.scatter(
            bmi_summary, x='bmi_category', y='mean_sys', size='count',
            color='mean_sys', color_continuous_scale='Sunsetdark',
            text=bmi_summary['mean_sys'].round(1)
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.add_hline(y=140, line_dash="dot", line_color="#EF4444", annotation_text="เกณฑ์ความดันสูง (140 mmHg)")
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_action:
        st.markdown("**⚡ สรุปเพื่อการตัดสินใจเชิงคลินิก**")
        st.markdown(f"""
        <div class="action-box">
            <b>1. กลุ่มเฝ้าระวังพิเศษ:</b> มีคนไข้กลุ่มเสี่ยงรวม <b>{risk_pts:,} ราย</b> ({risk_percent:.1f}%) ที่มีค่า BMI ≥ 25 และความดันเกิน 140 mmHg<br><br>
            <b>2. ทิศทางนโยบาย:</b> ควรเปิด Fast-track คัดกรองโรคหลอดเลือดหัวใจ (Cardiovascular Pathway) เพื่อป้องกันการเข้ารับการรักษาแบบฉุกเฉิน
        </div>
        """, unsafe_allow_html=True)
        
        if risk_pts > 0 and 'patient_id' in df_view.columns:
            st.markdown("<br>**ตัวอย่างเคสที่มีความเสี่ยงสูง (Top 5 Priority)**", unsafe_allow_html=True)
            top_risks = df_view[df_view['critical_risk'] == 1][['patient_id', 'disease_group', 'bmi', 'systolic']].drop_duplicates('patient_id').head(5)
            st.dataframe(top_risks, use_container_width=True, hide_index=True)

# ----------------- TAB 3: Revenue & Care Conversion -----------------
with tab3:
    st.markdown("**ระบบคัดกรองความพร้อมรับแพ็กเกจตรวจสุขภาพเชิงรุก (Predictive Lead Score)**")
    
    if clf is not None and 'patient_id' in df.columns:
        p_list = summary_df.index.tolist()
        
        c_p1, c_p2 = st.columns([1, 2])
        
        with c_p1:
            picked_id = st.selectbox("ค้นหารหัสคนไข้เพื่อดูคะแนน:", p_list[:100])
            patient_row = summary_df.loc[[picked_id]]
            
            prob = clf.predict_proba(patient_row[['visits', 'bmi', 'systolic', 'gender_code']])[0][1]
            score = int(prob * 100)
            
            badge_color = "#10B981" if score >= 60 else ("#F59E0B" if score >= 35 else "#64748B")
            
            st.markdown(f"""
            <div class="bento-card" style="text-align:center; padding: 30px 10px;">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">CONVERSION READINESS</div>
                <div style="font-size:3rem; font-weight:700; color:{badge_color}; margin: 8px 0;">{score}%</div>
                <div style="font-size:0.8rem; color:#94A3B8;">ความน่าจะเป็นในการตอบรับโปรแกรมสุขภาพ</div>
            </div>
            """, unsafe_allow_html=True)

        with c_p2:
            st.markdown("**Next Best Action สำหรับทีมประสานงาน**")
            if score >= 60:
                st.markdown("""
                <div style="padding:16px; border-radius:12px; background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46;">
                    <b>🔥 แนะนำโปรแกรม Comprehensive Wellness Check</b><br>
                    คนไข้มีความถี่การเข้าใช้บริการต่อเนื่องและมีประวัติดัชนีสุขภาพที่ต้องการการติดตามระยะยาว 
                    ควรส่งต่อทีมลูกค้าสัมพันธ์เพื่อแนะนำโปรแกรมตรวจสุขภาพชุดใหญ่
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding:16px; border-radius:12px; background:#F8FAFC; border:1px solid #E2E8F0; color:#475569;">
                    <b>🌱 แนะนำการติดตามผลทั่วไป (Standard Follow-up)</b><br>
                    ส่งบทความสาระสุขภาพเกี่ยวกับการควบคุมอาหารและการออกกำลังกายผ่าน LINE Official คลินิกตามรอบปกติ
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>**สัดส่วนความสำคัญของตัวแปรในการทำนาย (Feature Drivers)**", unsafe_allow_html=True)
            importances = pd.DataFrame({
                'Feature': ['ความถี่เข้ารับบริการ', 'ดัชนีมวลกาย (BMI)', 'ความดันตัวบน (Systolic)', 'เพศ'],
                'Weight': clf.feature_importances_ * 100
            }).sort_values('Weight', ascending=True)
            
            fig_imp = px.bar(importances, x='Weight', y='Feature', orientation='h', color_discrete_sequence=['#0F172A'], text_auto='.1f')
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=160, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("ℹ️ ต้องมีข้อมูล `patient_id` เพื่อเปิดใช้งานโมเดลการประเมินรายบุคคล")
