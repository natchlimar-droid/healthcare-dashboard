import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# ============================================================
# 1. ตั้งค่าหน้าเว็บและดีไซน์ UI ให้สะอาดตา
# ============================================================
st.set_page_config(
    page_title="Healthcare Intelligence Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f7f9fc; font-family: 'Sarabun', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
    }
    div[data-testid="stMetricValue"] {
        color: #1976D2;
        font-weight: 700;
    }
    .status-card {
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .risk-high { background-color: #ffebee; border-left: 5px solid #d32f2f; color: #b71c1c; }
    .risk-medium { background-color: #fff8e1; border-left: 5px solid #ffa000; color: #ff6f00; }
    .risk-low { background-color: #e8f5e9; border-left: 5px solid #388e3c; color: #1b5e20; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. การโหลดและเตรียมข้อมูล
# ============================================================
@st.cache_data
def load_and_clean_data():
    file_path = "visits_cleaned.csv"
    if not os.path.exists(file_path):
        return None, False, False
        
    df = pd.read_csv(file_path)

    # จัดการวันที่
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        has_date = df['visit_date'].notna().any()
    else:
        has_date = False

    if has_date:
        df['month_year'] = df['visit_date'].dt.strftime('%b %Y')
        df['month_period'] = df['visit_date'].dt.to_period('M').astype(str)
        df['weekday'] = df['visit_date'].dt.day_name()
    else:
        df['month_year'] = 'ไม่ระบุ'
        df['month_period'] = 'ไม่ระบุ'
        df['weekday'] = np.nan

    # แยกความดันโลหิต
    if 'bp_raw' in df.columns:
        df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
        bp_split = df['bp_raw'].str.split(' / ', expand=True)
        df['systolic_bp'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(0)
        df['diastolic_bp'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(0) if bp_split.shape[1] > 1 else 0
    else:
        df['systolic_bp'] = 0
        df['diastolic_bp'] = 0

    # เพศ และ BMI
    df['gender'] = df['gender'].fillna('ไม่ระบุ') if 'gender' in df.columns else 'ไม่ระบุ'
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        bmi_mid = df['bmi'].dropna().median() if not df['bmi'].dropna().empty else 22.5
        df['bmi'] = df['bmi'].fillna(bmi_mid)
    else:
        df['bmi'] = 22.5

    # ช่วงอายุ
    has_age = 'age_at_visit' in df.columns and df['age_at_visit'].notna().any()
    if has_age:
        df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce')
        age_mid = df['age_at_visit'].dropna().median() if not df['age_at_visit'].dropna().empty else 35
        df['age_at_visit'] = df['age_at_visit'].fillna(age_mid)
        df['age_group'] = pd.cut(
            df['age_at_visit'],
            bins=[0, 20, 40, 60, 120],
            labels=['วัยเด็ก-เยาวชน (≤20)', 'วัยเริ่มทำงาน (21-40)', 'วัยกลางคน (41-60)', 'ผู้สูงวัย (60+)']
        ).astype(str).replace('nan', 'ไม่ระบุ')
    else:
        df['age_group'] = 'ไม่ระบุ'
        df['age_at_visit'] = 0

    # จัดกลุ่ม BMI ตามมาตรฐานเอเชีย/ไทย
    df['bmi_status'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 23, 25, 30, 200],
        labels=['น้ำหนักน้อย', 'น้ำหนักสมส่วน', 'น้ำหนักเกิน (ท้วม)', 'โรคอ้วนระดับ 1', 'โรคอ้วนระดับ 2']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # จัดกลุ่มความดันตัวบน (Systolic)
    df['bp_status'] = pd.cut(
        df['systolic_bp'],
        bins=[-1, 120, 139, 300],
        labels=['ปกติ (<120)', 'เริ่มสูง/เฝ้าระวัง (120-139)', 'ความดันสูง (≥140)']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # นิยามผู้ป่วยกลุ่มเสี่ยงร่วม (Metabolic syndrome alert)
    df['is_risk_case'] = ((df['bmi'] >= 25) & (df['systolic_bp'] >= 140)).astype(int)

    if 'patient_id' in df.columns and 'visit_id' in df.columns:
        df['total_visits'] = df.groupby('patient_id')['visit_id'].transform('count')
    else:
        df['total_visits'] = 1

    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ทั่วไป'

    return df, has_date, has_age

df, has_date, has_age = load_and_clean_data()

if df is None:
    st.error("⚠️ ไม่พบไฟล์ `visits_cleaned.csv` กรุณาตรวจสอบการเชื่อมต่อไฟล์ข้อมูล")
    st.stop()

# ============================================================
# 3. สร้างโมเดลวิเคราะห์ความพร้อมแพ็กเกจสุขภาพ
# ============================================================
@st.cache_resource
def build_prediction_model(data_df):
    if data_df.empty or len(data_df) < 10:
        return None, None
    X = data_df[['visit_freq', 'bmi', 'systolic_bp', 'gender_code']]
    y = data_df['target_candidate']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except:
        auc = None
    return clf, auc

if 'patient_id' in df.columns:
    df_patient_summary = df.groupby('patient_id').agg(
        visit_freq=('visit_id', 'count') if 'visit_id' in df.columns else ('patient_id', 'count'),
        bmi=('bmi', 'mean'),
        systolic_bp=('systolic_bp', 'mean'),
        gender_code=('gender_code', 'first')
    ).round(1)

    # กฎคัดกรองกลุ่มเป้าหมายสำหรับ Health Checkup Package
    df_patient_summary['target_candidate'] = (
        (df_patient_summary['visit_freq'] >= 3) | (df_patient_summary['systolic_bp'] >= 135)
    ).astype(int)

    clf_model, auc_val = build_prediction_model(df_patient_summary)
    feature_cols = ['visit_freq', 'bmi', 'systolic_bp', 'gender_code']
else:
    clf_model, auc_val = None, None

# ============================================================
# 4. แถบตัวกรอง (Sidebar Filters)
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ ตัวกรองข้อมูล")
    
    selected_diseases = st.multiselect(
        "กลุ่มโรค / อาการ",
        options=sorted(df['disease_group'].unique()),
        default=sorted(df['disease_group'].unique())
    )
    
    selected_genders = st.multiselect(
        "เพศ",
        options=sorted(df['gender'].unique()),
        default=sorted(df['gender'].unique())
    )

    if has_age:
        min_age, max_age = int(df['age_at_visit'].min()), int(df['age_at_visit'].max())
        age_filter = st.slider("ช่วงอายุ", min_age, max_age, (min_age, max_age))
    else:
        age_filter = None

    if has_date:
        d_min = df['visit_date'].dropna().min().date()
        d_max = df['visit_date'].dropna().max().date()
        date_filter = st.date_input("ช่วงวันที่บันทึกเคส", (d_min, d_max), min_value=d_min, max_value=d_max)
    else:
        date_filter = None

# กรองข้อมูล
filter_mask = df['disease_group'].isin(selected_diseases) & df['gender'].isin(selected_genders)
if age_filter:
    filter_mask &= df['age_at_visit'].between(age_filter[0], age_filter[1])
if date_filter and isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    start_dt, end_dt = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    filter_mask &= df['visit_date'].between(start_dt, end_dt) | df['visit_date'].isna()

df_filtered = df[filter_mask]

# ============================================================
# 5. การแสดงผล Dashboard หลัก
# ============================================================
st.title("🩺 ข้อมูลสรุปบริการสุขภาพและคัดกรองความเสี่ยง")
st.caption("ระบบวิเคราะห์สถิติผู้ป่วย แนวโน้มสุขภาพ และแนะนำโอกาสการนำเสนอแพ็กเกจดูแลเฉพาะบุคคล")

if df_filtered.empty:
    st.warning("⚠️ ไม่พบข้อมูลตามเงื่อนไขที่เลือก กรุณาปรับเปลี่ยนตัวกรองทางแถบซ้ายมือ")
    st.stop()

# --- KPI Cards ภาพรวม ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
total_cases = len(df_filtered)
total_patients = df_filtered['patient_id'].nunique() if 'patient_id' in df_filtered.columns else total_cases
high_risk_cases = df_filtered[df_filtered['is_risk_case'] == 1]['patient_id'].nunique() if 'patient_id' in df_filtered.columns else 0
avg_systolic = df_filtered[df_filtered['systolic_bp'] > 0]['systolic_bp'].mean()

col_kpi1.metric("การรับบริการทั้งหมด", f"{total_cases:,} ครั้ง")
col_kpi2.metric("จำนวนคนไข้", f"{total_patients:,} คน")
col_kpi3.metric("ความดันตัวบนเฉลี่ย", f"{avg_systolic:.1f} mmHg" if pd.notna(avg_systolic) else "-")
col_kpi4.metric(
    "ผู้ป่วยกลุ่มเสี่ยงเฝ้าระวัง",
    f"{high_risk_cases:,} คน",
    f"{(high_risk_cases/total_patients*100):.1f}% ของผู้ป่วย" if total_patients else None,
    delta_color="inverse"
)

st.markdown("---")

# --- แยกเนื้อหาด้วย Tabs ที่อ่านง่าย ---
tab_clinical, tab_trend, tab_crm = st.tabs([
    "🩺 1. ระดับสุขภาพและความเสี่ยง (Clinical Insights)",
    "📈 2. พฤติกรรมและแนวโน้มการรักษา (Patterns)",
    "🎯 3. โอกาสเสนอแพ็กเกจสุขภาพ (Health Packages)"
])

# ==================== TAB 1: สุขภาพและความเสี่ยง ====================
with tab_clinical:
    st.markdown("#### สถานะความดันโลหิตและดัชนีมวลกาย (BMI)")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        # สรุปกลุ่มความดันโลหิต
        bp_dist = df_filtered['bp_status'].value_counts().reset_index()
        bp_dist.columns = ['สถานะความดัน', 'จำนวนครั้ง']
        fig_bp = px.pie(
            bp_dist,
            names='สถานะความดัน',
            values='จำนวนครั้ง',
            color='สถานะความดัน',
            color_discrete_map={
                'ปกติ (<120)': '#81C784',
                'เริ่มสูง/เฝ้าระวัง (120-139)': '#FFB74D',
                'ความดันสูง (≥140)': '#E57373'
            },
            hole=0.45,
            title="สัดส่วนระดับความดันโลหิต (Systolic)"
        )
        st.plotly_chart(fig_bp, use_container_width=True)
        st.caption("💡 **คำอธิบาย:** คนไข้ในกลุ่ม 'ความดันสูง' ควรได้รับการนัดติดตามผลวัดซ้ำหรือแนะนำปรับพฤติกรรม")

    with col_plot2:
        # BMI vs ความดันโลหิตเฉลี่ย
        bmi_bp_rel = df_filtered.groupby('bmi_status', observed=False).agg(
            mean_systolic=('systolic_bp', 'mean'),
            patient_count=('patient_id', 'nunique') if 'patient_id' in df_filtered.columns else ('bmi', 'count')
        ).reset_index()

        fig_bar_bmi = px.bar(
            bmi_bp_rel,
            x='bmi_status',
            y='mean_systolic',
            text='mean_systolic',
            color='mean_systolic',
            color_continuous_scale='Reds',
            title="ความดันตัวบนเฉลี่ยในแต่ละกลุ่มดัชนีมวลกาย (BMI)",
            labels={'bmi_status': 'ระดับ BMI', 'mean_systolic': 'ความดันตัวบนเฉลี่ย (mmHg)'}
        )
        fig_bar_bmi.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bar_bmi.add_hline(y=140, line_dash="dash", line_color="#D32F2F", annotation_text="เกณฑ์เริ่มอันตราย (140)")
        st.plotly_chart(fig_bar_bmi, use_container_width=True)
        st.caption("💡 **คำอธิบาย:** แสดงความสัมพันธ์ว่ากลุ่มที่มีภาวะโรคอ้วนมีค่าเฉลี่ยความดันโลหิตแตะระดับเสี่ยงหรือไม่")

    # ตารางคนไข้กลุ่มเสี่ยงที่ต้องการการดูแลเร่งด่วน
    st.markdown("#### 🚨 รายชื่อผู้ป่วยกลุ่มเสี่ยงสูง (BMI เกินเกณฑ์ร่วมกับความดันสูง)")
    high_risk_list = df_filtered[df_filtered['is_risk_case'] == 1]
    if not high_risk_list.empty and 'patient_id' in df_filtered.columns:
        display_risk = high_risk_list[['patient_id', 'gender', 'disease_group', 'bmi', 'systolic_bp', 'diastolic_bp']].drop_duplicates('patient_id')
        display_risk.columns = ['รหัสผู้ป่วย', 'เพศ', 'กลุ่มโรค', 'BMI', 'ความดันตัวบน', 'ความดันตัวล่าง']
        st.dataframe(display_risk.head(10), use_container_width=True, hide_index=True)
        st.info("ℹ️ แสดงรายชื่อสูงสุด 10 รายการแรก เพื่อให้ทีมพยาบาลหรือเจ้าหน้าที่ประสานโทรติดตามอาการ")
    else:
        st.success("✅ ไม่พบผู้ป่วยที่เข้าเกณฑ์ความเสี่ยงสูงพร้อมกันในกลุ่มตัวกรองปัจจุบัน")

# ==================== TAB 2: พฤติกรรมและแนวโน้ม ====================
with tab_trend:
    st.markdown("#### สถิติและการกระจายตัวของการรับบริการ")
    
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        if has_date:
            monthly_trend = df_filtered.groupby(['month_period', 'disease_group']).size().reset_index(name='จำนวนเคส')
            fig_monthly = px.line(
                monthly_trend,
                x='month_period',
                y='จำนวนเคส',
                color='disease_group',
                markers=True,
                title="แนวโน้มจำนวนครั้งการมารับบริการแยกตามกลุ่มโรค (รายเดือน)",
                labels={'month_period': 'เดือน', 'จำนวนเคส': 'จำนวนเคส'}
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            disease_counts = df_filtered['disease_group'].value_counts().reset_index()
            disease_counts.columns = ['กลุ่มโรค', 'จำนวนครั้ง']
            fig_bar_dis = px.bar(
                disease_counts,
                x='กลุ่มโรค',
                y='จำนวนครั้ง',
                color='กลุ่มโรค',
                title="จำนวนการเข้ารับบริการแยกตามกลุ่มโรค",
                text_auto=True
            )
            st.plotly_chart(fig_bar_dis, use_container_width=True)

    with col_t2:
        if has_age:
            age_dist = df_filtered['age_group'].value_counts().reset_index()
            age_dist.columns = ['ช่วงอายุ', 'จำนวนครั้ง']
            fig_age = px.bar(
                age_dist,
                x='จำนวนครั้ง',
                y='ช่วงอายุ',
                orientation='h',
                color='ช่วงอายุ',
                title="สัดส่วนผู้เข้ารับบริการแยกตามช่วงวัย",
                text_auto=True
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลช่วงอายุสำหรับแสดงผลกราฟนี้")

# ==================== TAB 3: โอกาสเสนอแพ็กเกจสุขภาพ ====================
with tab_crm:
    st.markdown("#### 🎯 ระบบค้นหาผู้ป่วยที่เหมาะสมกับแพ็กเกจตรวจสุขภาพเฉพาะทาง")
    st.markdown("""
        ระบบใช้การประเมินจาก **ความถี่ในการมารักษา ค่าความดันโลหิตเฉลี่ย และดัชนีมวลกาย** 
        เพื่อช่วยระบุว่าคนไข้คนใดควรได้รับการแนะนำโปรแกรมตรวจสุขภาพเชิงป้องกัน (Preventive Checkup)
    """)

    if clf_model is not None and 'patient_id' in df.columns:
        patient_options = df_patient_summary.index.tolist()
        
        col_select, col_card = st.columns([1, 2])
        
        with col_select:
            selected_pid = st.selectbox("🔍 เลือกรหัสผู้ป่วย (Patient ID):", patient_options)
            patient_profile = df_patient_summary.loc[[selected_pid]]
            
            # คำนวณความน่าจะเป็น
            lead_prob = clf_model.predict_proba(patient_profile[feature_cols])[0][1]
            prob_percent = lead_prob * 100
            
            st.metric("คะแนนความเร่งด่วน / ความพร้อม", f"{prob_percent:.1f}%")

        with col_card:
            st.markdown("<br>", unsafe_allow_html=True)
            if prob_percent >= 65:
                st.markdown("""
                <div class="status-card risk-high">
                    <b>🔥 คำแนะนำ: ควรนำเสนอแพ็กเกจดูแลต่อเนื่อง</b><br>
                    คนไข้รายนี้มีความถี่ในการรับบริการต่อเนื่อง หรือมีความดันโลหิตสะสมที่ต้องเฝ้าระวัง เหมาะสำหรับ:
                    <ul>
                        <li>แพ็กเกจตรวจคัดกรองหลอดเลือดและหัวใจ (Cardiovascular Screening)</li>
                        <li>แพ็กเกจตรวจสุขภาพประจำปีกลุ่ม Silver/Gold</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif prob_percent >= 35:
                st.markdown("""
                <div class="status-card risk-medium">
                    <b>⚖️ คำแนะนำ: ติดตามผลตามระยะปกติ</b><br>
                    คนไข้อยู่ในระดับปานกลาง สามารถส่งข้อมูลสาระความรู้ด้านการดูแลตนเองผ่าน SMS หรือ LINE Official คลินิก
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card risk-low">
                    <b>🌿 คำแนะนำ: ติดตามผลทั่วไป</b><br>
                    สุขภาพอยู่ในเกณฑ์ปกติ ความถี่รับบริการไม่บ่อย ไม่จำเป็นต้องเร่งติดตามแพ็กเกจเฉพาะทาง
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 📌 ปัจจัยสำคัญที่มีผลต่อการคัดกรอง (Feature Importance)")
        
        feat_labels = {
            'visit_freq': 'ความถี่ในการมารักษา',
            'systolic_bp': 'ระดับความดันตัวบนเฉลี่ย',
            'bmi': 'ดัชนีมวลกาย (BMI)',
            'gender_code': 'เพศ'
        }
        
        importance_df = pd.DataFrame({
            'ปัจจัย': [feat_labels.get(c, c) for c in feature_cols],
            'น้ำหนักความสำคัญ (%)': clf_model.feature_importances_ * 100
        }).sort_values('น้ำหนักความสำคัญ (%)', ascending=True)

        fig_imp = px.bar(
            importance_df,
            x='น้ำหนักความสำคัญ (%)',
            y='ปัจจัย',
            orientation='h',
            text='น้ำหนักความสำคัญ (%)',
            color='น้ำหนักความสำคัญ (%)',
            color_continuous_scale='Blues'
        )
        fig_imp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("⚠️ ไม่พบข้อมูลเพียงพอสำหรับการประเมินรายบุคคล")
