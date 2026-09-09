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
# 1. ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(page_title="Data 4 U", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { color: #1E88E5; font-weight: bold; }
    div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1E88E5; }
    h1, h2, h3 { color: #0D47A1; }
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. โหลดและเตรียมข้อมูล
# ============================================================
@st.cache_data
def load_data():
    file_path = "visits_cleaned.csv"
    if not os.path.exists(file_path):
        return None, False, False
        
    df = pd.read_csv(file_path)

    # -- วันที่ --
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        has_valid_date = df['visit_date'].notna().any()
    else:
        has_valid_date = False

    if has_valid_date:
        df['month'] = df['visit_date'].dt.to_period('M').astype(str)
        df['weekday'] = df['visit_date'].dt.day_name()
        df['week'] = df['visit_date'].dt.to_period('W').astype(str)
    else:
        df['month'] = 'ไม่ระบุ'
        df['weekday'] = np.nan
        df['week'] = 'ไม่ระบุ'

    # -- ความดันโลหิต --
    if 'bp_raw' in df.columns:
        df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
        split_data = df['bp_raw'].str.split(' / ', expand=True)
        df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
        df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0) if split_data.shape[1] > 1 else 0
    else:
        df['systolic_bp'] = 0
        df['diastolic_bp'] = 0

    # -- เพศ และ BMI --
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('ไม่ระบุ')
        df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    else:
        df['gender'] = 'ไม่ระบุ'
        df['gender_code'] = 0.5

    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        valid_bmi = df['bmi'].dropna()
        bmi_median = valid_bmi.median() if not valid_bmi.empty else 22.0
        df['bmi'] = df['bmi'].fillna(bmi_median)
    else:
        df['bmi'] = 22.0

    # -- อายุ --
    has_valid_age = False
    if 'age_at_visit' in df.columns:
        df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce')
        has_valid_age = df['age_at_visit'].notna().any()
        
    if has_valid_age:
        valid_age = df['age_at_visit'].dropna()
        age_median = valid_age.median() if not valid_age.empty else 30.0
        df['age_at_visit'] = df['age_at_visit'].fillna(age_median)
        df['age_group'] = pd.cut(
            df['age_at_visit'], bins=[0, 20, 40, 60, 150], labels=['0-20', '21-40', '41-60', '60+']
        ).astype(str).replace('nan', 'ไม่ระบุ')
    else:
        df['age_group'] = 'ไม่ระบุ'
        df['age_at_visit'] = 0

    # -- จัดกลุ่ม BMI (มาตรฐาน WHO คร่าว ๆ) --
    df['bmi_group'] = pd.cut(
        df['bmi'], bins=[0, 18.5, 23, 25, 30, 100],
        labels=['ผอม', 'ปกติ', 'ท้วม', 'อ้วน', 'อ้วนมาก']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    if 'patient_id' in df.columns and 'visit_id' in df.columns:
        df['monthly_visit_count'] = df.groupby(['patient_id', 'month'])['visit_id'].transform('count')
    else:
        df['monthly_visit_count'] = 1

    df['bp_category'] = pd.cut(
        df['systolic_bp'], bins=[-1, 120, 140, 300], labels=['ปกติ', 'เสี่ยง', 'สูง']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # -- Feature engineering: กลุ่มเสี่ยง (BMI สูง + ความดันสูง) --
    df['is_high_risk'] = ((df['bmi'] >= 25) & (df['systolic_bp'] >= 140)).astype(int)

    # ป้องกันกรณีไม่มี disease_group
    if 'disease_group' not in df.columns:
        df['disease_group'] = 'ไม่ระบุ'

    return df, has_valid_date, has_valid_age

df, has_valid_date, has_valid_age = load_data()

# ตรวจสอบว่าโหลดไฟล์ได้หรือไม่
if df is None:
    st.error("❌ ไม่พบไฟล์ 'visits_cleaned.csv' กรุณาตรวจสอบว่ามีไฟล์อยู่ในโฟลเดอร์เดียวกันกับสคริปต์")
    st.stop()

has_diagnosis = 'diagnosis_text' in df.columns and df['diagnosis_text'].notna().any()
has_clinic = 'clinic_name' in df.columns and df['clinic_name'].notna().any()

# ============================================================
# 3. เตรียมโมเดล
# ============================================================
@st.cache_resource
def train_scoring_model(df_model):
    if df_model.empty or len(df_model) < 10:
        return None, None
    X = df_model[['frequency', 'bmi', 'systolic_bp', 'gender_code']]
    y = df_model['bought_package']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except ValueError:
        auc = None
    return model, auc

# เตรียมข้อมูลสำหรับ Model
if 'patient_id' in df.columns and 'visit_id' in df.columns:
    df_model = df.groupby('patient_id').agg(
        {'visit_id': 'count', 'bmi': 'mean', 'systolic_bp': 'mean', 'gender_code': 'first'}
    ).rename(columns={'visit_id': 'frequency'})
    df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)
    model, auc_score = train_scoring_model(df_model)
    X_cols = ['frequency', 'bmi', 'systolic_bp', 'gender_code']
else:
    model, auc_score = None, None

# ============================================================
# 4. Sidebar Control Panel
# ============================================================
st.sidebar.markdown("## ⚙️ Control Panel")

# Alerts สำหรับฟีเจอร์ที่หายไป
if not has_valid_date:
    st.sidebar.warning("⚠️ ไม่มีข้อมูลวันที่ ปิดการใช้งานตัวกรองเวลา")
if not has_valid_age:
    st.sidebar.warning("⚠️ ไม่มีข้อมูลอายุ ปิดการใช้งานตัวกรองอายุ")

disease_sel = st.sidebar.multiselect("เลือกกลุ่มโรค", sorted(df['disease_group'].unique()), default=df['disease_group'].unique())
gender_sel = st.sidebar.multiselect("เลือกเพศ", sorted(df['gender'].unique()), default=df['gender'].unique())

if has_valid_age:
    age_min, age_max = int(df['age_at_visit'].min()), int(df['age_at_visit'].max())
    age_range = st.sidebar.slider("ช่วงอายุ", age_min, age_max, (age_min, age_max))
else:
    age_range = None

if has_valid_date:
    valid_dates = df['visit_date'].dropna()
    d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input("ช่วงวันที่", (d_min, d_max), min_value=d_min, max_value=d_max)
else:
    date_range = None

if has_clinic:
    clinic_sel = st.sidebar.multiselect("เลือกคลินิก", sorted(df['clinic_name'].dropna().unique()), default=df['clinic_name'].dropna().unique())

# --- Apply Filters ---
mask = df['disease_group'].isin(disease_sel) & df['gender'].isin(gender_sel)

if age_range is not None:
    mask &= df['age_at_visit'].between(age_range[0], age_range[1])
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask &= df['visit_date'].between(start, end) | df['visit_date'].isna()
if has_clinic:
    mask &= df['clinic_name'].isin(clinic_sel)

df_f = df[mask]

# ============================================================
# 5. Dashboard หลัก (แบ่งเป็น Tabs)
# ============================================================
st.title("🏥 Data 4 U: Clinic Dashboard")

if df_f.empty:
    st.warning("⚠️ ไม่พบข้อมูลที่ตรงกับเงื่อนไขการกรอง กรุณาปรับเปลี่ยนตัวกรองในแถบด้านข้าง")
    st.stop()

# สร้าง Tabs
tab1, tab2, tab3 = st.tabs(["📊 ภาพรวม (Overview)", "🔍 วิเคราะห์เชิงลึก (Deep Analytics)", "🎯 ระบบประเมิน (Scoring)"])

# ---------------- Tab 1: ภาพรวม ----------------
with tab1:
    st.markdown("### สรุปตัวชี้วัด (KPIs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("เคสรับบริการทั้งหมด", f"{len(df_f):,}")
    
    unique_patients = df_f['patient_id'].nunique() if 'patient_id' in df_f.columns else 0
    c2.metric("จำนวนผู้ป่วย (คน)", f"{unique_patients:,}")
    c3.metric("อายุเฉลี่ย (ปี)", f"{df_f['age_at_visit'].mean():.1f}" if has_valid_age else "N/A")
    c4.metric("BMI เฉลี่ย", f"{df_f['bmi'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 แนวโน้มการรักษา (แบ่งตามกลุ่มโรค)")
        if has_valid_date:
            trend_data = df_f.groupby(['month', 'disease_group']).size().reset_index(name='counts')
            fig1 = px.area(
                trend_data, x='month', y='counts', color='disease_group', 
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fig_disease_only = px.bar(
                df_f['disease_group'].value_counts().reset_index(name='counts'),
                x='disease_group', y='counts', color='disease_group', 
                color_discrete_sequence=px.colors.qualitative.Pastel, text_auto=True
            )
            st.plotly_chart(fig_disease_only, use_container_width=True)
            
    with col2:
        st.subheader("👥 สัดส่วนผู้ป่วยตามเพศ")
        fig2 = px.pie(df_f, names='gender', hole=0.5, color_discrete_sequence=['#90CAF9', '#1565C0', '#BDBDBD'])
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 2: วิเคราะห์เชิงลึก ----------------
with tab2:
    st.markdown("### 📊 การวิเคราะห์เชิงสุขภาพและพฤติกรรม")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        if has_valid_date and has_valid_age:
            st.markdown("##### 🗓️ พฤติกรรมการมารักษาแยกตามอายุและเดือน")
            pivot_data = df_f.groupby(['month', 'age_group'], observed=False).size().reset_index(name='count')
            fig_heat = px.density_heatmap(pivot_data, x='month', y='age_group', z='count', color_continuous_scale='Blues')
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.markdown("##### 🩺 สัดส่วนระดับความดันโลหิตโดยรวม")
            bp_pie = df_f['bp_category'].value_counts().reset_index()
            bp_pie.columns = ['bp_category', 'count']
            fig_bp_pie = px.pie(bp_pie, names='bp_category', values='count', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_bp_pie, use_container_width=True)

    with d_col2:
        st.markdown("##### 🩺 ระดับความดันแยกตามกลุ่มโรค")
        bp_summary = df_f.groupby(['disease_group', 'bp_category'], observed=False).size().reset_index(name='count')
        fig_bar = px.bar(bp_summary, x='disease_group', y='count', color='bp_category', barmode='group',
                         color_discrete_sequence=px.colors.qualitative.Pastel, text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    c_bmi1, c_bmi2 = st.columns(2)
    
    with c_bmi1:
        st.markdown("##### ⚖️ BMI กับความดันโลหิตเฉลี่ย")
        bmi_bp = df_f.groupby('bmi_group', observed=False).agg(
            avg_systolic=('systolic_bp', 'mean'),
            avg_diastolic=('diastolic_bp', 'mean')
        ).reset_index()
        fig_bmi_bp = go.Figure()
        fig_bmi_bp.add_bar(x=bmi_bp['bmi_group'], y=bmi_bp['avg_systolic'], name='Systolic', marker_color='#1E88E5', text=bmi_bp['avg_systolic'].round(1))
        fig_bmi_bp.add_bar(x=bmi_bp['bmi_group'], y=bmi_bp['avg_diastolic'], name='Diastolic', marker_color='#90CAF9', text=bmi_bp['avg_diastolic'].round(1))
        fig_bmi_bp.update_layout(barmode='group')
        st.plotly_chart(fig_bmi_bp, use_container_width=True)

    with c_bmi2:
        st.markdown("##### 📈 ความสัมพันธ์ BMI กับความดัน (รายเคส)")
        df_f_clean = df_f.dropna(subset=['bmi', 'systolic_bp', 'monthly_visit_count'])
        df_f_clean = df_f_clean[df_f_clean['systolic_bp'] > 0]
        if not df_f_clean.empty:
            fig_scatter = px.scatter(
                df_f_clean, x='bmi', y='systolic_bp', color='bp_category', size='monthly_visit_count',
                hover_data=['patient_id'] if 'patient_id' in df_f_clean.columns else None, 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับกราฟนี้")

    st.markdown("---")
    st.markdown("### 🚨 การเฝ้าระวังกลุ่มเสี่ยง (Risk Monitoring)")
    risk_patients = df_f[df_f['is_high_risk'] == 1]['patient_id'].nunique() if 'patient_id' in df_f.columns else 0
    total_patients = df_f['patient_id'].nunique() if 'patient_id' in df_f.columns else 1
    risk_pct = (risk_patients / total_patients * 100) if total_patients > 0 else 0
    
    rc1, rc2 = st.columns([1, 2])
    rc1.metric("ผู้ป่วยกลุ่มเสี่ยง (BMI≥25 และ BP≥140)", f"{risk_patients:,} คน", f"{risk_pct:.1f}% ของผู้ป่วยทั้งหมด", delta_color="inverse")
    
    if risk_patients > 0:
        rc2.markdown("**รายชื่อผู้ป่วยกลุ่มเสี่ยง (Top 10)**")
        rc2.dataframe(
            df_f[df_f['is_high_risk'] == 1][['patient_id', 'disease_group', 'bmi', 'systolic_bp', 'diastolic_bp']]
            .drop_duplicates('patient_id').head(10),
            use_container_width=True, hide_index=True
        )

# ---------------- Tab 3: ระบบประเมิน ----------------
with tab3:
    st.markdown("### 🎯 ระบบให้คะแนนความสนใจแพ็กเกจสุขภาพ")
    if model is not None:
        st.info(
            "⚠️ **หมายเหตุ:** โมเดลนี้สร้างขึ้นจาก Rule-based Label (ความถี่ > 2 หรือ Systolic > 130) "
            "เพื่อใช้เป็นตัวอย่างการคัดกรองเบื้องต้น"
            + (f" (AUC Score: {auc_score:.2f})" if auc_score is not None else "")
        )

        valid_patients = df_model.index.tolist()
        if valid_patients:
            p_id = st.selectbox("🔍 ค้นหา / เลือก Patient ID เพื่อดูคะแนน:", valid_patients)
            p_data = df_model.loc[[p_id]]
            prob = model.predict_proba(p_data[X_cols])[0][1]

            col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
            col_p1.metric("โอกาสสนใจแพ็กเกจ", f"{prob*100:.1f}%")
            
            with col_p2:
                st.markdown("<br>", unsafe_allow_html=True) # เว้นบรรทัดให้ตรงกัน
                if prob > 0.5:
                    st.success("🔥 แนวโน้มสูง: ควรติดต่อเสนอแพ็กเกจ")
                else:
                    st.info("❄️ แนวโน้มทั่วไป: ติดตามผลตามปกติ")

            st.markdown("##### 💡 ความสำคัญของปัจจัยที่ส่งผลต่อคะแนน (Feature Importance)")
            feat_imp = pd.DataFrame({'Feature': X_cols, 'Importance': model.feature_importances_})
            fig_imp = px.bar(feat_imp.sort_values('Importance', ascending=True), 
                             x='Importance', y='Feature', orientation='h', 
                             color='Importance', color_continuous_scale='Blues', text_auto='.2f')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลผู้ป่วยสำหรับให้คะแนน")
    else:
        st.error("ไม่สามารถสร้างโมเดลได้ เนื่องจากข้อมูลไม่เพียงพอ หรือโครงสร้างไฟล์ไม่รองรับ")
