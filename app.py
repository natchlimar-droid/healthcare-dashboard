import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(page_title="Healthcare Pro Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1E88E5; }
    .stPlotlyChart { background-color: white; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0D47A1; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. โหลดและเตรียมข้อมูล
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df['month'] = df['visit_date'].dt.to_period('M').astype(str)
    df['weekday'] = df['visit_date'].dt.day_name()
    df['week'] = df['visit_date'].dt.to_period('W').astype(str)

    # -- ความดันโลหิต --
    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
    split_data = df['bp_raw'].str.split(' / ', expand=True)
    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)

    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce').fillna(df['age_at_visit'].median())
    df['age_group'] = pd.cut(
        df['age_at_visit'], bins=[0, 20, 40, 60, 100], labels=['0-20', '21-40', '41-60', '60+']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # -- กลุ่ม BMI (มาตรฐาน WHO แบบคร่าว ๆ) --
    df['bmi_group'] = pd.cut(
        df['bmi'], bins=[0, 18.5, 23, 25, 30, 100],
        labels=['ผอม', 'ปกติ', 'ท้วม', 'อ้วน', 'อ้วนมาก']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    df['monthly_visit_count'] = df.groupby(['patient_id', 'month'])['visit_id'].transform('count')
    df['bp_category'] = pd.cut(
        df['systolic_bp'], bins=[0, 120, 140, 200], labels=['ปกติ', 'เสี่ยง', 'สูง']
    ).astype(str).replace('nan', 'ไม่ระบุ')

    # -- Feature engineering: กลุ่มเสี่ยง (BMI สูง + ความดันสูง) --
    df['is_high_risk'] = ((df['bmi'] >= 25) & (df['systolic_bp'] >= 140)).astype(int)

    return df

df = load_data()
has_diagnosis = 'diagnosis_text' in df.columns
has_clinic = 'clinic_name' in df.columns

# ============================================================
# 3. เตรียมโมเดล (รีเฟรมเป็น "กฎการให้คะแนนความสนใจ" อย่างตรงไปตรงมา)
# ============================================================
# หมายเหตุสำคัญ: เนื่องจากไม่มีข้อมูลการซื้อแพ็กเกจจริง เราสร้าง label จากกฎ
# (ความถี่มาใช้บริการ + ความดันสูง) แล้วเทรนโมเดลจากฟีเจอร์ชุดเดียวกัน
# ผลลัพธ์จึงเป็น "การจำลองกฎการให้คะแนน" ไม่ใช่การทำนายพฤติกรรมจริง
# ควรแทนที่ target ด้วยข้อมูลการซื้อแพ็กเกจจริงเมื่อมี เพื่อให้โมเดลมีความหมาย
@st.cache_resource
def train_scoring_model(df_model):
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

df_model = df.groupby('patient_id').agg(
    {'visit_id': 'count', 'bmi': 'mean', 'systolic_bp': 'mean', 'gender_code': 'first'}
).rename(columns={'visit_id': 'frequency'})
df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)
model, auc_score = train_scoring_model(df_model)
X_cols = ['frequency', 'bmi', 'systolic_bp', 'gender_code']

# ============================================================
# 4. Sidebar & Dashboard หลัก
# ============================================================
st.sidebar.markdown("## ⚙️ Control Panel")
disease_sel = st.sidebar.multiselect("เลือกกลุ่มโรค", df['disease_group'].unique(), default=df['disease_group'].unique())
gender_sel = st.sidebar.multiselect("เลือกเพศ", df['gender'].unique(), default=df['gender'].unique())

age_min, age_max = int(df['age_at_visit'].min()), int(df['age_at_visit'].max())
age_range = st.sidebar.slider("ช่วงอายุ", age_min, age_max, (age_min, age_max))

valid_dates = df['visit_date'].dropna()
if not valid_dates.empty:
    d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input("ช่วงวันที่", (d_min, d_max), min_value=d_min, max_value=d_max)
else:
    date_range = None

if has_clinic:
    clinic_sel = st.sidebar.multiselect("เลือกคลินิก", df['clinic_name'].unique(), default=df['clinic_name'].unique())

st.title("🏥 Healthcare Executive Dashboard")

mask = (
    df['disease_group'].isin(disease_sel)
    & df['gender'].isin(gender_sel)
    & df['age_at_visit'].between(age_range[0], age_range[1])
)
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask &= df['visit_date'].between(start, end) | df['visit_date'].isna()
if has_clinic:
    mask &= df['clinic_name'].isin(clinic_sel)

df_f = df[mask]

# --- KPI Section ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("เคสทั้งหมด", f"{len(df_f):,}")
c2.metric("ผู้ป่วย", f"{df_f['patient_id'].nunique():,}")
c3.metric("อายุเฉลี่ย", f"{df_f['age_at_visit'].mean():.1f} ปี")
c4.metric("BMI เฉลี่ย", f"{df_f['bmi'].mean():.1f}")

# --- กราฟหลัก ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📈 แนวโน้มการรักษา")
    fig1 = px.area(
        df_f.groupby(['month', 'disease_group']).size().reset_index(name='counts'),
        x='month', y='counts', color='disease_group', color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("👥 สัดส่วนเพศ")
    fig2 = px.pie(df_f, names='gender', hole=0.6, color_discrete_sequence=['#90CAF9', '#1565C0'])
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# --- Deep Analytics ---
# ============================================================
st.markdown("---")
st.subheader("📊 การวิเคราะห์เชิงลึก (Deep Analytics)")
d_col1, d_col2 = st.columns(2)

with d_col1:
    st.markdown("##### 🗓️ พฤติกรรมการมาใช้บริการ (รายเดือน)")
    pivot_data = df_f.groupby(['month', 'age_group'], observed=False).size().reset_index(name='count')
    fig_heat = px.density_heatmap(pivot_data, x='month', y='age_group', z='count', color_continuous_scale='Blues')
    st.plotly_chart(fig_heat, use_container_width=True)

with d_col2:
    st.markdown("##### 🩺 ระดับความดันแยกตามช่วงอายุ")
    bp_summary = df_f.groupby(['age_group', 'bp_category'], observed=False).size().reset_index(name='count')
    fig_bar = px.bar(bp_summary, x='age_group', y='count', color='bp_category', barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- BMI กับความดันโลหิต (grouped view) ---
st.markdown("##### ⚖️ BMI กับความดันโลหิตเฉลี่ย (ตามกลุ่ม BMI)")
bmi_bp = df_f.groupby('bmi_group', observed=False).agg(
    avg_systolic=('systolic_bp', 'mean'),
    avg_diastolic=('diastolic_bp', 'mean'),
    n_patients=('patient_id', 'nunique')
).reset_index()
fig_bmi_bp = go.Figure()
fig_bmi_bp.add_bar(x=bmi_bp['bmi_group'], y=bmi_bp['avg_systolic'], name='Systolic เฉลี่ย', marker_color='#1E88E5')
fig_bmi_bp.add_bar(x=bmi_bp['bmi_group'], y=bmi_bp['avg_diastolic'], name='Diastolic เฉลี่ย', marker_color='#90CAF9')
fig_bmi_bp.update_layout(barmode='group')
st.plotly_chart(fig_bmi_bp, use_container_width=True)

# --- Scatter: BMI vs ความดัน (รายบุคคล) ---
st.markdown("##### 📈 ความสัมพันธ์ BMI กับความดัน (รายเคส)")
df_f_clean = df_f.dropna(subset=['bmi', 'systolic_bp', 'monthly_visit_count'])
fig_scatter = px.scatter(
    df_f_clean, x='bmi', y='systolic_bp', color='bp_category', size='monthly_visit_count',
    hover_data=['patient_id'], color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- วิเคราะห์กลุ่มโรค / diagnosis_text ---
if has_diagnosis:
    st.markdown("##### 🔤 โรค/อาการที่พบบ่อยที่สุด (จาก diagnosis_text)")
    diag_counts = (
        df_f['diagnosis_text'].dropna().astype(str).str.strip()
        .value_counts().head(15).reset_index()
    )
    diag_counts.columns = ['diagnosis', 'count']
    fig_diag = px.bar(diag_counts.sort_values('count'), x='count', y='diagnosis', orientation='h', color='count', color_continuous_scale='Blues')
    st.plotly_chart(fig_diag, use_container_width=True)
else:
    st.caption("ℹ️ ไม่พบคอลัมน์ diagnosis_text ในข้อมูล — ข้ามส่วนวิเคราะห์ข้อความวินิจฉัย")

# --- เปรียบเทียบเบาหวาน vs ความดันโลหิตสูง (ตาม disease_group) ---
st.markdown("##### 🆚 เปรียบเทียบจำนวนผู้ป่วยตามกลุ่มโรค")
disease_counts = df_f['disease_group'].value_counts().reset_index()
disease_counts.columns = ['disease_group', 'count']
fig_disease = px.bar(disease_counts, x='disease_group', y='count', color='disease_group', color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_disease, use_container_width=True)

# --- การเฝ้าระวังความเสี่ยง (Risk Monitoring) ---
st.markdown("##### 🚨 กลุ่มเสี่ยง (BMI ≥ 25 และ Systolic ≥ 140)")
risk_patients = df_f[df_f['is_high_risk'] == 1]['patient_id'].nunique()
total_patients = df_f['patient_id'].nunique()
risk_pct = (risk_patients / total_patients * 100) if total_patients else 0
rc1, rc2 = st.columns(2)
rc1.metric("จำนวนผู้ป่วยกลุ่มเสี่ยง", f"{risk_patients:,}", f"{risk_pct:.1f}% ของผู้ป่วยทั้งหมด")
rc2.dataframe(
    df_f[df_f['is_high_risk'] == 1][['patient_id', 'bmi', 'systolic_bp', 'age_at_visit']]
    .drop_duplicates('patient_id').head(20),
    use_container_width=True
)

# --- ประสิทธิภาพ/ความหนาแน่นของคลินิก ---
if has_clinic:
    st.markdown("##### 🏢 ความหนาแน่นผู้ป่วยตามคลินิก (รายสัปดาห์ x วัน)")
    clinic_pivot = df_f.dropna(subset=['weekday']).groupby(['clinic_name', 'weekday']).size().reset_index(name='count')
    fig_clinic = px.density_heatmap(clinic_pivot, x='weekday', y='clinic_name', z='count', color_continuous_scale='Blues',
                                     category_orders={'weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    st.plotly_chart(fig_clinic, use_container_width=True)
else:
    st.caption("ℹ️ ไม่พบคอลัมน์ clinic_name ในข้อมูล — ข้ามส่วนวิเคราะห์ความหนาแน่นคลินิก")

# --- ตารางสรุปกลุ่มเป้าหมาย ---
st.markdown("##### 📋 ตารางสรุปกลุ่มเป้าหมาย")
target_table = df_f.groupby(['disease_group', 'bp_category'], observed=False).agg(
    {'patient_id': 'nunique', 'monthly_visit_count': 'mean'}
).rename(columns={'patient_id': 'unique_patients', 'monthly_visit_count': 'avg_visits'})
st.dataframe(target_table, use_container_width=True)

# ============================================================
# --- Scoring Section (เดิมชื่อ "Prediction") ---
# ============================================================
st.markdown("---")
st.subheader("🎯 ระบบให้คะแนนความสนใจแพ็กเกจ (Rule-based Score, ไม่ใช่การทำนายจากข้อมูลจริง)")
st.info(
    "⚠️ โมเดลนี้เทรนจาก label ที่สร้างขึ้นจากกฎ (ความถี่ > 2 ครั้ง หรือ ความดัน > 130) "
    "ไม่ใช่ข้อมูลการซื้อแพ็กเกจจริง ผลลัพธ์จึงสะท้อน 'กฎการให้คะแนน' ไม่ใช่พฤติกรรมลูกค้าจริง "
    "ควรเชื่อมข้อมูลการซื้อจริงเมื่อมี เพื่อให้ผลลัพธ์มีความหมายในการตัดสินใจทางธุรกิจ"
    + (f" (AUC บนชุดทดสอบ: {auc_score:.2f})" if auc_score is not None else "")
)

p_id = st.selectbox("เลือก ID ผู้ป่วยเพื่อให้คะแนน:", df_model.index)
p_data = df_model.loc[[p_id]]
prob = model.predict_proba(p_data[X_cols])[0][1]

col_p1, col_p2 = st.columns([1, 2])
col_p1.metric("คะแนนความสนใจ (ตามกฎ)", f"{prob*100:.1f}%")
if prob > 0.5:
    col_p2.success("แนวโน้ม: เข้าเกณฑ์กลุ่มที่ควรติดต่อทีมขาย")
else:
    col_p2.info("แนวโน้ม: กลุ่มลูกค้าทั่วไป ติดตามผลตามปกติ")

st.subheader("💡 ปัจจัยที่มีผลต่อคะแนน")
feat_imp = pd.DataFrame({'Feature': X_cols, 'Importance': model.feature_importances_})
fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
st.plotly_chart(fig_imp, use_container_width=True)

