import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Healthcare Pro Dashboard", layout="wide")

# CSS ปรับแต่ง
st.markdown("""
    <style>
    div.stMetric { background-color: #f0f2f6; padding: 20px; border-radius: 15px; border-left: 5px solid #0068c9; }
    .stPlotlyChart { background-color: white; border-radius: 15px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. โหลดและเตรียมข้อมูล (ต้องทำความสะอาดก่อนรันโมเดล)
@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
    split_data = df['bp_raw'].str.split(' / ', expand=True)
    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce').fillna(df['age_at_visit'].median())
    return df

df = load_data()

# 3. เตรียมโมเดล (ใส่ไว้หลังโหลดข้อมูล)
df_model = df.groupby('patient_id').agg({
    'visit_id': 'count',
    'bmi': 'mean',
    'systolic_bp': 'mean',
    'gender_code': 'first'
}).rename(columns={'visit_id': 'frequency'})
df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)

X = df_model[['frequency', 'bmi', 'systolic_bp', 'gender_code']]
y = df_model['bought_package']
model = RandomForestClassifier(random_state=42).fit(X, y)

# 4. Sidebar - เมนูและเครื่องมือทำนาย
st.sidebar.markdown("## ⚙️ Controls & Prediction")
disease_sel = st.sidebar.multiselect("กลุ่มโรค", df['disease_group'].unique(), default=df['disease_group'].unique())
gender_sel = st.sidebar.multiselect("เพศ", df['gender'].unique(), default=df['gender'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 ระบบทำนายความสนใจ")
p_id = st.sidebar.selectbox("เลือก ID ผู้ป่วยเพื่อทำนาย:", df_model.index)
p_data = df_model.loc[[p_id]]
prob = model.predict_proba(p_data[['frequency', 'bmi', 'systolic_bp', 'gender_code']])[0][1]
st.sidebar.metric("โอกาสสนใจแพ็กเกจ", f"{prob*100:.1f}%")

# 5. Dashboard หลัก
df_f = df[(df['disease_group'].isin(disease_sel)) & (df['gender'].isin(gender_sel))]

st.title("🏥 Healthcare Executive Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนเคสทั้งหมด", f"{len(df_f):,}")
c2.metric("ผู้ป่วยไม่ซ้ำ", f"{df_f['patient_id'].nunique():,}")
c3.metric("อายุเฉลี่ย", f"{df_f['age_at_visit'].mean():.1f} ปี")
c4.metric("ค่า BMI เฉลี่ย", f"{df_f['bmi'].mean():.1f}")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("แนวโน้มการรักษา")
    fig1 = px.area(df_f.groupby(['visit_date', 'disease_group']).size().reset_index(name='counts'), x='visit_date', y='counts', color='disease_group', template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("สัดส่วนเพศ")
    fig2 = px.pie(df_f, names='gender', hole=0.6, color_discrete_sequence=['#ff9999','#66b3ff'])
    st.plotly_chart(fig2, use_container_width=True)
