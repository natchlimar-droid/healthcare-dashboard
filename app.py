import streamlit as st
import pandas as pd
import plotly.express as px

# 1. ปรับแต่งหน้าเว็บ
st.set_page_config(page_title="Healthcare Analytics", layout="wide", page_icon="🏥")

# ใส่ CSS เพื่อปรับดีไซน์
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. โหลดข้อมูล
@st.cache_data
def load_data():
    return pd.read_csv("visits_cleaned.csv")

df = load_data()

# 3. Sidebar
st.sidebar.header("🔍 ตัวกรองข้อมูล")
disease_filter = st.sidebar.multiselect("เลือกกลุ่มโรค:", options=df['disease_group'].unique(), default=df['disease_group'].unique())
gender_filter = st.sidebar.multiselect("เลือกเพศ:", options=df['gender'].unique(), default=df['gender'].unique())

# กรองข้อมูล
df_filtered = df[(df['disease_group'].isin(disease_filter)) & (df['gender'].isin(gender_filter))]

# 4. Header & Metrics
st.title("🏥 Healthcare Management Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("จำนวน Visit ทั้งหมด", len(df_filtered))
col2.metric("จำนวนผู้ป่วยไม่ซ้ำ", df_filtered['patient_id'].nunique())
col3.metric("ค่าเฉลี่ย BMI", round(df_filtered['bmi'].mean(), 2))

# 5. กราฟ
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig1 = px.pie(df_filtered, names='disease_group', title="สัดส่วนกลุ่มโรค", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    fig2 = px.histogram(df_filtered, x='age_at_visit', nbins=20, title="กระจายตัวของอายุผู้ป่วย", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig2, use_container_width=True)

# 6. ตาราง
st.subheader("📋 รายการข้อมูลรายละเอียด")
st.dataframe(df_filtered, use_container_width=True)
