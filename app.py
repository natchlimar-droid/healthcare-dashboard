import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

st.title("📊 Healthcare Analysis Dashboard")

# โหลดข้อมูล
df = pd.read_csv("visits_cleaned.csv")

# ตัวกรองด้านข้าง
group = st.sidebar.selectbox("เลือกกลุ่มโรค:", ["ทั้งหมด"] + list(df['disease_group'].unique()))

if group != "ทั้งหมด":
    df = df[df['disease_group'] == group]

# แสดงผล
col1, col2 = st.columns(2)
with col1:
    st.metric("จำนวน Visit", len(df))
with col2:
    st.metric("จำนวนผู้ป่วย", df['patient_id'].nunique())

st.dataframe(df)

# กราฟ
fig = px.bar(df['disease_group'].value_counts(), title="จำนวนผู้ป่วยแยกตามกลุ่มโรค")
st.plotly_chart(fig)