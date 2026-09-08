import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Healthcare Pro Dashboard", layout="wide")

# CSS ปรับแต่งให้ดูสะอาดและทันสมัย
st.markdown("""
    <style>
    div.stMetric { background-color: #f0f2f6; padding: 20px; border-radius: 15px; border-left: 5px solid #0068c9; }
    .stPlotlyChart { background-color: white; border-radius: 15px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# โหลดข้อมูล
df = pd.read_csv("visits_cleaned.csv")

# Sidebar - ออกแบบให้เป็นเหมือนเมนูควบคุม
st.sidebar.markdown("## ⚙️ Filters")
disease_sel = st.sidebar.multiselect("กลุ่มโรค", df['disease_group'].unique(), default=df['disease_group'].unique())
gender_sel = st.sidebar.multiselect("เพศ", df['gender'].unique(), default=df['gender'].unique())

df_f = df[(df['disease_group'].isin(disease_sel)) & (df['gender'].isin(gender_sel))]

# Header
st.title("🏥 Healthcare Executive Dashboard")
st.markdown("ภาพรวมข้อมูลการรักษาและกลุ่มผู้ป่วย")

# Metrics (KPI Cards)
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนเคสทั้งหมด", f"{len(df_f):,}")
c2.metric("ผู้ป่วยไม่ซ้ำ (Unique)", f"{df_f['patient_id'].nunique():,}")
c3.metric("อายุเฉลี่ย", f"{df_f['age_at_visit'].mean():.1f} ปี")
c4.metric("ค่า BMI เฉลี่ย", f"{df_f['bmi'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# กราฟแถวที่ 1
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("แนวโน้มการรักษาแยกตามกลุ่มโรค")
    fig1 = px.area(df_f.groupby(['visit_date', 'disease_group']).size().reset_index(name='counts'), 
                   x='visit_date', y='counts', color='disease_group', template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("สัดส่วนเพศ")
    fig2 = px.pie(df_f, names='gender', hole=0.6, color_discrete_sequence=['#ff9999','#66b3ff'])
    st.plotly_chart(fig2, use_container_width=True)

# กราฟแถวที่ 2
col3, col4 = st.columns(2)
with col3:
    st.subheader("ความดันโลหิต (Systolic vs Diastolic)")
    fig3 = px.scatter(df_f, x='systolic_bp', y='diastolic_bp', color='disease_group', opacity=0.6)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Top 5 Clinic ที่มีผู้ใช้บริการสูงสุด")
    top_clinics = df_f['clinic_name'].value_counts().head(5).reset_index()
    fig4 = px.bar(top_clinics, x='count', y='clinic_name', orientation='h', color='count', color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)
