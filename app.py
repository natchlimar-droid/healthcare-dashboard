import streamlit as st
import pandas as pd
import plotly.express as px

# 1. ตั้งค่าหน้าเว็บแบบ Wide เพื่อให้จัด Layout ได้เหมือนในรูป
st.set_page_config(layout="wide", page_title="Clean Dashboard")

# 2. CSS สำหรับทำ Sidebar สีเข้ม และ Card สไตล์ Clean
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #5b6e8a; color: white; }
    .card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    h1, h2 { color: #5b6e8a; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar (เมนูซ้ายมือ)
with st.sidebar:
    st.title("DASHBOARD")
    st.write("---")
    st.write("🏠 Home")
    st.write("📊 Charts")
    st.write("⭐ Favorites")
    st.write("💬 Chat")
    st.write("⚙️ Setting")
    st.write("❓ Help")

# 4. โหลดข้อมูล (ใช้ไฟล์ของคุณ)
df = pd.read_csv("visits_cleaned.csv")

# 5. Header (Search Bar จำลอง)
st.text_input("🔍", placeholder="Search...")

# 6. กราฟใหญ่ด้านบน (Detailed Chart 01)
st.subheader("Detailed Chart 01")
fig_line = px.line(df, x='visit_date', y='systolic_bp', markers=True)
fig_line.update_layout(template="simple_white")
st.plotly_chart(fig_line, use_container_width=True)

# 7. Card 3 ช่อง (Earnings, Downloads, Favorites)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Visits", len(df))
    st.area_chart(df.head(10)['age_at_visit'])
with c2:
    st.metric("Patients", df['patient_id'].nunique())
    st.area_chart(df.head(10)['bmi'])
with c3:
    st.metric("Clinic Count", df['clinic_name'].nunique())
    st.area_chart(df.head(10)['systolic_bp'])

# 8. กราฟแท่งและข่าว (ล่าง)
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Detailed Chart 02")
    fig_bar = px.bar(df.head(10), x='visit_date', y='age_at_visit', color='disease_group')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Recently News")
    for i in range(3):
        st.write(f"**Update {i+1}**: ข้อมูลล่าสุดประจำวันที่ {df['visit_date'].iloc[i]}")
        st.caption("Lorem ipsum dolor sit amet...")
