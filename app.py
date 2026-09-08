import streamlit as st
import pandas as pd

# ตั้งค่าหน้าจอ
st.set_page_config(layout="wide", page_title="Status Dashboard")

# CSS สำหรับทำ Card Layout
st.markdown("""
    <style>
    .card { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e6e6e6; box-shadow: 2px 2px 5px #f0f0f0; margin-bottom: 20px; }
    .status-box { text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 1. ส่วนแถบ Progress Bar ด้านบน
st.markdown("### 📊 Project Status")
st.progress(67) # แสดง 67% ตามรูป

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Planning", "✅")
c2.metric("Design", "✅")
c3.metric("Development", "67%")
c4.metric("Testing", "⏳")
c5.metric("Launch", "107 Days")

st.markdown("---")

# 2. ส่วน Project Budget และ Overdue Tasks
col1, col2 = st.columns(2)

with col1:
    st.subheader("Project Budget")
    # ใส่กราฟของคุณที่นี่
    st.bar_chart(pd.DataFrame({'Total': [50, 40, 30]}, index=['Budget', 'Used', 'Target']))
    st.write("Remaining: **$8,770** | **8.1% Over Target**")

with col2:
    st.subheader("Overdue Tasks")
    overdue_data = pd.DataFrame({
        "Task": ["Update profile", "Update plan", "Configure", "Setup DB"],
        "Deadline": ["2017-08-15", "2017-08-06", "2017-08-01", "2017-07-18"],
        "Status": ["1 Day", "4 Days", "10 Days", "24 Days"]
    })
    st.table(overdue_data)

# 3. ส่วน Workload และ Upcoming Deadlines
col3, col4 = st.columns(2)

with col3:
    st.subheader("Workload")
    st.bar_chart(pd.DataFrame({'Workload': [67, 55, 48, 45, 30]}, index=['Georg', 'Nancy', 'Richard', 'Kate', 'Paula']))

with col4:
    st.subheader("Upcoming Deadlines")
    deadline_data = pd.DataFrame({
        "Employee": ["Kate", "Georg", "Nancy", "Paula"],
        "Task": ["Twitter", "E-Commerce", "Dev env", "Hire DS"],
        "Deadline": ["2017-08-15", "2017-08-06", "2017-08-01", "2017-07-18"]
    })
    st.table(deadline_data)
