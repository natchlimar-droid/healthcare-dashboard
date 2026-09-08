import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ตั้งค่าหน้าจอ
st.set_page_config(layout="wide", page_title="Healthcare Dashboard")

# 1. โหลดและเตรียมข้อมูล (แบบปลอดภัย)
@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
    split_data = df['bp_raw'].str.split(' / ', expand=True)
    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data()

# 2. เตรียมข้อมูลสำหรับโมเดล
df_model = df.groupby('patient_id').agg({
    'visit_id': 'count',
    'bmi': 'mean',
    'systolic_bp': 'mean',
    'gender_code': 'first'
}).rename(columns={'visit_id': 'frequency'})

df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)

# 3. เทรนโมเดล
X = df_model[['frequency', 'bmi', 'systolic_bp', 'gender_code']]
y = df_model['bought_package']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 4. แสดงผล Dashboard
st.title("🏥 Healthcare Executive Dashboard")

# แถวบน: KPI
c1, c2, c3 = st.columns(3)
c1.metric("จำนวนผู้ป่วยทั้งหมด", f"{len(df_model):,}")
c2.metric("โอกาสซื้อแพ็กเกจเฉลี่ย", f"{df_model['bought_package'].mean()*100:.1f}%")
c3.metric("ความดัน Systolic เฉลี่ย", f"{df_model['systolic_bp'].mean():.1f}")

st.markdown("---")

# แถวกลาง: ทำนายรายบุคคล
st.subheader("🤖 ระบบทำนายความสนใจแพ็กเกจ")
p_id = st.selectbox("เลือก ID ผู้ป่วย:", df_model.index)
p_data = df_model.loc[[p_id]]
prob = model.predict_proba(p_data[['frequency', 'bmi', 'systolic_bp', 'gender_code']])[0][1]

st.metric(f"ความน่าจะเป็นที่ {p_id} จะซื้อแพ็กเกจ", f"{prob*100:.1f}%")

if prob > 0.5:
    st.success("แนวโน้ม: สนใจแพ็กเกจพิเศษ")
else:
    st.info("แนวโน้ม: กลุ่มลูกค้าทั่วไป")

# แถวท้าย: กราฟสำคัญ
st.subheader("💡 ปัจจัยที่มีผลต่อการตัดสินใจ")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance')
st.plotly_chart(fig, use_container_width=True)
