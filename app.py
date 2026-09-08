import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(layout="wide", page_title="Healthcare Propensity Dashboard")

# 1. โหลดและทำความสะอาดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    # แยก Systolic/Diastolic จาก bp_raw
    df[['systolic_bp', 'diastolic_bp']] = df['bp_raw'].str.split(' / ', expand=True).astype(float)
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data()

# 2. สร้าง Feature Engineering & Synthetic Target (สำหรับการทำนาย)
# เราจำลองว่าคนที่มาบ่อย (Freq > 2) หรือมีค่า BP สูง มีแนวโน้มจะซื้อแพ็กเกจ (1)
df_model = df.groupby('patient_id').agg({
    'visit_id': 'count',
    'bmi': 'mean',
    'systolic_bp': 'mean',
    'gender_code': 'first'
}).rename(columns={'visit_id': 'frequency'})

df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)

# 3. เทรนโมเดล (Random Forest)
X = df_model[['frequency', 'bmi', 'systolic_bp', 'gender_code']]
y = df_model['bought_package']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. หน้า Dashboard
st.title("🏥 Patient Propensity Dashboard")

# ส่วนแสดงความแม่นยำโมเดล
with st.expander("📊 ดูรายละเอียดประสิทธิภาพโมเดล (Model Evaluation)"):
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)
    st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")

# ส่วนการทำนายรายบุคคล
st.subheader("🤖 ทำนายโอกาสสนใจแพ็กเกจ")
selected_patient = st.selectbox("เลือก ID ผู้รับบริการ:", df_model.index)

patient_data = df_model.loc[[selected_patient]]
prob = model.predict_proba(patient_data[['frequency', 'bmi', 'systolic_bp', 'gender_code']])[0][1]

st.metric(f"โอกาสที่ผู้รับบริการ {selected_patient} จะสนใจแพ็กเกจ", f"{prob*100:.1f}%")

if prob > 0.6:
    st.success("แนวโน้มสูง: แนะนำเสนอแพ็กเกจตรวจสุขภาพหัวใจ/เบาหวาน")
else:
    st.info("แนวโน้มต่ำ: แนะนำแพ็กเกจสุขภาพทั่วไป")

# กราฟ Feature Importance (จุดขายโปรเจกต์)
st.subheader("💡 ปัจจัยที่มีผลต่อการตัดสินใจซื้อ")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance')
st.plotly_chart(fig, use_container_width=True)
@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    
    # 1. จัดการค่าว่างใน bp_raw ก่อน
    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
    
    # 2. ทำการ split และใช้ errors='coerce' เพื่อให้ค่าที่แปลไม่ได้กลายเป็น NaN แทนที่จะ Error
    split_data = df['bp_raw'].str.split(' / ', expand=True)
    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)
    
    # 3. จัดการข้อมูลอื่นๆ
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    return df
