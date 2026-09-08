import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier


@st.cache_data

def load_data():

    df = pd.read_csv("visits_cleaned.csv")

    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')

    df['month'] = df['visit_date'].dt.to_period('M').astype(str)

    

    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')

    split_data = df['bp_raw'].str.split(' / ', expand=True)

    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)

    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)

    

    df['age_at_visit'] = pd.to_numeric(df['age_at_visit'], errors='coerce').fillna(0)

    df['age_group'] = pd.cut(df['age_at_visit'], bins=[0, 20, 40, 60, 100], labels=['0-20', '21-40', '41-60', '60+'])

    

    df['monthly_visit_count'] = df.groupby(['patient_id', 'month'])['visit_id'].transform('count')

    df['bp_category'] = pd.cut(df['systolic_bp'], bins=[0, 120, 140, 200], labels=['ปกติ', 'เสี่ยง', 'สูง'])

    

    return df


# --- 2. เรียกใช้ฟังก์ชันทันทีหลังประกาศ ---

df = load_data()


# --- 3. ส่วนอื่นๆ ของ Dashboard (ใส่โค้ดกราฟ/โมเดล ต่อจากนี้ได้เลย) ---

st.title("Healthcare Dashboard")

st.dataframe(df.head()) # ทดสอบการแสดงผลข้อมูล


# ตัวอย่างการใช้งานข้อมูลใหม่

st.write(f"จำนวนผู้ป่วยกลุ่มเสี่ยงความดัน: {len(df[df['bp_category'] == 'เสี่ยง'])}")

st.markdown("---")

st.subheader("📊 การวิเคราะห์เชิงลึก (Deep Analytics)")


col1, col2 = st.columns(2)


# 1. กราฟ Heatmap ความถี่การมาตามช่วงอายุ (หาช่วงเวลาขายของ)

with col1:

    st.markdown("##### 🗓️ พฤติกรรมการมาใช้บริการ (รายเดือน)")

    # สร้าง Pivot Table สำหรับ Heatmap

    visit_trend = df.groupby(['month', 'age_group']).size().reset_index(name='count')

    fig_heat = px.density_heatmap(visit_trend, x='month', y='age_group', z='count', 

                                  color_continuous_scale='Blues', title="จำนวนเคสแยกตามกลุ่มอายุรายเดือน")

    st.plotly_chart(fig_heat, use_container_width=True)


# 2. กราฟเปรียบเทียบ BP Category (หากลุ่มเป้าหมายขายแพ็คเกจ)

with col2:

    st.markdown("##### 🩺 กลุ่มเสี่ยงด้านความดันโลหิต")

    bp_summary = df['bp_category'].value_counts().reset_index()

    fig_bar = px.bar(bp_summary, x='bp_category', y='count', color='bp_category',

                     color_discrete_map={'ปกติ': '#90CAF9', 'เสี่ยง': '#FFA726', 'สูง': '#EF5350'},

                     title="จำนวนผู้ป่วยแบ่งตามระดับความเสี่ยงความดัน")

    st.plotly_chart(fig_bar, use_container_width=True)


# 3. กราฟวิเคราะห์ความสัมพันธ์ BMI vs Systolic BP

st.markdown("##### 📈 ความสัมพันธ์ BMI กับความดัน (หาคนกลุ่มเสี่ยงสูง)")

fig_scatter = px.scatter(df, x='bmi', y='systolic_bp', color='bp_category', 

                         size='monthly_visit_count', hover_data=['patient_id'],

                         color_discrete_map={'ปกติ': '#90CAF9', 'เสี่ยง': '#FFA726', 'สูง': '#EF5350'})

st.plotly_chart(fig_scatter, use_container_width=True)


# 4. ตารางสรุปเพื่อการตัดสินใจเสนอขาย

st.markdown("##### 📋 ตารางสรุปกลุ่มเป้าหมาย")

target_table = df.groupby(['disease_group', 'bp_category']).agg({

    'patient_id': 'nunique',

    'monthly_visit_count': 'mean'

}).rename(columns={'patient_id': 'unique_patients', 'monthly_visit_count': 'avg_visits'})

st.dataframe(target_table, use_container_width=True)

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Healthcare Pro Dashboard", layout="wide")

# CSS สำหรับ Theme สีน้ำเงิน-ฟ้า-ขาว
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1E88E5; }
    .stPlotlyChart { background-color: white; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0D47A1; }
    </style>
    """, unsafe_allow_html=True)

# 2. โหลดและเตรียมข้อมูล (ต้องมาก่อน)
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

# 3. เตรียมโมเดล
df_model = df.groupby('patient_id').agg({'visit_id': 'count', 'bmi': 'mean', 'systolic_bp': 'mean', 'gender_code': 'first'}).rename(columns={'visit_id': 'frequency'})
df_model['bought_package'] = ((df_model['frequency'] > 2) | (df_model['systolic_bp'] > 130)).astype(int)
X = df_model[['frequency', 'bmi', 'systolic_bp', 'gender_code']]
y = df_model['bought_package']
model = RandomForestClassifier(random_state=42).fit(X, y)

# 4. Sidebar Controls
st.sidebar.markdown("## ⚙️ Control Panel")
disease_sel = st.sidebar.multiselect("เลือกกลุ่มโรค", df['disease_group'].unique(), default=df['disease_group'].unique())
gender_sel = st.sidebar.multiselect("เลือกเพศ", df['gender'].unique(), default=df['gender'].unique())

# 5. Dashboard หลัก
st.title("🏥 Healthcare Executive Dashboard")
df_f = df[(df['disease_group'].isin(disease_sel)) & (df['gender'].isin(gender_sel))]

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("เคสทั้งหมด", f"{len(df_f):,}")
c2.metric("ผู้ป่วย", f"{df_f['patient_id'].nunique():,}")
c3.metric("อายุเฉลี่ย", f"{df_f['age_at_visit'].mean():.1f} ปี")
c4.metric("BMI เฉลี่ย", f"{df_f['bmi'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# Graph Row
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📈 แนวโน้มการรักษา")
    fig1 = px.area(df_f.groupby(['visit_date', 'disease_group']).size().reset_index(name='counts'), x='visit_date', y='counts', color='disease_group', color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("👥 สัดส่วนเพศ")
    fig2 = px.pie(df_f, names='gender', hole=0.6, color_discrete_sequence=['#90CAF9', '#1565C0'])
    st.plotly_chart(fig2, use_container_width=True)

# Prediction Section
st.markdown("---")
st.subheader("🤖 ระบบทำนายความสนใจแพ็กเกจ")
p_id = st.selectbox("เลือก ID ผู้ป่วยเพื่อดูการทำนาย:", df_model.index)
p_data = df_model.loc[[p_id]]
prob = model.predict_proba(p_data[['frequency', 'bmi', 'systolic_bp', 'gender_code']])[0][1]

col_pred1, col_pred2 = st.columns([1, 2])
with col_pred1:
    st.metric("โอกาสสนใจแพ็กเกจ", f"{prob*100:.1f}%")
with col_pred2:
    if prob > 0.5: st.success("แนวโน้ม: สนใจแพ็กเกจพิเศษ แนะนำให้ติดต่อทีมขาย")
    else: st.info("แนวโน้ม: กลุ่มลูกค้าทั่วไป ติดตามผลตามปกติ")

st.subheader("💡 ปัจจัยที่มีผลต่อการตัดสินใจ")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
st.plotly_chart(fig_imp, use_container_width=True)
