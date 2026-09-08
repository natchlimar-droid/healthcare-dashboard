import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(layout="wide", page_title="Healthcare Propensity Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("visits_cleaned.csv")
    
    # วิธีแก้ Error: ใช้ .str.split แล้วจัดการด้วย to_numeric ทีละคอลัมน์
    # ป้องกันการแปลงทั้ง DataFrame ที่อาจติดค่าว่างหรือรูปแบบผิด
    df['bp_raw'] = df['bp_raw'].fillna('0 / 0')
    split_data = df['bp_raw'].str.split(' / ', expand=True)
    
    # แปลงเป็นตัวเลข ถ้าแปลงไม่ได้ให้เป็น 0
    df['systolic_bp'] = pd.to_numeric(split_data[0], errors='coerce').fillna(0)
    df['diastolic_bp'] = pd.to_numeric(split_data[1], errors='coerce').fillna(0)
    
    df['gender_code'] = df['gender'].map({'ช': 0, 'ญ': 1}).fillna(0.5)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data()

# --- ส่วนโมเดลและ Dashboard ต่อจากเดิม ---
# (วางส่วนที่เหลือที่คุณมีอยู่ได้เลยครับ)
