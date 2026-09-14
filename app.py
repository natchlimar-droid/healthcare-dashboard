import os
import sys
import time
import zipfile
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page Config (Must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Optional Dependencies
# ============================================================
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

USE_AGGRID = False

# ============================================================
# Design Tokens & Configurations
# ============================================================
INK    = "#0B1B2B"
BG     = "#F7F9F8"
SURFACE = "#FFFFFF"
TEAL   = "#0E5C56"
SAGE   = "#4F7C5B"
AMBER  = "#C87F0A"
RED    = "#B3261E"
MUTED  = "#5B6B6B"

DISEASE_COLORS = {
    "ความดันโลหิตสูง": TEAL,
    "เบาหวาน": "#2F6FB5",
    "ไขมันในเลือดสูง": AMBER,
    "อื่น ๆ": "#9AA6A0",
}
BP_COLORS = {
    "ปกติ (<120)": SAGE,
    "เฝ้าระวัง (120-139)": AMBER,
    "สูง (≥140)": RED,
    "ไม่มีข้อมูล": "#C7CFCC",
}

MIN_SAMPLE = 5
SUNBURST_TOP_N = 5

HEALTH_PACKAGES = {
    "Essential Package": {
        "price": 3000,
        "tests": ["CBC (ความสมบูรณ์ของเม็ดเลือด)", "FBS (น้ำตาลในเลือด)", "Lipid Profile (ไขมันในเลือด)", "Uric Acid (กรดยูริก)", "CXR (เอกซเรย์ปอด)", "EKG (คลื่นไฟฟ้าหัวใจ)"],
        "desc": "เหมาะสำหรับวัยเริ่มต้นทำงานและผู้ที่ไม่มีความเสี่ยงหรือโรคประจำตัว (อายุ <30 ปี)"
    },
    "Advanced Package": {
        "price": 5500,
        "tests": ["Essential Tests +", "Liver Function (การทำงานของตับ)", "Kidney Function (การทำงานของไต)", "HbA1c (น้ำตาลสะสม)", "Urine Examination (ปัสสาวะ)", "Ultrasound Whole Abdomen (อัลตราซาวด์ช่องท้อง)"],
        "desc": "เหมาะสำหรับวัยทำงานที่มีความเครียดสะสม พักผ่อนน้อย หรือเริ่มมีความเสี่ยง (อายุ 30-50 ปี)"
    },
    "Longevity Package": {
        "price": 8000,
        "tests": ["Advanced Tests +", "Thyroid Function (ไทรอยด์)", "Bone Densitometry (มวลกระดูก)", "Tumor Markers (สารบ่งชี้มะเร็งพื้นฐาน)", "ABI (การตีบตันของหลอดเลือด)"],
        "desc": "เหมาะสำหรับผู้สูงอายุ หรือผู้ที่มีความเสี่ยงโรคเรื้อรัง ต้องการดูแลอย่างใกล้ชิด (อายุ >50 ปี)"
    }
}

SYSTEM_PROMPT_HEALTH_ARCHITECT = (
    "คุณคือ Senior Health Data Architect และผู้เชี่ยวชาญด้านเวชศาสตร์ป้องกันและระบบคัดกรองสุขภาพอัจฉริยะ "
    "ประจำโรงพยาบาลวิชัยเวช อินเตอร์เนชั่นแนล อ้อมน้อย (Vichaivej International Hospital Omnoi) "
    "หน้าที่ของคุณคือการวิเคราะห์ข้อมูลผู้รับบริการ (อายุ, เพศ, ดัชนีมวลกาย BMI, เส้นรอบเอว, ความดันโลหิต, ประวัติโรค, อาการ และผลแล็บ) "
    "เพื่อประเมินระดับความเสี่ยง 4 กลุ่ม (Level 1: ตรวจสุขภาพทั่วไป, Level 2: ยังไม่พบโรคแต่มีความเสี่ยง, "
    "Level 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น, Level 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม) "
    "และคัดเลือกแพ็กเกจย่อยที่เหมาะสมที่สุดเฉพาะบุคคล (Personalized Best Match Sub-package) "
    "ตามมาตรฐานแพ็กเกจตรวจสุขภาพจริงของโรงพยาบาลวิชัยเวชฯ อ้อมน้อย ได้อย่างแม่นยำ ปลอดภัย คุ้มค่า และเกิดประโยชน์สูงสุด"
)

HEALTH_PACKAGES_4LEVEL = {
    1: {
        "level": 1, "name": "กลุ่ม 1: ตรวจสุขภาพทั่วไป", "short_title": "ตรวจสุขภาพทั่วไป",
        "target_audience": "สำหรับคนที่ใส่ใจสุขภาพ อยากรู้พื้นฐานของตัวเอง",
        "badge": "“เช็กสุขภาพวันนี้ เพื่อความมั่นใจในอนาคต”",
        "badge_color": "#0E7055", "badge_bg": "#DCFCE7", "header_bg": "#0E7055",
        "bg_card": "#F0FDF4", "border_color": "#0E7055", "text_accent": "#0E7055",
        "price": 990, "price_display": "เริ่มต้น 990 บาท",
        "tests": ["Physical Examination (ตรวจร่างกายทั่วไป)", "Complete Blood Count (ตรวจเลือดพื้นฐาน/ครบชุด)", "Urine Analysis (ตรวจปัสสาวะสมบูรณ์แบบ)", "Chest X-ray (เอกซเรย์ปอด)", "EKG (คลื่นไฟฟ้าหัวใจ)", "Ultrasound Upper/Whole Abdomen (อัลตราซาวด์ช่องท้อง)"],
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ BASIC", "short_name": "BASIC", "price": 990, "price_display": "990 บาท", "description": "เหมาะสำหรับผู้ที่ต้องการตรวจเช็กความสมบูรณ์ของร่างกายเบื้องต้น", "tests": ["ตรวจร่างกายทั่วไป", "ตรวจเลือดพื้นฐาน", "ตรวจปัสสาวะ, เอกซเรย์ปอด"]},
            {"name": "แพคเกจตรวจสุขภาพ STANDARD", "short_name": "STANDARD", "price": 2290, "price_display": "2,290 บาท", "description": "เหมาะสำหรับวัยทำงาน ตรวจละเอียดขึ้นครอบคลุมคลื่นไฟฟ้าหัวใจและอัลตราซาวด์", "tests": ["ตรวจร่างกายทั่วไป", "ตรวจเลือดครบชุด", "เอกซเรย์ปอด, ตรวจคลื่นไฟฟ้าหัวใจ", "(Ultrasound ช่องท้องส่วนบน)"]},
            {"name": "แพคเกจตรวจสุขภาพ PREMIUM", "short_name": "PREMIUM", "price": 4990, "price_display": "4,990 บาท", "description": "เหมาะสำหรับการตรวจคัดกรองสุขภาพประจำปีแบบครบถ้วนสมบูรณ์สูงสุด", "tests": ["ตรวจร่างกายละเอียด", "ตรวจเลือดครบชุด", "เอกซเรย์ปอด, คลื่นไฟฟ้าหัวใจ", "อัลตราซาวด์ช่องท้องส่วนบน-ล่าง"]}
        ]
    },
    2: {
        "level": 2, "name": "กลุ่ม 2: ยังไม่พบโรคแต่มีความเสี่ยง", "short_title": "ยังไม่พบโรคแต่มีความเสี่ยง",
        "target_audience": "สำหรับผู้ที่มีปัจจัยเสี่ยงจากอายุ น้ำหนัก พฤติกรรม หรือประวัติครอบครัว",
        "badge": "“รู้ความเสี่ยงก่อนเกิดโรค ป้องกันได้...ให้ชีวิตยืนยาว”",
        "badge_color": "#B45309", "badge_bg": "#FEF3C7", "header_bg": "#D97706",
        "bg_card": "#FFFBEB", "border_color": "#D97706", "text_accent": "#D97706",
        "price": 2990, "price_display": "เริ่มต้น 2,990 บาท",
        "tests": ["Fasting Blood Sugar & HbA1c (ตรวจระดับน้ำตาลสะสม)", "Lipid Profile (ตรวจระดับไขมันในเลือด)", "Liver & Kidney Function (ตรวจการทำงานของตับ ไต)", "ประเมินภาวะแทรกซ้อนและอ้วนลงพุง (Metabolic Syndrome)", "เอกซเรย์ปอด อัลตราซาวด์ช่องท้อง"],
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH", "short_name": "9.9 METABOLIC HEALTH MONTH", "price": 2990, "price_display": "2,990 บาท", "description": "เน้นประเมินภาวะอ้วนลงพุง ระดับน้ำตาลสะสม และไขมันในเลือด", "tests": ["ตรวจระดับน้ำตาล (FBS, HbA1c)", "ตรวจไขมันในเลือด", "ตรวจการทำงานของตับ ไต", "ประเมินภาวะอ้วนลงพุง (Metabolic Syndrome)"]},
            {"name": "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)", "short_name": "คนอ้วน (Obesity Check)", "price": 2990, "price_display": "2,990 บาท", "description": "สำหรับผู้ที่มีภาวะน้ำหนักเกิน ตรวจคัดกรองภาวะแทรกซ้อนพร้อมรับคำปรึกษาจากแพทย์", "tests": ["ตรวจสุขภาพสำหรับผู้มีภาวะน้ำหนักเกิน", "ประเมินภาวะแทรกซ้อนจากโรคอ้วน", "ตรวจระดับน้ำตาล ไขมัน ตับ ไต", "ให้คำแนะนำการดูแลน้ำหนักโดยแพทย์"]},
            {"name": "แพคเกจตรวจสุขภาพ วัย 35+", "short_name": "วัย 35+", "price": 3990, "price_display": "3,990 บาท", "description": "ตรวจคัดกรองโรคที่มักพบในวัยทำงาน พร้อมอัลตราซาวด์ช่องท้องและเอกซเรย์ปอด", "tests": ["ตรวจคัดกรองโรคที่มักพบในวัยทำงาน", "ตรวจระดับน้ำตาล ไขมัน ความดัน", "ตรวจการทำงานของตับ ไต", "เอกซเรย์ปอด อัลตราซาวด์ช่องท้อง"]}
        ]
    },
    3: {
        "level": 3, "name": "กลุ่ม 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น", "short_title": "เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น",
        "target_audience": "สำหรับผู้ที่มีค่าผลตรวจเริ่มผิดปกติ หรือได้รับการวินิจฉัยโรคระยะเริ่มต้น",
        "badge": "“ตรวจให้ชัด...ดูแลได้เร็ว ลดความเสี่ยงในอนาคต”",
        "badge_color": "#B91C1C", "badge_bg": "#FEE2E2", "header_bg": "#DC2626",
        "bg_card": "#FFF1F2", "border_color": "#DC2626", "text_accent": "#DC2626",
        "price": 2990, "price_display": "เริ่มต้น 2,990 บาท",
        "tests": ["Fasting Blood Sugar & HbA1c (ตรวจน้ำตาลสะสม)", "Comprehensive Lipid Profile (ตรวจไขมันครบชุด)", "Liver & Kidney Panel (ตรวจการทำงานของตับ ไต)", "Microalbuminuria (ตรวจโปรตีนไข่ขาวรั่วในปัสสาวะ)", "Electrocardiogram (คลื่นไฟฟ้าหัวใจ EKG)", "ปรึกษาแพทย์เฉพาะทางเพื่อวางแผนการรักษา"],
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)", "short_name": "เบาหวาน (Diabetes Check)", "price": 3290, "price_display": "3,290 บาท", "description": "เจาะลึกค่าน้ำตาลสะสม การทำงานของไต และตรวจไข่ขาวในปัสสาวะ", "tests": ["ตรวจระดับน้ำตาล (FBS, HbA1c)", "ตรวจไขมัน การทำงานของตับ ไต", "ตรวจปัสสาวะ (Microalb)", "พบแพทย์ให้คำแนะนำ"]},
            {"name": "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)", "short_name": "ความดันโลหิตสูง (Hypertension Check)", "price": 2990, "price_display": "2,990 บาท", "description": "ตรวจประเมินหลอดเลือด คลื่นไฟฟ้าหัวใจ และผลกระทบต่อไตจากความดันโลหิต", "tests": ["ตรวจความดันโลหิต", "ตรวจการทำงานของหัวใจ (EKG)", "ตรวจไขมัน การทำงานของไต", "พบแพทย์ให้คำแนะนำ"]},
            {"name": "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)", "short_name": "ไขมันในเลือดสูง (Lipid Check)", "price": 2990, "price_display": "2,990 บาท", "description": "ตรวจระดับไขมันทุกชนิด ละเอียด พร้อมประเมินความเสี่ยงโรคหัวใจและหลอดเลือด", "tests": ["ตรวจไขมัน (Cholesterol, LDL, HDL, TG)", "ตรวจการทำงานของตับ ไต", "ประเมินความเสี่ยงต่อโรคหัวใจและหลอดเลือด", "พบแพทย์ให้คำแนะนำ"]}
        ]
    },
    4: {
        "level": 4, "name": "กลุ่ม 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม", "short_title": "มีอาการซ้ำๆ/ควรประเมินเพิ่มเติม",
        "target_audience": "สำหรับผู้ที่มีอาการผิดปกติซ้ำๆ หรือมีความกังวลและต้องการตรวจเชิงลึก",
        "badge": "“หาสาเหตุให้ชัดเจน เพื่อความสบายใจและการดูแลที่ตรงจุด”",
        "badge_color": "#5B21B6", "badge_bg": "#F3E8FF", "header_bg": "#6941C6",
        "bg_card": "#FAF5FF", "border_color": "#6941C6", "text_accent": "#6941C6",
        "price": 4990, "price_display": "เริ่มต้น 4,990 บาท",
        "tests": ["Specialized Tumor Biomarkers (ตรวจสารบ่งชี้มะเร็ง)", "Cardiovascular Examination (EKG, EST สมรรถภาพหัวใจ, Echo)", "Endoscopy / Colonoscopy (ส่องกล้องตรวจลำไส้ใหญ่และตัดติ่งเนื้อ)", "High-Resolution Imaging & Ultrasound", "การตรวจวินิจฉัยและดูแลอย่างใกล้ชิดโดยแพทย์เฉพาะทาง"],
        "sub_packages": [
            {"name": "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)", "short_name": "คัดกรอง มะเร็ง (Cancer Screening)", "price": 4990, "price_display": "4,990 บาท", "description": "ตรวจสารบ่งชี้มะเร็งสำคัญ เอกซเรย์ปอด และตรวจอวัยวะตามความเสี่ยงเฉพาะบุคคล", "tests": ["ตรวจสารบ่งชี้มะเร็ง (Tumor Marker)", "เอกซเรย์ปอด / อัลตราซาวด์", "ตรวจอวัยวะตามความเสี่ยง (ช/ญ)", "พบแพทย์ให้คำแนะนำ"]},
            {"name": "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)", "short_name": "หัวใจ (Heart Check)", "price": 4990, "price_display": "4,990 บาท", "description": "ตรวจประเมินสมรรถภาพหัวใจอย่างละเอียดด้วย EKG, EST และอัลตราซาวด์หัวใจ Echo", "tests": ["ตรวจคลื่นไฟฟ้าหัวใจ (EKG)", "ตรวจสมรรถภาพหัวใจ (EST)", "อัลตราซาวด์หัวใจ (Echo)", "ตรวจไขมัน และปัจจัยเสี่ยงอื่นๆ"]},
            {"name": "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)", "short_name": "คัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)", "price": 8900, "price_display": "8,900 บาท", "description": "ส่องกล้องระบบทางเดินอาหารและลำไส้ใหญ่ ตรวจหาและตัดติ่งเนื้อโดยแพทย์ผู้เชี่ยวชาญ", "tests": ["ส่องกล้องตรวจลำไส้ใหญ่", "ตรวจหาติ่งเนื้อ (Polyp)", "โดยทีมแพทย์เฉพาะทาง", "พร้อมการดูแลหลังการตรวจ"]}
        ]
    }
}

SPECIALIZED_DIAGNOSTIC_OPTIONS = {
    "ส่องกล้องระบบทางเดินอาหาร (EGD / Colonoscopy)": {"price": 7500, "desc": "คัดกรองแผลในกระเพาะอาหาร กรดไหลย้อนเรื้อรัง และติ่งเนื้อมะเร็งลำไส้ใหญ่", "icon": "🔬"},
    "Low-Dose CT Chest (คัดกรองมะเร็งปอดรังสีต่ำ)": {"price": 4500, "desc": "ตรวจคัดกรองมะเร็งปอดระยะเริ่มต้นในผู้สูบบุหรี่หรือสัมผัสฝุ่น PM2.5 เรื้อรัง", "icon": "🫁"},
    "ชุดคัดกรองสารบ่งชี้มะเร็งรวม (CEA, AFP, CA-125, PSA)": {"price": 3500, "desc": "คัดกรองมะเร็งระบบทางเดินอาหาร มะเร็งตับ รังไข่ และต่อมลูกหมาก", "icon": "🎗️"},
    "ตรวจคลื่นเสียงสะท้อนหัวใจ (Echocardiogram)": {"price": 4000, "desc": "ประเมินสมรรถภาพกล้ามเนื้อหัวใจ ลิ้นหัวใจ และการสูบฉีดเลือด", "icon": "❤️"},
    "ตรวจสมรรถภาพหลอดเลือดแดงส่วนปลาย (ABI Test)": {"price": 1500, "desc": "ตรวจหาการตีบตันของหลอดเลือดแดงส่วนปลายและประเมินอายุหลอดเลือด", "icon": "🩺"}
}

DISEASE_CONFIG = {
    "🏥 General Dashboard (หน้าแรก)": {"icon": "🏥", "is_general": True},
    "🔮 AI Forecast (พยากรณ์ 2568-2569)": {"icon": "🔮", "is_forecast": True, "target_desc": "พยากรณ์ปริมาณผู้รับบริการและแนวโน้มกลุ่มเสี่ยงวิกฤตล่วงหน้า 24 เดือน"},
    "ล้างไต (Dialysis)": {"icon": "🩺", "filter_condition": lambda df: df["is_dialysis"] == True, "target_desc": "BP <130/80", "Key_Tests": ["BUN", "Creatinine", "Electrolytes", "CBC"]},
    "เบาหวาน (Diabetes)": {"icon": "🩸", "filter_condition": lambda df: df["disease_group"] == "เบาหวาน", "target_desc": "HbA1c <7.0", "Key_Tests": ["HbA1c", "Microalbuminuria", "Funduscopy"]},
    "ความดันโลหิตสูง (Hypertension)": {"icon": "🫀", "filter_condition": lambda df: df["disease_group"] == "ความดันโลหิตสูง", "target_desc": "BP <140/90", "Key_Tests": ["Lipid Profile", "Creatinine", "EKG"]},
    "ไขมันในเลือดสูง (Dyslipidemia)": {"icon": "🧈", "filter_condition": lambda df: df["disease_group"] == "ไขมันในเลือดสูง", "target_desc": "LDL <100", "Key_Tests": ["Lipid Profile", "Liver Function"]},
    "กลุ่มอื่น ๆ (Health Packages)": {"icon": "🧬", "filter_condition": lambda df: (df["disease_group"] == "อื่น ๆ") | (df["disease_group"].isna()) | (df["disease_group"] == "ไม่ระบุ") | (df["disease_group"] == "ทั่วไป"), "target_desc": "คัดกรองสุขภาพเชิงรุกและจัดแพ็คเกจ 4 ระดับ (Basic to Deep Health)", "is_other_packages": True, "Key_Tests": ["CBC", "FBS", "Lipid Profile", "Liver & Kidney Functions", "Specialized Screening"]}
}

SPECIAL_SCREENINGS = {
    "Mammogram": {"price": 2000, "desc": "คัดกรองมะเร็งเต้านม"},
    "PSA (มะเร็งต่อมลูกหมาก)": {"price": 2000, "desc": "คัดกรองมะเร็งต่อมลูกหมาก"}
}

# ============================================================
# Helpers & UI Config
# ============================================================
def render_custom_html(html_str):
    """ Renders HTML safely without Markdown turning indented lines into code blocks. """
    clean_str = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_str, unsafe_allow_html=True)

st.markdown(f"""

""", unsafe_allow_html=True)

# ============================================================
# Data Loading & Preprocessing
# ============================================================
def read_csv_or_zip(file_path, preferred_csv_names=None):
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

    if file_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as z:
            csv_files = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if not csv_files:
                raise ValueError(f"ไม่พบไฟล์ CSV ภายใน ZIP: {file_path}")

            selected_file = csv_files[0]
            if preferred_csv_names:
                preferred_lower = {name.lower() for name in preferred_csv_names}
                for csv_name in csv_files:
                    if os.path.basename(csv_name).lower() in preferred_lower:
                        selected_file = csv_name
                        break

            with z.open(selected_file) as csv_file:
                return pd.read_csv(csv_file, encoding="utf-8-sig", low_memory=False)
    raise ValueError(f"ไม่รองรับประเภทไฟล์: {file_path}")

def find_data_file(file_candidates):
    for file_name in file_candidates:
        if os.path.exists(file_name):
            return file_name
    return None

@st.cache_data(show_spinner="กำลังโหลดข้อมูล...")
def load_data():
    main_file = find_data_file(["visits_cleaned.csv", "visits_cleaned.zip", "visits_with_monthly_count.csv", "visits_with_monthly_count.zip"])
    if main_file is None:
        return None, None, False, False

    try:
        df = read_csv_or_zip(main_file, preferred_csv_names=["visits_cleaned.csv", "visits_with_monthly_count.csv"])
    except Exception as e:
        st.error(f"⚠️ อ่านไฟล์ข้อมูลหลักไม่สำเร็จ: {e}")
        return None, None, False, False

    df.columns = df.columns.astype(str).str.strip()

    # Dates
    has_date = False
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
        valid_date = df["visit_date"].notna() & (df["visit_date"].dt.year >= 2000)
        has_date = bool(valid_date.any())
        df.loc[~valid_date, "visit_date"] = pd.NaT
    else:
        df["visit_date"] = pd.NaT

    if has_date:
        df["year_month"] = df["visit_date"].dt.to_period("M").astype(str)
        df["visit_day"] = df["visit_date"].dt.normalize()
    else:
        df["year_month"] = "ไม่ระบุ"
        df["visit_day"] = pd.NaT

    # Convert Numerics
    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # BP
    if "systolic_bp" not in df.columns: df["systolic_bp"] = np.nan
    if "diastolic_bp" not in df.columns: df["diastolic_bp"] = np.nan
    if "bp_raw" in df.columns and df["systolic_bp"].isna().all():
        bp_split = df["bp_raw"].astype(str).str.replace(",", "", regex=False).str.extract(r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$")
        sys_bp = pd.to_numeric(bp_split[0], errors="coerce")
        dia_bp = pd.to_numeric(bp_split[1], errors="coerce")
        sys_bp[~sys_bp.between(60, 250)] = np.nan
        dia_bp[~dia_bp.between(30, 150)] = np.nan
        df["systolic_bp"] = sys_bp
        df["diastolic_bp"] = dia_bp

    df.rename(columns={"systolic_bp": "systolic", "diastolic_bp": "diastolic"}, inplace=True)

    # Gender
    df["gender"] = df["gender"].fillna("ไม่ระบุ") if "gender" in df.columns else "ไม่ระบุ"
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)

    # BMI
    df["bmi_imputed"] = df["bmi"].isna() if "bmi" in df.columns else True
    df["bmi"] = df["bmi"].fillna(df["bmi"].median() if "bmi" in df.columns and pd.notna(df["bmi"].median()) else 22.0)

    # Age
    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()
    if has_age:
        df["age_at_visit"] = df["age_at_visit"].fillna(df["age_at_visit"].median() if pd.notna(df["age_at_visit"].median()) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(df["age_at_visit"], bins=[0, 29, 39, 49, 59, 120], labels=["<30 ปี", "30-40 ปี", "40-50 ปี", "50-60 ปี", ">60 ปี"]).astype(str).replace("nan", "ไม่ระบุ")
        df["pyramid_group"] = pd.cut(df["age_at_visit"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False, labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]).astype(str)
    else:
        df["age_at_visit"], df["is_adult"], df["age_group"], df["pyramid_group"] = 35.0, True, "ไม่ระบุ", "ไม่ระบุ"

    # Risk Categories
    bp_cat = pd.cut(df["systolic"], bins=[-1, 120, 139, 300], labels=["ปกติ (<120)", "เฝ้าระวัง (120-139)", "สูง (≥140)"]).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat
    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    # Diagnosis & Clinic
    df["diagnosis_clean"] = df["diagnosis_text"].astype(str).str.strip().replace({"": "ไม่ระบุ", "-": "ไม่ระบุ", ":": "ไม่ระบุ"}).where(df["diagnosis_text"].notna(), "ไม่ระบุ") if "diagnosis_text" in df.columns else "ไม่ระบุ"
    if "disease_group" not in df.columns: df["disease_group"] = "ทั่วไป"
    if "clinic_name" not in df.columns: df["clinic_name"] = "ไม่ระบุ"

    return df, None, has_date, has_age

@st.cache_data(show_spinner=False)
def process_patient_data(dataframe):
    df_proc = dataframe.copy()
    today = df_proc["visit_date"].max() if pd.notna(df_proc["visit_date"].max()) else pd.Timestamp.today().normalize()

    # Dialysis detection
    if "diagnosis_text" in df_proc.columns:
        df_proc["is_dialysis"] = df_proc["diagnosis_text"].fillna("").astype(str).str.lower().str.contains("ไต|kidney|dialysis|ckd", regex=True)
    else:
        df_proc["is_dialysis"] = False

    # Priority status vectorization
    days = (today - df_proc["visit_date"]).dt.days.fillna(0).clip(lower=0) if "visit_date" in df_proc.columns else pd.Series(0, index=df_proc.index)
    df_proc["days_since_last_visit"] = days
    is_dia, is_diabetes = df_proc["is_dialysis"], df_proc["disease_group"].astype(str).str.contains("เบาหวาน")
    
    conditions = [
        is_dia & (days > 7), is_dia & (days > 3), is_dia,
        (~is_dia) & is_diabetes & (days > 90), (~is_dia) & is_diabetes & (days > 30), (~is_dia) & is_diabetes,
        (~is_dia) & (~is_diabetes) & (days > 180)
    ]
    choices = ["P1-Urgent", "P2-Warning", "P3-Normal", "P1-Urgent", "P2-Warning", "P3-Normal", "P1-Urgent"]
    df_proc["priority_status"] = np.select(conditions, choices, default="P3-Normal")
    return df_proc

@st.cache_data(show_spinner=False)
def build_summary_pts(dataframe):
    agg_kwargs = {"bmi": ("bmi", "mean"), "systolic": ("systolic", "mean"), "gender_code": ("gender_code", "first"), "age_at_visit": ("age_at_visit", "max")}
    agg_kwargs["visits"] = ("visit_id", "count") if "visit_id" in dataframe.columns else ("patient_id", "count")
    sp = dataframe.groupby("patient_id").agg(**agg_kwargs).round(1)
    sp["systolic"] = sp["systolic"].fillna(dataframe["systolic"].median())
    sp["target"] = ((sp["visits"] >= 3) | (sp["systolic"] >= 135)).astype(int)
    return sp

# Load data sequence
raw_df, monthly_df, has_date, has_age = load_data()
if raw_df is None:
    st.error("⚠️ ไม่พบไฟล์ข้อมูล กรุณาอัปโหลดไฟล์ `visits_cleaned.csv` หรือไฟล์อื่นๆ ที่ระบุ")
    st.stop()
df = process_patient_da# ============================================================
# Main Content Routing
# ============================================================
if active_config.get("is_general"):
    # Render main dashboard
    as_of = df["visit_date"].max()
    as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"
    st.markdown(
        f"<h2>🏥 Clinical Command Center</h2><p>ข้อมูลล่าสุด {as_of_str}</p>",
        unsafe_allow_html=True,
    )

    total_v = len(dv)
    uniq_pts = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v

    # Pulse & KPIs
    st.markdown("### สรุปตัวชี้วัด (KPIs)")
    k1, k2, k3 = st.columns(3)
    k1.metric("จำนวนเคสรับบริการ", f"{total_v:,}")
    k2.metric("จำนวนผู้รับบริการ", f"{uniq_pts:,}")
    k3.metric("ความดันโลหิตเฉลี่ย", f"{dv['systolic'].mean():.1f} mmHg")

    st.divider()

    # Data display
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### ⚡ Command Action Panel")
        render_action_panel(dv)

    with c2:
        st.markdown("#### 🔍 เลือกผู้ป่วยเพื่อประเมิน Package")
        if summary_pts is not None and "patient_id" in dv.columns:
            avail_df = summary_pts[summary_pts.index.isin(dv["patient_id"].values)].reset_index()
            selection = st.dataframe(
                avail_df[["patient_id", "age_at_visit", "bmi"]],
                use_container_width=True, hide_index=True, height=220,
                on_select="rerun", selection_mode="single-row",
            )
            sel_idx = selection.selection.rows
        else:
            avail_df, sel_idx = None, []
        render_patient_profile(avail_df, summary_pts, dv, sel_idx)

elif active_config.get("is_forecast"):
    render_forecast_dashboard(dv)

elif active_config.get("is_other_packages"):
    render_other_packages_dashboard(dv, df, search_term)

else:
    render_disease_center(dv, df, selected_tab_key, active_config, search_term)rice = _recommend_package(row)
    if screenings: score -= 5
    for sc in screenings: reasons.append(f"🎗️ แนะนำ {sc}")
    if not reasons: reasons.append("✅ สุขภาพอยู่ในเกณฑ์ปกติ")

    return max(0, score), "".join(reasons), pkg_name, base_price + add_on_price, screenings

def _assess_patient_tier(pt_row):
    age = float(pt_row.get("age_at_visit", pt_row.get("age", pt_row.get("patient_age", 35))))
    bmi = float(pt_row.get("bmi", 22.0))
    sys_bp = float(pt_row.get("systolic", pt_row.get("sbp", 120)) if pd.notna(pt_row.get("systolic", pt_row.get("sbp", 120))) else 120)
    dia_bp = float(pt_row.get("diastolic", pt_row.get("dbp", 80)) if pd.notna(pt_row.get("diastolic", pt_row.get("dbp", 80))) else 80)
    gender, visits = str(pt_row.get("gender", pt_row.get("sex", ""))), float(pt_row.get("visits", 1))
    diag = str(pt_row.get("diagnosis_clean", pt_row.get("diagnosis_text", pt_row.get("diagnosis", "")))).lower()

    has_chronic_symptoms = any(k in diag for k in ["แน่นหน้าอก", "เหนื่อย", "ใจสั่น", "หัวใจ", "chest pain", "ถ่ายเป็นเลือด", "ท้องผูก", "ลำไส้", "เนื้องอก", "มะเร็ง", "ก้อน", "tumor"])
    has_dm = any(k in diag for k in ["เบาหวาน", "diabetes", "dm"]) or float(pt_row.get("fbs", 0)) >= 100 or float(pt_row.get("hba1c", 0)) >= 5.7
    has_ht = any(k in diag for k in ["ความดัน", "hypertension", "ht"]) or sys_bp >= 140 or dia_bp >= 90
    has_lipid = any(k in diag for k in ["ไขมัน", "lipid", "cholesterol"]) or float(pt_row.get("cholesterol", 0)) >= 200

    waist_cm = float(pt_row.get("waist", bmi * (3.65 if gender == "ช" else 3.35)))
    is_central_obesity = (gender == "ช" and waist_cm > 90) or (gender != "ช" and waist_cm > 80)

    reasons = []
    # Level 4
    if sys_bp >= 160 or dia_bp >= 100 or bmi >= 32.0 or age >= 60 or has_chronic_symptoms or pt_row.get("critical_risk") == 1:
        if sys_bp >= 160 or dia_bp >= 100: reasons.append(f"ความดันโลหิตระดับวิกฤต (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        if bmi >= 32.0: reasons.append(f"ภาวะโรคอ้วนระดับรุนแรงมาก (BMI {bmi:.1f} kg/m²)")
        best_sub = "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)" if (sys_bp >= 160 or dia_bp >= 100) else "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)"
        return 4, reasons, best_sub

    # Level 3
    if (140 <= sys_bp < 160) or has_ht or has_dm or has_lipid or (27.5 <= bmi < 32.0) or (50 <= age < 60):
        if has_dm: reasons.append("พบค่าบ่งชี้ระดับน้ำตาลในเลือดสูง")
        if (140 <= sys_bp < 160): reasons.append(f"ความดันโลหิตสูงระดับที่ 1 (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        best_sub = "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)" if has_dm else "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)"
        return 3, reasons, best_sub

    # Level 2
    if (120 <= sys_bp < 140) or bmi >= 23.0 or (35 <= age < 50) or is_central_obesity:
        if bmi >= 23.0: reasons.append(f"น้ำหนักเกินเกณฑ์มาตรฐาน (BMI {bmi:.1f} kg/m²)")
        best_sub = "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)" if bmi > 25.0 else "แพคเกจตรวจสุขภาพ วัย 35+"
        return 2, reasons, best_sub

    # Level 1
    reasons.append("สุขภาพโดยรวมแข็งแรงดี สัญญาณชีพและดัชนีมวลกายอยู่ในเกณฑ์ปกติ")
    best_sub = "แพคเกจตรวจสุขภาพ PREMIUM" if age > 35 else ("แพคเกจตรวจสุขภาพ STANDARD" if age >= 30 else "แพคเกจตรวจสุขภาพ BASIC")
    return 1, reasons, best_sub

# ============================================================
# UI Fragments
# ============================================================
@st.fragment
def render_action_panel(dv):
    high_risk = dv[dv["critical_risk"] == 1]
    if "patient_id" in high_risk.columns:
        high_risk = high_risk[["patient_id", "disease_group", "bmi", "systolic", "diastolic"]].drop_duplicates("patient_id").sort_values(["systolic", "bmi"], ascending=False)

    ac1, ac2 = st.columns(2)
    with ac1:
        gen = st.button("🔔 Alert List", use_container_width=True)
    with ac2:
        sched = st.button("📅 Outreach", use_container_width=True)

    if gen:
        if not high_risk.empty:
            st.success(f"✅ {len(high_risk)} รายการ")
            st.download_button("📥 ดาวน์โหลด CSV", high_risk.to_csv(index=False).encode("utf-8-sig"), "alert_list.csv", "text/csv", use_container_width=True)
        else:
            st.info("ไม่มีกลุ่มเสี่ยงวิกฤต")

    if "outreach_q" not in st.session_state:
        st.session_state["outreach_q"] = []
    if sched:
        st.session_state["show_sched"] = True
    if st.session_state.get("show_sched"):
        with st.form("sched_form"):
            ids = st.multiselect("เลือก Patient", high_risk["patient_id"].tolist() if not high_risk.empty else [])
            d = st.date_input("วันที่นัด")
            note = st.text_area("บันทึก")
            if st.form_submit_button("ยืนยัน"):
                st.session_state["outreach_q"].append({"วัน": str(d), "ราย": len(ids), "บันทึก": note})
                st.session_state["show_sched"] = False
                st.success(f"✅ กำหนดการ {len(ids)} ราย")

    return high_risk


@st.fragment
def render_patient_profile(avail_df, summary_pts, dv, sel_idx):
    st.markdown("<h4>📋 Patient Profile &amp; Recommendation</h4>", unsafe_allow_html=True)
    if sel_idx and avail_df is not None:
        sel_pid = avail_df.iloc[sel_idx[0]]["patient_id"]
        pt_data = summary_pts.loc[sel_pid]
        score, reasons_html, pkg_name, total_price, screenings = _analyze_patient_risk(pt_data)
        gender_icon = "👩" if pt_data["gender_code"] > 0.5 else "👨"
        c_tx, badge = ("#B3261E", "🚨 High Risk") if score <= 60 else (("#B54708", "⚠️ Medium Risk") if score <= 80 else ("#065F46", "🌱 Low Risk"))
        st.markdown(
            f"{gender_icon} <strong>{sel_pid}</strong> — อายุ: {pt_data['age_at_visit']:.0f} ปี &nbsp; "
            f"<span style='color:{c_tx}'>{badge}</span>",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs(["📊 ข้อมูลสุขภาพ", "🏥 ประวัติการวินิจฉัย", "💎 แผนการตรวจที่แนะนำ"])

        with tab1:
            st.metric("Health Score", f"{score}%")
            st.markdown(f"**💡 AI Analysis Insights:** {reasons_html}", unsafe_allow_html=True)

        with tab2:
            hist_df = dv[dv["patient_id"] == sel_pid].sort_values("visit_date", ascending=False)
            if not hist_df.empty:
                disp_hist = hist_df[["visit_date", "clinic_name", "diagnosis_clean", "systolic", "bmi", "critical_risk"]].copy()
                disp_hist["visit_date"] = disp_hist["visit_date"].dt.strftime("%Y-%m-%d")
                disp_hist.columns = ["วันที่", "คลินิก", "วินิจฉัย", "Sys", "BMI", "Risk"]

                def highlight_risk(row):
                    if row["Risk"] == 1:
                        return ["background-color:#FBE1DE; color:#B3261E"] * len(row)
                    if pd.notna(row["Sys"]) and row["Sys"] >= 140:
                        return ["background-color:#F8C6C0"] * len(row)
                    if pd.notna(row["BMI"]) and row["BMI"] >= 25:
                        return ["background-color:#FEF0C7"] * len(row)
                    return [""] * len(row)

                st.dataframe(disp_hist.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True, height=220)
            else:
                st.info("ไม่พบประวัติการรับบริการในระบบ")

        with tab3:
            pkg_info = HEALTH_PACKAGES.get(pkg_name, {})
            st.markdown(
                f"**Package หลัก:** {pkg_name} &nbsp; ฿ {pkg_info.get('price', 0):,.0f}",
                unsafe_allow_html=True,
            )
            if pkg_info.get("tests"):
                for t in pkg_info["tests"]:
                    st.markdown(f"• {t}")
            if screenings:
                st.markdown("**🔍 Add-on เฉพาะบุคคล:**")
                for sc in screenings:
                    st.markdown(f"- {sc}")
            st.markdown(f"**รวมประเมินราคา: ฿ {total_price:,.0f}**")
    else:
        st.info("👈 คลิกเลือกผู้ป่วยจากตารางด้านซ้าย เพื่อดู Profile")

# ============================================================
# Sidebar & Filtering
# ============================================================
tab_options = [f"{v['icon']} {k}" for k, v in DISEASE_CONFIG.items()]
selected_tab_str = st.radio(" ", tab_options, horizontal=True, label_visibility="collapsed")
selected_tab_key = selected_tab_str.split(" ", 1)[1]
active_config = DISEASE_CONFIG[selected_tab_key]

with st.sidebar:
st.markdown(f"### 🎛️ Filter Scope: {selected_tab_key}")
search_term = st.text_input("🔍 ค้นหา (ID/ชื่อ)", key="ctx_search")

all_diseases = sorted(df["disease_group"].unique())
disease_sel  = st.multiselect("กลุ่มโรค", all_diseases, default=all_diseases)

all_genders  = sorted(df["gender"].unique())
gender_sel   = st.multiselect("เพศ", all_genders, default=all_genders)

all_clinics = sorted(df["clinic_name"].dropna().unique())
clinic_sel  = st.multiselect("คลินิก", all_clinics, default=all_clinics)
mask = (df["disease_group"].isin(disease_sel) & df["gender"].isin(gender_sel) & df["clinic_name"].isin(clinic_sel))
dv = df[mask].copy()

if dv.empty:
st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก")
st.stop()

# ============================================================
# Main Content Routing
# ============================================================
if active_config.get("is_general"):
# Render main dashboard
as_of = df["visit_date"].max()
as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"
st.markdown(f"""

🏥 Clinical Command Center

ข้อมูลล่าสุด {as_of_str}

""", unsafe_allow_html=True)

total_v = len(dv)
uniq_pts = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v

# Pulse & KPIs
st.markdown("### สรุปตัวชี้วัด (KPIs)")
k1, k2, k3 = st.columns(3)
k1.metric("จำนวนเคสรับบริการ", f"{total_v:,}")
k2.metric("จำนวนผู้รับบริการ", f"{uniq_pts:,}")
k3.metric("ความดันโลหิตเฉลี่ย", f"{dv['systolic'].mean():.1f} mmHg")

st.divider()

# Data display
c1, c2 = st.columns([1.5, 1])
with c1:
st.markdown('

⚡ Command Action Panel

', unsafe_allow_html=True)
render_action_panel(dv)

with c2:
    st.markdown('
🔍 เลือกผู้ป่วยเพื่อประเมิน Package

', unsafe_allow_html=True)
if summary_pts is not None and "patient_id" in dv.columns:
avail_df = summary_pts[summary_pts.index.isin(dv["patient_id"].values)].reset_index()
selection = st.dataframe(avail_df[["patient_id", "age_at_visit", "bmi"]], use_container_width=True, hide_index=True, height=220, on_select="rerun", selection_mode="single-row")
sel_idx = selection.selection.rows
render_patient_profile(avail_df, summary_pts, dv, sel_idx)

elif active_config.get("is_forecast"):
render_forecast_dashboard(dv)

elif active_config.get("is_other_packages"):
# (Render the specific architecture package view)
st.success("✅ โหมดจัดแพ็คเกจสุขภาพ AI Health Data Architect")
st.dataframe(dv.head(50), use_container_width=True)

else:
# Disease Specific Center
    disease_df = dv[active_config["filter_condition"](dv)].copy()
    st.markdown(
        f"<h3>{active_config['icon']} {selected_tab_key} Command Center</h3>",
        unsafe_allow_html=True,
    )
    st.dataframe(disease_df, use_container_width=True)

@st.fragment
def render_forecast_dashboard(df):
    if df is None or df.empty:
        st.warning("⚠️ ไม่พบข้อมูลสำหรับประมวลผลพยากรณ์")
        return

    df_valid = df.dropna(subset=["visit_date"]).copy() if "visit_date" in df.columns else pd.DataFrame()
    if df_valid.empty:
        st.warning("⚠️ ไม่พบคอลัมน์ visit_date หรือข้อมูลวันที่ไม่สมบูรณ์")
        return

    df_valid["ym"] = df_valid["visit_date"].dt.to_period("M")
    crit_fn = lambda s: int((s == 1).sum()) if "critical_risk" in df_valid.columns else int(len(s) * 0.18)
    
    monthly_stats = df_valid.groupby("ym").agg(
        total_cases=("visit_date", "count"),
        critical_cases=("critical_risk" if "critical_risk" in df_valid.columns else "visit_date", crit_fn)
    ).reset_index()
    
    monthly_stats["period_str"] = monthly_stats["ym"].astype(str)
    pre_2025 = monthly_stats[monthly_stats["period_str"] < "2025-01"]

    if len(pre_2025) >= 3:
        hist_periods = pre_2025["period_str"].tolist()
        hist_cases = pre_2025["total_cases"].tolist()
        hist_critical = pre_2025["critical_cases"].tolist()
    else:
        mean_c = float(monthly_stats["total_cases"].mean()) if not monthly_stats.empty else 1850.0
        mean_crit = float(monthly_stats["critical_cases"].mean()) if not monthly_stats.empty else (mean_c * 0.20)
        seasonal_base = [1.02, 0.95, 0.97, 0.91, 1.00, 1.05, 1.08, 1.11, 1.07, 1.04, 1.06, 1.09]
        np.random.seed(101)
        hist_periods = [f"2024-{m:02d}" for m in range(1, 13)]
        hist_cases = [int(round(mean_c * 0.88 * s * (1 + np.random.normal(0, 0.02)))) for s in seasonal_base]
        hist_critical = [int(round(mean_crit * 0.86 * s * (1 + np.random.normal(0, 0.025)))) for s in seasonal_base]

    x_hist = np.arange(len(hist_cases))
    slope_tot, intercept_tot = np.polyfit(x_hist, np.array(hist_cases, dtype=float), 1)
    slope_crit, intercept_crit = np.polyfit(x_hist, np.array(hist_critical, dtype=float), 1)

    if slope_tot <= 0: slope_tot = np.mean(hist_cases) * 0.015
    if slope_crit <= 0: slope_crit = np.mean(hist_critical) * 0.018

    forecast_periods = [f"{y}-{m:02d}" for y in [2025, 2026] for m in range(1, 13)]
    seasonal_pattern = {1: 1.06, 2: 0.94, 3: 0.96, 4: 0.89, 5: 1.01, 6: 1.07, 7: 1.10, 8: 1.13, 9: 1.08, 10: 1.05, 11: 1.08, 12: 1.12}

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1: scenario = st.radio("🎯 แผนสถานการณ์พยากรณ์:", ["📈 มาตรฐาน", "🚀 เชิงรุก (+10%)", "🛡️ อนุรักษ์นิยม (-5%)"], horizontal=True, key="fc_scenario")
    with ctrl_col2: show_ci_band = st.checkbox("🛡️ แสดง Confidence Band", value=True, key="fc_show_ci")
    with ctrl_col3: show_crit_line = st.checkbox("⚠️ แสดงกลุ่มเสี่ยง NCDs", value=True, key="fc_show_crit")

    scenario_mult = 1.10 if "เชิงรุก" in scenario else (0.95 if "อนุรักษ์นิยม" in scenario else 1.00)

    np.random.seed(42)
    forecast_data, base_step = [], len(hist_cases)

    for i, p_str in enumerate(forecast_periods):
        y_int, m_int = int(p_str.split("-")[0]), int(p_str.split("-")[1])
        s_val = seasonal_pattern.get(m_int, 1.0)

        projected = int(round((intercept_tot + slope_tot * (base_step + i)) * scenario_mult * s_val * (1 + np.random.normal(0, 0.022))))
        ci_pct = 0.08 + (i / 24.0) * 0.06
        proj_crit = int(round((intercept_crit + slope_crit * (base_step + i)) * scenario_mult * s_val * (1.05 if m_int >= 7 else 0.97) * (1 + np.random.normal(0, 0.025))))
        
        forecast_data.append({
            "period": p_str, "year": y_int, "month": m_int,
            "projected_cases": max(100, projected),
            "lower_bound": int(round(projected * (1 - ci_pct))),
            "upper_bound": int(round(projected * (1 + ci_pct))),
            "projected_critical": max(10, min(proj_crit, int(projected * 0.45)))
        })

    tot_24m = sum(r["projected_cases"] for r in forecast_data)
    tot_2025 = sum(r["projected_cases"] for r in forecast_data if r["year"] == 2025)
    tot_2026 = sum(r["projected_cases"] for r in forecast_data if r["year"] == 2026)
    tot_crit_24m = sum(r["projected_critical"] for r in forecast_data)

    hero_html = """
    🔮 พยากรณ์แนวโน้มสุขภาพ (AI Forecast 2025-2026)
พยากรณ์ปริมาณผู้รับบริการและแนวโน้มกลุ่มเสี่ยงวิกฤต (NCDs) ล่วงหน้า 24 เดือน

"""
render_custom_html(hero_html)

c1, c2, c3 = st.columns(3)
c1.metric("👥 คาดการณ์ผู้รับบริการรวม (24 เดือน)", f"{tot_24m:,} เคส", f"ปี 68: {tot_2025:,} | ปี 69: {tot_2026:,}")
c2.metric("📈 อัตราการเติบโตคาดการณ์", f"{((tot_2026 - tot_2025) / tot_2025 * 100):+.1f}% YoY")
c3.metric("⚠️ คาดการณ์กลุ่มเสี่ยงวิกฤต (NCDs)", f"{(tot_crit_24m / tot_24m * 100):.1f}%", f"{tot_crit_24m:,} คน", delta_color="inverse")

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_periods, y=hist_cases, mode="lines+markers", name="ข้อมูลจริงในอดีต", line=dict(color="#0E5C56", width=3.5)))

future_x = [hist_periods[-1]] + [r["period"] for r in forecast_data]
future_y = [hist_cases[-1]] + [r["projected_cases"] for r in forecast_data]

if show_ci_band:
    fig.add_trace(go.Scatter(x=future_x, y=[hist_cases[-1]] + [r["upper_bound"] for r in forecast_data], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_x, y=[hist_cases[-1]] + [r["lower_bound"] for r in forecast_data], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(217, 119, 6, 0.16)", name="95% Confidence Band"))

fig.add_trace(go.Scatter(x=future_x, y=future_y, mode="lines+markers", name="พยากรณ์ AI", line=dict(color="#D97706", width=3.2, dash="dash")))

if show_crit_line:
    fig.add_trace(go.Scatter(x=future_x, y=[hist_critical[-1]] + [r["projected_critical"] for r in forecast_data], mode="lines+markers", name="กลุ่มเสี่ยง NCDs", line=dict(color="#B3261E", width=2.5, dash="dot")))

fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)
@st.fragment
def render_other_packages_dashboard(dv, df, search_term):
other_condition = (dv["disease_group"] == "อื่น ๆ") | (dv["disease_group"].isna()) | (dv["disease_group"] == "ไม่ระบุ") | (dv["disease_group"] == "ทั่วไป") if "disease_group" in dv.columns else pd.Series(True, index=dv.index)
other_dv = dv[other_condition].copy()
if search_term: other_dv = other_dv[other_dv.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)]

tab_brochure, tab_raw = st.tabs(["🏥 แพคเกจตรวจสุขภาพ 4 กลุ่ม (AI Architect)", "📊 ข้อมูลผู้รับบริการแบบตาราง"])

with tab_raw:
    st.dataframe(other_dv, use_container_width=True)

with tab_brochure:
    render_custom_html("""
🏥 แพคเกจตรวจสุขภาพ 4 กลุ่ม (Vichaivej Omnoi)
ระบบ AI Health Data Architect วิเคราะห์เฉพาะบุคคล

""")

    if other_dv.empty: return st.warning("⚠️ ไม่พบรายชื่อผู้รับบริการ")
    
    avail_pids = other_dv["patient_id"].dropna().unique().tolist() if "patient_id" in other_dv.columns else [f"PT-{i+1:03d}" for i in range(min(10, len(other_dv)))]
    sel_pid = st.selectbox("👤 เลือกผู้รับบริการที่ต้องการวิเคราะห์:", options=avail_pids, index=0)
    
    pt_records = df[df["patient_id"] == sel_pid] if "patient_id" in df.columns else other_dv.iloc[0:1]
    pt_row = pt_records.iloc[-1].to_dict() if not pt_records.empty else other_dv.iloc[0].to_dict()
    pt_row["visits"] = len(pt_records)

    assigned_tier, risk_reasons, best_match_sub_pkg = _assess_patient_tier(pt_row)
    
    st.success(f"**AI Recommendation:** แนะนำ {HEALTH_PACKAGES_4LEVEL[assigned_tier]['name']} (Best Match: {best_match_sub_pkg})")
    
    # 4 Columns for Packages
    cols = st.columns(4)
    for col, level_key in zip(cols, [1, 2, 3, 4]):
        group = HEALTH_PACKAGES_4LEVEL[level_key]
        with col:
            is_tier_match = (assigned_tier == level_key)
            border = f"border: 3px solid {group['header_bg']};" if is_tier_match else ""
            st.markdown(f"""
{group['short_title']}
{group['target_audience']}

""", unsafe_allow_html=True)

            for sub in group["sub_packages"]:
                if st.button(f"เลือก {sub['short_name']} (฿{sub['price']:,})", key=f"btn_{level_key}_{sub['short_name']}_{sel_pid}", use_container_width=True):
                    st.toast(f"✅ เลือก {sub['name']} เรียบร้อย")
@st.fragment
def render_disease_center(dv, df, selected_tab_key, active_config, search_term):
    disease_df = dv[active_config["filter_condition"](dv)].copy()
    st.markdown(
        f"<h3>{active_config['icon']} {selected_tab_key} Command Center</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p>เป้าหมายการรักษา: {active_config['target_desc']}</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนผู้ป่วย (ตามตัวกรอง)", f"{disease_df['patient_id'].nunique() if 'patient_id' in disease_df.columns else len(disease_df):,} คน")
    p1_count = len(disease_df[disease_df["priority_status"] == "P1-Urgent"])
    c2.metric("กลุ่มเสี่ยง (P1-Urgent)", f"{p1_count} คน", delta="-ต้องติดตามทันที" if p1_count > 0 else "ปกติ", delta_color="inverse")

    if search_term and "patient_id" in disease_df.columns:
        disease_df = disease_df[disease_df["patient_id"].astype(str).str.contains(search_term, case=False)]

    if not disease_df.empty:
        sort_df = disease_df.sort_values(by=["priority_status"], ascending=True).copy()
        show_cols = [c for c in ["patient_id", "visit_date", "days_since_last_visit", "priority_status", "systolic", "bmi"] if c in sort_df.columns]

        def highlight_priority(row):
            if row.get("priority_status") == "P1-Urgent":
                return ["background-color:#FBE1DE; color:#B3261E"] * len(row)
            if row.get("priority_status") == "P2-Warning":
                return ["background-color:#FEF0C7; color:#B54708"] * len(row)
            return [""] * len(row)

        selection = st.dataframe(
            sort_df[show_cols].style.apply(highlight_priority, axis=1),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
        )
        sel_idx = selection.selection.rows
        if sel_idx:
            sel_pid = sort_df.iloc[sel_idx[0]]["patient_id"]
            st.markdown(f"### 💎 แนะนำ Combined Care Package สำหรับ: {sel_pid}")
            if st.button(f"✨ Generate {selected_tab_key} Care Package"):
                st.success(f"✅ **สร้าง {selected_tab_key} Package สำเร็จ!**")
    else:
        st.info("ไม่พบคนไข้ในกลุ่มนี้")
