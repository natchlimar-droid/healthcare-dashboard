import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time
import zipfile
import io

# optional dependencies
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

# ตั้งเป็น True เฉพาะตอนต้องการ interactive grid จริงๆ (AgGrid โหลด JS bundle ใหม่ทุก rerun ทำให้หน่วง)
USE_AGGRID = False

# ============================================================
# Design tokens
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
    "ความดันโลหิตสูง":  TEAL,
    "เบาหวาน":          "#2F6FB5",
    "ไขมันในเลือดสูง":  AMBER,
    "อื่น ๆ":           "#9AA6A0",
}
BP_COLORS = {
    "ปกติ (<120)":         SAGE,
    "เฝ้าระวัง (120-139)": AMBER,
    "สูง (≥140)":          RED,
    "ไม่มีข้อมูล":         "#C7CFCC",
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
        "level": 1,
        "name": "กลุ่ม 1: ตรวจสุขภาพทั่วไป",
        "short_title": "ตรวจสุขภาพทั่วไป",
        "target_audience": "สำหรับคนที่ใส่ใจสุขภาพ อยากรู้พื้นฐานของตัวเอง",
        "badge": "“เช็กสุขภาพวันนี้ เพื่อความมั่นใจในอนาคต”",
        "badge_color": "#0E7055",
        "badge_bg": "#DCFCE7",
        "header_bg": "#0E7055",
        "bg_card": "#F0FDF4",
        "border_color": "#0E7055",
        "text_accent": "#0E7055",
        "price": 990,
        "price_display": "เริ่มต้น 990 บาท",
        "tests": [
            "Physical Examination (ตรวจร่างกายทั่วไป)",
            "Complete Blood Count (ตรวจเลือดพื้นฐาน/ครบชุด)",
            "Urine Analysis (ตรวจปัสสาวะสมบูรณ์แบบ)",
            "Chest X-ray (เอกซเรย์ปอด)",
            "EKG (คลื่นไฟฟ้าหัวใจ)",
            "Ultrasound Upper/Whole Abdomen (อัลตราซาวด์ช่องท้อง)"
        ],
        "sub_packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ BASIC",
                "short_name": "BASIC",
                "price": 990,
                "price_display": "990 บาท",
                "description": "เหมาะสำหรับผู้ที่ต้องการตรวจเช็กความสมบูรณ์ของร่างกายเบื้องต้น",
                "tests": [
                    "ตรวจร่างกายทั่วไป",
                    "ตรวจเลือดพื้นฐาน",
                    "ตรวจปัสสาวะ, เอกซเรย์ปอด"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ STANDARD",
                "short_name": "STANDARD",
                "price": 2290,
                "price_display": "2,290 บาท",
                "description": "เหมาะสำหรับวัยทำงาน ตรวจละเอียดขึ้นครอบคลุมคลื่นไฟฟ้าหัวใจและอัลตราซาวด์",
                "tests": [
                    "ตรวจร่างกายทั่วไป",
                    "ตรวจเลือดครบชุด",
                    "เอกซเรย์ปอด, ตรวจคลื่นไฟฟ้าหัวใจ",
                    "(Ultrasound ช่องท้องส่วนบน)"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ PREMIUM",
                "short_name": "PREMIUM",
                "price": 4990,
                "price_display": "4,990 บาท",
                "description": "เหมาะสำหรับการตรวจคัดกรองสุขภาพประจำปีแบบครบถ้วนสมบูรณ์สูงสุด",
                "tests": [
                    "ตรวจร่างกายละเอียด",
                    "ตรวจเลือดครบชุด",
                    "เอกซเรย์ปอด, คลื่นไฟฟ้าหัวใจ",
                    "อัลตราซาวด์ช่องท้องส่วนบน-ล่าง"
                ]
            }
        ]
    },
    2: {
        "level": 2,
        "name": "กลุ่ม 2: ยังไม่พบโรคแต่มีความเสี่ยง",
        "short_title": "ยังไม่พบโรคแต่มีความเสี่ยง",
        "target_audience": "สำหรับผู้ที่มีปัจจัยเสี่ยงจากอายุ น้ำหนัก พฤติกรรม หรือประวัติครอบครัว",
        "badge": "“รู้ความเสี่ยงก่อนเกิดโรค ป้องกันได้...ให้ชีวิตยืนยาว”",
        "badge_color": "#B45309",
        "badge_bg": "#FEF3C7",
        "header_bg": "#D97706",
        "bg_card": "#FFFBEB",
        "border_color": "#D97706",
        "text_accent": "#D97706",
        "price": 2990,
        "price_display": "เริ่มต้น 2,990 บาท",
        "tests": [
            "Fasting Blood Sugar & HbA1c (ตรวจระดับน้ำตาลสะสม)",
            "Lipid Profile (ตรวจระดับไขมันในเลือด)",
            "Liver & Kidney Function (ตรวจการทำงานของตับ ไต)",
            "ประเมินภาวะแทรกซ้อนและอ้วนลงพุง (Metabolic Syndrome)",
            "เอกซเรย์ปอด อัลตราซาวด์ช่องท้อง"
        ],
        "sub_packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH",
                "short_name": "9.9 METABOLIC HEALTH MONTH",
                "price": 2990,
                "price_display": "2,990 บาท",
                "description": "เน้นประเมินภาวะอ้วนลงพุง ระดับน้ำตาลสะสม และไขมันในเลือด",
                "tests": [
                    "ตรวจระดับน้ำตาล (FBS, HbA1c)",
                    "ตรวจไขมันในเลือด",
                    "ตรวจการทำงานของตับ ไต",
                    "ประเมินภาวะอ้วนลงพุง (Metabolic Syndrome)"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)",
                "short_name": "คนอ้วน (Obesity Check)",
                "price": 2990,
                "price_display": "2,990 บาท",
                "description": "สำหรับผู้ที่มีภาวะน้ำหนักเกิน ตรวจคัดกรองภาวะแทรกซ้อนพร้อมรับคำปรึกษาจากแพทย์",
                "tests": [
                    "ตรวจสุขภาพสำหรับผู้มีภาวะน้ำหนักเกิน",
                    "ประเมินภาวะแทรกซ้อนจากโรคอ้วน",
                    "ตรวจระดับน้ำตาล ไขมัน ตับ ไต",
                    "ให้คำแนะนำการดูแลน้ำหนักโดยแพทย์"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ วัย 35+",
                "short_name": "วัย 35+",
                "price": 3990,
                "price_display": "3,990 บาท",
                "description": "ตรวจคัดกรองโรคที่มักพบในวัยทำงาน พร้อมอัลตราซาวด์ช่องท้องและเอกซเรย์ปอด",
                "tests": [
                    "ตรวจคัดกรองโรคที่มักพบในวัยทำงาน",
                    "ตรวจระดับน้ำตาล ไขมัน ความดัน",
                    "ตรวจการทำงานของตับ ไต",
                    "เอกซเรย์ปอด อัลตราซาวด์ช่องท้อง"
                ]
            }
        ]
    },
    3: {
        "level": 3,
        "name": "กลุ่ม 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น",
        "short_title": "เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น",
        "target_audience": "สำหรับผู้ที่มีค่าผลตรวจเริ่มผิดปกติ หรือได้รับการวินิจฉัยโรคระยะเริ่มต้น",
        "badge": "“ตรวจให้ชัด...ดูแลได้เร็ว ลดความเสี่ยงในอนาคต”",
        "badge_color": "#B91C1C",
        "badge_bg": "#FEE2E2",
        "header_bg": "#DC2626",
        "bg_card": "#FFF1F2",
        "border_color": "#DC2626",
        "text_accent": "#DC2626",
        "price": 2990,
        "price_display": "เริ่มต้น 2,990 บาท",
        "tests": [
            "Fasting Blood Sugar & HbA1c (ตรวจน้ำตาลสะสม)",
            "Comprehensive Lipid Profile (ตรวจไขมันครบชุด)",
            "Liver & Kidney Panel (ตรวจการทำงานของตับ ไต)",
            "Microalbuminuria (ตรวจโปรตีนไข่ขาวรั่วในปัสสาวะ)",
            "Electrocardiogram (คลื่นไฟฟ้าหัวใจ EKG)",
            "ปรึกษาแพทย์เฉพาะทางเพื่อวางแผนการรักษา"
        ],
        "sub_packages": [
            {
                "name": "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)",
                "short_name": "เบาหวาน (Diabetes Check)",
                "price": 3290,
                "price_display": "3,290 บาท",
                "description": "เจาะลึกค่าน้ำตาลสะสม การทำงานของไต และตรวจไข่ขาวในปัสสาวะ",
                "tests": [
                    "ตรวจระดับน้ำตาล (FBS, HbA1c)",
                    "ตรวจไขมัน การทำงานของตับ ไต",
                    "ตรวจปัสสาวะ (Microalb)",
                    "พบแพทย์ให้คำแนะนำ"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)",
                "short_name": "ความดันโลหิตสูง (Hypertension Check)",
                "price": 2990,
                "price_display": "2,990 บาท",
                "description": "ตรวจประเมินหลอดเลือด คลื่นไฟฟ้าหัวใจ และผลกระทบต่อไตจากความดันโลหิต",
                "tests": [
                    "ตรวจความดันโลหิต",
                    "ตรวจการทำงานของหัวใจ (EKG)",
                    "ตรวจไขมัน การทำงานของไต",
                    "พบแพทย์ให้คำแนะนำ"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)",
                "short_name": "ไขมันในเลือดสูง (Lipid Check)",
                "price": 2990,
                "price_display": "2,990 บาท",
                "description": "ตรวจระดับไขมันทุกชนิด ละเอียด พร้อมประเมินความเสี่ยงโรคหัวใจและหลอดเลือด",
                "tests": [
                    "ตรวจไขมัน (Cholesterol, LDL, HDL, TG)",
                    "ตรวจการทำงานของตับ ไต",
                    "ประเมินความเสี่ยงต่อโรคหัวใจและหลอดเลือด",
                    "พบแพทย์ให้คำแนะนำ"
                ]
            }
        ]
    },
    4: {
        "level": 4,
        "name": "กลุ่ม 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม",
        "short_title": "มีอาการซ้ำๆ/ควรประเมินเพิ่มเติม",
        "target_audience": "สำหรับผู้ที่มีอาการผิดปกติซ้ำๆ หรือมีความกังวลและต้องการตรวจเชิงลึก",
        "badge": "“หาสาเหตุให้ชัดเจน เพื่อความสบายใจและการดูแลที่ตรงจุด”",
        "badge_color": "#5B21B6",
        "badge_bg": "#F3E8FF",
        "header_bg": "#6941C6",
        "bg_card": "#FAF5FF",
        "border_color": "#6941C6",
        "text_accent": "#6941C6",
        "price": 4990,
        "price_display": "เริ่มต้น 4,990 บาท",
        "tests": [
            "Specialized Tumor Biomarkers (ตรวจสารบ่งชี้มะเร็ง)",
            "Cardiovascular Examination (EKG, EST สมรรถภาพหัวใจ, Echo)",
            "Endoscopy / Colonoscopy (ส่องกล้องตรวจลำไส้ใหญ่และตัดติ่งเนื้อ)",
            "High-Resolution Imaging & Ultrasound",
            "การตรวจวินิจฉัยและดูแลอย่างใกล้ชิดโดยแพทย์เฉพาะทาง"
        ],
        "sub_packages": [
            {
                "name": "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)",
                "short_name": "คัดกรอง มะเร็ง (Cancer Screening)",
                "price": 4990,
                "price_display": "4,990 บาท",
                "description": "ตรวจสารบ่งชี้มะเร็งสำคัญ เอกซเรย์ปอด และตรวจอวัยวะตามความเสี่ยงเฉพาะบุคคล",
                "tests": [
                    "ตรวจสารบ่งชี้มะเร็ง (Tumor Marker)",
                    "เอกซเรย์ปอด / อัลตราซาวด์",
                    "ตรวจอวัยวะตามความเสี่ยง (ช/ญ)",
                    "พบแพทย์ให้คำแนะนำ"
                ]
            },
            {
                "name": "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)",
                "short_name": "หัวใจ (Heart Check)",
                "price": 4990,
                "price_display": "4,990 บาท",
                "description": "ตรวจประเมินสมรรถภาพหัวใจอย่างละเอียดด้วย EKG, EST และอัลตราซาวด์หัวใจ Echo",
                "tests": [
                    "ตรวจคลื่นไฟฟ้าหัวใจ (EKG)",
                    "ตรวจสมรรถภาพหัวใจ (EST)",
                    "อัลตราซาวด์หัวใจ (Echo)",
                    "ตรวจไขมัน และปัจจัยเสี่ยงอื่นๆ"
                ]
            },
            {
                "name": "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)",
                "short_name": "คัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)",
                "price": 8900,
                "price_display": "8,900 บาท",
                "description": "ส่องกล้องระบบทางเดินอาหารและลำไส้ใหญ่ ตรวจหาและตัดติ่งเนื้อโดยแพทย์ผู้เชี่ยวชาญ",
                "tests": [
                    "ส่องกล้องตรวจลำไส้ใหญ่",
                    "ตรวจหาติ่งเนื้อ (Polyp)",
                    "โดยทีมแพทย์เฉพาะทาง",
                    "พร้อมการดูแลหลังการตรวจ"
                ]
            }
        ]
    }
}

SPECIALIZED_DIAGNOSTIC_OPTIONS = {
    "ส่องกล้องระบบทางเดินอาหาร (EGD / Colonoscopy)": {
        "price": 7500,
        "desc": "คัดกรองแผลในกระเพาะอาหาร กรดไหลย้อนเรื้อรัง และติ่งเนื้อมะเร็งลำไส้ใหญ่",
        "icon": "🔬"
    },
    "Low-Dose CT Chest (คัดกรองมะเร็งปอดรังสีต่ำ)": {
        "price": 4500,
        "desc": "ตรวจคัดกรองมะเร็งปอดระยะเริ่มต้นในผู้สูบบุหรี่หรือสัมผัสฝุ่น PM2.5 เรื้อรัง",
        "icon": "🫁"
    },
    "ชุดคัดกรองสารบ่งชี้มะเร็งรวม (CEA, AFP, CA-125, PSA)": {
        "price": 3500,
        "desc": "คัดกรองมะเร็งระบบทางเดินอาหาร มะเร็งตับ รังไข่ และต่อมลูกหมาก",
        "icon": "🎗️"
    },
    "ตรวจคลื่นเสียงสะท้อนหัวใจ (Echocardiogram)": {
        "price": 4000,
        "desc": "ประเมินสมรรถภาพกล้ามเนื้อหัวใจ ลิ้นหัวใจ และการสูบฉีดเลือด",
        "icon": "❤️"
    },
    "ตรวจสมรรถภาพหลอดเลือดแดงส่วนปลาย (ABI Test)": {
        "price": 1500,
        "desc": "ตรวจหาการตีบตันของหลอดเลือดแดงส่วนปลายและประเมินอายุหลอดเลือด",
        "icon": "🩺"
    }
}

DISEASE_CONFIG = {
    "🏥 General Dashboard (หน้าแรก)": {
        "icon": "🏥",
        "is_general": True
    },
    "ล้างไต (Dialysis)": {
        "icon": "🩺",
        "filter_condition": lambda df: df["is_dialysis"] == True,
        "target_desc": "BP <130/80",
        "Key_Tests": ["BUN", "Creatinine", "Electrolytes", "CBC"]
    },
    "เบาหวาน (Diabetes)": {
        "icon": "🩸",
        "filter_condition": lambda df: df["disease_group"] == "เบาหวาน",
        "target_desc": "HbA1c <7.0",
        "Key_Tests": ["HbA1c", "Microalbuminuria", "Funduscopy"]
    },
    "ความดันโลหิตสูง (Hypertension)": {
        "icon": "🫀",
        "filter_condition": lambda df: df["disease_group"] == "ความดันโลหิตสูง",
        "target_desc": "BP <140/90",
        "Key_Tests": ["Lipid Profile", "Creatinine", "EKG"]
    },
    "ไขมันในเลือดสูง (Dyslipidemia)": {
        "icon": "🧈",
        "filter_condition": lambda df: df["disease_group"] == "ไขมันในเลือดสูง",
        "target_desc": "LDL <100",
        "Key_Tests": ["Lipid Profile", "Liver Function"]
    },
    "กลุ่มอื่น ๆ (Health Packages)": {
        "icon": "🧬",
        "filter_condition": lambda df: (df["disease_group"] == "อื่น ๆ") | (df["disease_group"].isna()) | (df["disease_group"] == "ไม่ระบุ") | (df["disease_group"] == "ทั่วไป"),
        "target_desc": "คัดกรองสุขภาพเชิงรุกและจัดแพ็คเกจ 4 ระดับ (Basic to Deep Health)",
        "is_other_packages": True,
        "Key_Tests": ["CBC", "FBS", "Lipid Profile", "Liver & Kidney Functions", "Specialized Screening"]
    }
}

SPECIAL_SCREENINGS = {
    "Mammogram": {"price": 2000, "desc": "คัดกรองมะเร็งเต้านม"},
    "PSA (มะเร็งต่อมลูกหมาก)": {"price": 2000, "desc": "คัดกรองมะเร็งต่อมลูกหมาก"}
}


# ============================================================
# Page config + CSS
# ============================================================
st.set_page_config(
    page_title="Clinical Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {INK}; }}
.stApp {{ background-color: {BG}; }}
.block-container {{ padding-top: 1.1rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }}

div[data-testid="stMetric"] {{
    background: {SURFACE}; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(11,27,43,0.07);
}}
div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem !important;
    font-weight: 600 !important; color: {INK};
}}
div[data-testid="stMetricLabel"] {{
    font-size: 0.8rem !important; font-weight: 600 !important; color: {MUTED} !important;
}}

.header-bar {{ padding-bottom: 12px; margin-bottom: 14px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; }}
.header-title {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
.header-sub {{ font-size: 0.83rem; color: {MUTED}; margin-top: 2px; }}
.asof-chip {{ background: {TEAL}; color: white; padding: 5px 14px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}

.pulse-card {{ background: {SURFACE}; border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 2px 14px rgba(11,27,43,0.08);
    display: flex; align-items: center; gap: 16px; height: 100%; }}
.pulse-dot {{ width: 54px; height: 54px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 1.05rem; color: white; flex-shrink: 0; }}
.pulse-label {{ font-size: 0.78rem; color: {MUTED}; font-weight: 600; }}
.pulse-status {{ font-size: 1.05rem; font-weight: 700; }}

.meter-wrap {{ margin-bottom: 10px; }}
.meter-top {{ display: flex; justify-content: space-between;
    font-size: 0.76rem; color: {MUTED}; margin-bottom: 4px; }}
.meter-track {{ background: #E7ECEA; border-radius: 20px; height: 8px; overflow: hidden; }}
.meter-fill {{ height: 100%; border-radius: 20px; }}
.panel-title {{ font-size: 0.9rem; font-weight: 700; color: {INK}; margin-bottom: 2px; }}
.panel-sub {{ font-size: 0.72rem; color: {MUTED}; margin-bottom: 10px; }}
.drill-banner {{ background: #EAF3F1; border-radius: 10px; padding: 8px 14px;
    font-size: 0.78rem; color: {TEAL}; margin-bottom: 8px; font-weight: 600; }}
.action-btn button {{ background: {TEAL} !important; color: white !important; border: none !important;
    border-radius: 10px !important; box-shadow: 0 2px 8px rgba(14,92,86,.25) !important; font-weight: 600 !important; }}
.action-btn-secondary button {{ background: {SURFACE} !important; color: {TEAL} !important;
    border: 1.5px solid {TEAL} !important; border-radius: 10px !important; font-weight: 600 !important; }}

/* --- 4-Level Package Grid Styling --- */
.pkg-header-hero {{
    background: linear-gradient(135deg, #0B1B2B 0%, #0E5C56 100%);
    border-radius: 16px;
    padding: 22px 24px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(11,27,43,0.12);
}}
.pkg-card-box {{
    border-radius: 16px;
    padding: 18px 16px;
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
.pkg-card-box:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.09);
}}
.pkg-pill-match {{
    background: linear-gradient(135deg, #0E5C56, #16A34A);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 8px;
    box-shadow: 0 2px 6px rgba(14,92,86,0.3);
}}
.metric-tile {{
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}}
.metric-tile-title {{
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 2px;
}}
.metric-tile-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
}}
.metric-tile-badge {{
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 2px;
}}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading — รองรับ CSV และ ZIP
# ============================================================

def read_csv_or_zip(file_path, preferred_csv_names=None):
    """
    อ่านไฟล์ CSV ปกติ หรือ ZIP ที่มี CSV อยู่ข้างใน

    preferred_csv_names:
    รายชื่อไฟล์ CSV ที่อยากเลือกก่อน หาก ZIP มีหลาย CSV
    """
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

    if file_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as z:
            csv_files = [
                name for name in z.namelist()
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
            ]
            if not csv_files:
                raise ValueError(f"ไม่พบไฟล์ CSV ภายใน ZIP: {file_path}")

            selected_file = None
            if preferred_csv_names:
                preferred_lower = {name.lower() for name in preferred_csv_names}
                for csv_name in csv_files:
                    base_name = os.path.basename(csv_name).lower()
                    if base_name in preferred_lower:
                        selected_file = csv_name
                        break

            if selected_file is None:
                selected_file = csv_files[0]

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
    main_file = find_data_file([
        "visits_cleaned.csv",
        "visits_cleaned.zip",
        "visits_with_monthly_count.csv",
        "visits_with_monthly_count.zip",
    ])

    if main_file is None:
        return None, None, False, False

    try:
        df = read_csv_or_zip(
            main_file,
            preferred_csv_names=["visits_cleaned.csv", "visits_with_monthly_count.csv"]
        )
    except Exception as e:
        st.error(f"⚠️ อ่านไฟล์ข้อมูลหลักไม่สำเร็จ: {e}")
        return None, None, False, False

    df.columns = df.columns.astype(str).str.strip()

    # --- วันที่ ---
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

    # --- แปลงตัวเลข ---
    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- ความดันโลหิต ---
    if "systolic_bp" not in df.columns:
        df["systolic_bp"] = np.nan
    if "diastolic_bp" not in df.columns:
        df["diastolic_bp"] = np.nan

    if "bp_raw" in df.columns and df["systolic_bp"].isna().all():
        bp_clean = df["bp_raw"].astype(str).str.replace(",", "", regex=False)
        bp_split = bp_clean.str.extract(r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$")
        sys_bp = pd.to_numeric(bp_split[0], errors="coerce")
        dia_bp = pd.to_numeric(bp_split[1], errors="coerce")
        sys_bp[~sys_bp.between(60, 250)] = np.nan
        dia_bp[~dia_bp.between(30, 150)] = np.nan
        df["systolic_bp"] = sys_bp
        df["diastolic_bp"] = dia_bp

    df.rename(columns={"systolic_bp": "systolic", "diastolic_bp": "diastolic"}, inplace=True)

    # --- เพศ ---
    if "gender" not in df.columns:
        df["gender"] = "ไม่ระบุ"
    else:
        df["gender"] = df["gender"].fillna("ไม่ระบุ")
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)

    # --- BMI ---
    if "bmi" not in df.columns:
        df["bmi"] = 22.0
        df["bmi_imputed"] = True
    else:
        df["bmi_imputed"] = df["bmi"].isna()
        median_bmi = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(median_bmi if pd.notna(median_bmi) else 22.0)

    # --- อายุ ---
    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()

    if has_age:
        median_age = df["age_at_visit"].median()
        df["age_at_visit"] = df["age_at_visit"].fillna(median_age if pd.notna(median_age) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18

        df["age_group"] = pd.cut(
            df["age_at_visit"], bins=[0, 29, 39, 49, 59, 120],
            labels=["<30 ปี", "30-40 ปี", "40-50 ปี", "50-60 ปี", ">60 ปี"]
        ).astype(str).replace("nan", "ไม่ระบุ")

        df["pyramid_group"] = pd.cut(
            df["age_at_visit"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False,
            labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        ).astype(str)
    else:
        df["age_at_visit"] = 35.0
        df["is_adult"] = True
        df["age_group"] = "ไม่ระบุ"
        df["pyramid_group"] = "ไม่ระบุ"

    # --- ระดับความดัน / กลุ่มเสี่ยง ---
    bp_cat = pd.cut(
        df["systolic"], bins=[-1, 120, 139, 300],
        labels=["ปกติ (<120)", "เฝ้าระวัง (120-139)", "สูง (≥140)"]
    ).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat

    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    # --- Diagnosis / clinic / disease (vectorized) ---
    if "diagnosis_text" not in df.columns:
        df["diagnosis_text"] = ""
        df["diagnosis_clean"] = "ไม่ระบุ"
    else:
        cleaned = df["diagnosis_text"].astype(str).str.strip()
        cleaned = cleaned.where(df["diagnosis_text"].notna(), "ไม่ระบุ")
        cleaned = cleaned.replace({"": "ไม่ระบุ", "-": "ไม่ระบุ", ":": "ไม่ระบุ"})
        df["diagnosis_clean"] = cleaned

    if "disease_group" not in df.columns:
        df["disease_group"] = "ทั่วไป"
    if "clinic_name" not in df.columns:
        df["clinic_name"] = "ไม่ระบุ"

    # --- Monthly summary — ไม่บังคับ ---
    monthly = None
    monthly_file = find_data_file(["monthly_visit_summary.csv", "monthly_visit_summary.zip"])
    if monthly_file:
        try:
            monthly = read_csv_or_zip(monthly_file, preferred_csv_names=["monthly_visit_summary.csv"])
            monthly.columns = monthly.columns.astype(str).str.strip()
            if "visit_count" in monthly.columns:
                monthly["visit_count"] = pd.to_numeric(monthly["visit_count"], errors="coerce")
        except Exception:
            monthly = None

    return df, monthly, has_date, has_age


# โหลดข้อมูล
df, monthly_df, has_date, has_age = load_data()

if df is None:
    st.error(
        "⚠️ ไม่พบไฟล์ข้อมูล กรุณาอัปโหลดไฟล์ใดไฟล์หนึ่งต่อไปนี้: "
        "`visits_cleaned.csv`, `visits_cleaned.zip`, "
        "`visits_with_monthly_count.csv` หรือ "
        "`visits_with_monthly_count.zip`"
    )
    st.stop()


# ============================================================
# Process patient data — cached + vectorized (จุดที่หน่วงที่สุดเดิม)
# ============================================================
@st.cache_data(show_spinner=False)
def process_patient_data(dataframe):
    df_proc = dataframe.copy()

    today = df_proc["visit_date"].max()
    if pd.isna(today):
        today = pd.Timestamp.today().normalize()

    # 1) ระบุผู้ป่วยกลุ่มล้างไต — vectorized string match แทน .apply row-by-row
    if "diagnosis_text" in df_proc.columns:
        text_lower = df_proc["diagnosis_text"].fillna("").astype(str).str.lower()
        df_proc["is_dialysis"] = text_lower.str.contains("ไต|kidney|dialysis|ckd", regex=True)
    else:
        df_proc["is_dialysis"] = False

    # 2) จำนวนวันตั้งแต่มารับบริการครั้งนั้น
    if "visit_date" in df_proc.columns:
        df_proc["days_since_last_visit"] = (
            (today - df_proc["visit_date"]).dt.days.fillna(0).clip(lower=0)
        )
    else:
        df_proc["days_since_last_visit"] = 0

    # 3) ระดับความเร่งด่วน — np.select แทน .apply(axis=1)
    days = df_proc["days_since_last_visit"]
    disease = df_proc["disease_group"].astype(str)
    is_dia = df_proc["is_dialysis"]
    is_diabetes = disease.str.contains("เบาหวาน")

    conditions = [
        is_dia & (days > 7),
        is_dia & (days > 3),
        is_dia,
        (~is_dia) & is_diabetes & (days > 90),
        (~is_dia) & is_diabetes & (days > 30),
        (~is_dia) & is_diabetes,
        (~is_dia) & (~is_diabetes) & (days > 180),
    ]
    choices = [
        "P1-Urgent", "P2-Warning", "P3-Normal",
        "P1-Urgent", "P2-Warning", "P3-Normal",
        "P1-Urgent",
    ]
    df_proc["priority_status"] = np.select(conditions, choices, default="P3-Normal")

    return df_proc


# เรียกใช้หลังประกาศฟังก์ชันแล้วเท่านั้น
df = process_patient_data(df)


# ============================================================
# Lead scoring (optional sklearn) — แยก cache การสร้าง summary_pts
# ออกจากการเทรนโมเดล เพื่อไม่ให้ groupby รันใหม่ทุก rerun
# ============================================================
@st.cache_data(show_spinner=False)
def build_summary_pts(dataframe):
    agg_kwargs = {
        "bmi": ("bmi", "mean"),
        "systolic": ("systolic", "mean"),
        "gender_code": ("gender_code", "first"),
        "age_at_visit": ("age_at_visit", "max"),
    }
    if "visit_id" in dataframe.columns:
        agg_kwargs["visits"] = ("visit_id", "count")
    else:
        agg_kwargs["visits"] = ("patient_id", "count")

    sp = dataframe.groupby("patient_id").agg(**agg_kwargs).round(1)
    sp["systolic"] = sp["systolic"].fillna(dataframe["systolic"].median())
    sp["target"] = ((sp["visits"] >= 3) | (sp["systolic"] >= 135)).astype(int)
    return sp


@st.cache_resource(show_spinner=False)
def build_scorer(data):
    if not SKLEARN_AVAILABLE or len(data) < 10:
        return None
    feats = ["visits", "bmi", "systolic", "gender_code"]
    X, y = data[feats], data["target"]
    return RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42).fit(X, y)


summary_pts = high_lead_count = est_pipeline = clf = None
if "patient_id" in df.columns:
    summary_pts = build_summary_pts(df)
    clf = build_scorer(summary_pts)
    if clf is not None:
        probs = clf.predict_proba(summary_pts[["visits", "bmi", "systolic", "gender_code"]])[:, 1]
        summary_pts["lead_score"] = (probs * 100).astype(int)
        high_lead_count = int((summary_pts["lead_score"] >= 60).sum())
        est_pipeline = high_lead_count * 3000
    else:
        high_lead_count = est_pipeline = 0


# ============================================================
# Fragment-wrapped interactive sections
# ห่อด้วย @st.fragment เพื่อให้คลิกปุ่ม/เลือกแถวในตาราง ไม่ต้อง
# rerun กราฟหนักๆ (sunburst, scatter, pyramid) ทั้งหน้า
# ============================================================

@st.fragment
def render_action_panel(dv):
    """Command Action Panel: Alert List + Outreach scheduling"""
    high_risk = dv[dv["critical_risk"] == 1]
    if "patient_id" in high_risk.columns:
        high_risk = (
            high_risk[["patient_id", "disease_group", "bmi", "systolic", "diastolic"]]
            .drop_duplicates("patient_id")
            .sort_values(["systolic", "bmi"], ascending=False)
        )

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        gen = st.button("🔔 Alert List", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ac2:
        st.markdown('<div class="action-btn-secondary">', unsafe_allow_html=True)
        sched = st.button("📅 Outreach", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if gen:
        if not high_risk.empty:
            st.success(f"✅ {len(high_risk)} รายการ")
            st.download_button(
                "📥 ดาวน์โหลด CSV",
                high_risk.to_csv(index=False).encode("utf-8-sig"),
                "alert_list.csv", "text/csv", use_container_width=True
            )
        else:
            st.info("ไม่มีกลุ่มเสี่ยงวิกฤต")

    if "outreach_q" not in st.session_state:
        st.session_state["outreach_q"] = []
    if sched:
        st.session_state["show_sched"] = True
    if st.session_state.get("show_sched"):
        with st.form("sched_form"):
            ids = st.multiselect(
                "เลือก Patient",
                high_risk["patient_id"].tolist() if not high_risk.empty else [],
                default=(high_risk["patient_id"].tolist()[:5] if not high_risk.empty else [])
            )
            d = st.date_input("วันที่นัด")
            note = st.text_area("บันทึก")
            if st.form_submit_button("ยืนยัน"):
                st.session_state["outreach_q"].append({"วัน": str(d), "ราย": len(ids), "บันทึก": note})
                st.session_state["show_sched"] = False
                st.success(f"✅ กำหนดการ {len(ids)} ราย")
    if st.session_state["outreach_q"]:
        st.dataframe(pd.DataFrame(st.session_state["outreach_q"]), use_container_width=True, hide_index=True)

    return high_risk


def _recommend_package(row):
    """ประเมินแพ็กเกจจากช่วงอายุและเพศ รวมถึงการคัดกรองพิเศษ"""
    age = row.get("age_at_visit", 35)
    gender_code = row.get("gender_code", 0.5)

    if age >= 50:
        pkg_name, base_price = "Longevity Package", 8000
    elif age >= 30:
        pkg_name, base_price = "Advanced Package", 5500
    else:
        pkg_name, base_price = "Essential Package", 3000

    screenings = []
    add_on_price = 0
    if gender_code > 0.5 and age >= 40:
        screenings.append("Mammogram")
        add_on_price += 2000
    if gender_code < 0.5 and age >= 50:
        screenings.append("PSA (มะเร็งต่อมลูกหมาก)")
        add_on_price += 2000

    return pkg_name, base_price, screenings, add_on_price


def _analyze_patient_risk(row):
    """คำนวณ Health Score (0-100%) และคำแนะนำแพ็กเกจ"""
    score = 100
    reasons = []

    sys_val = row.get("systolic", 0)
    bmi_val = row.get("bmi", 22)
    visits_val = row.get("visits", 1)

    if sys_val >= 160:
        score -= 40
        reasons.append(f"<span style='background:#FBE1DE; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันวิกฤต ({sys_val:.0f})</span>")
    elif sys_val >= 140:
        score -= 25
        reasons.append(f"<span style='background:#F8C6C0; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันสูง ({sys_val:.0f})</span>")
    elif sys_val >= 130:
        score -= 10
        reasons.append(f"<span style='background:#FEF0C7; color:#B54708; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 เฝ้าระวังความดัน ({sys_val:.0f})</span>")

    if bmi_val >= 30:
        score -= 20
        reasons.append(f"<span style='background:#FBE1DE; color:#B3261E; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 โรคอ้วน ({bmi_val:.1f})</span>")
    elif bmi_val >= 25:
        score -= 10
        reasons.append(f"<span style='background:#FEF0C7; color:#B54708; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 น้ำหนักเกิน ({bmi_val:.1f})</span>")

    if visits_val >= 5:
        score -= 15
        reasons.append(f"<span style='background:#E0F2FE; color:#0369A1; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มารพ. บ่อยผิดปกติ ({visits_val:.0f} ครั้ง)</span>")
    elif visits_val >= 3:
        score -= 5
        reasons.append(f"<span style='background:#F1F5F9; color:#475569; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มีประวัติมาซ้ำ ({visits_val:.0f} ครั้ง)</span>")

    pkg_name, base_price, screenings, add_on_price = _recommend_package(row)

    if screenings:
        score -= 5
        for sc in screenings:
            reasons.append(f"<span style='background:#F3E8FF; color:#7E22CE; border:1px solid #D8B4FE; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🎗️ แนะนำ {sc}</span>")

    if not reasons:
        reasons.append("<span style='background:#ECFDF5; color:#065F46; padding:4px 8px; border-radius:12px; font-size:0.7rem; display:inline-block; margin-bottom:4px;'>✅ สุขภาพอยู่ในเกณฑ์ปกติ</span>")

    return max(0, score), "".join(reasons), pkg_name, base_price + add_on_price, screenings


@st.fragment
def render_patient_profile(avail_df, summary_pts, dv, sel_idx):
    """Patient Profile & Package Recommendation — ห่อ fragment เพื่อไม่ให้
    การเลือกคนไข้/สลับ tab ไปกระทบกราฟส่วนอื่นของหน้า"""
    st.markdown('<div class="panel-title">📋 Patient Profile & Recommendation</div>', unsafe_allow_html=True)

    if sel_idx and avail_df is not None:
        sel_pid = avail_df.iloc[sel_idx[0]]["patient_id"]
        pt_data = summary_pts.loc[sel_pid]

        score, reasons_html, pkg_name, total_price, screenings = _analyze_patient_risk(pt_data)
        gender_icon = "👩" if pt_data["gender_code"] > 0.5 else "👨"

        if score <= 60: c_tx, badge = "#B3261E", "🚨 High Risk"
        elif score <= 80: c_tx, badge = "#B54708", "⚠️ Medium Risk"
        else: c_tx, badge = "#065F46", "🌱 Low Risk"

        st.markdown(f"""
        <div style="background:{SURFACE}; border:1px solid #E2E8F0; border-radius:12px; padding:18px; margin-top:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="font-size:2.4rem; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:50%; width:54px; height:54px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">{gender_icon}</div>
                <div>
                    <div style="font-weight:700; color:{INK}; font-size:1.15rem;">{sel_pid}</div>
                    <div style="font-size:0.75rem; color:{MUTED}; margin-bottom:2px;">อายุ: {pt_data['age_at_visit']:.0f} ปี</div>
                    <div style="font-size:0.75rem; font-weight:600; color:{c_tx}; background:#F8FAFC; padding:2px 6px; border-radius:6px; display:inline-block;">{badge}</div>
                </div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 ข้อมูลสุขภาพ", "🏥 ประวัติการวินิจฉัย", "💎 แผนการตรวจที่แนะนำ"])

        with tab1:
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:16px;">
                <div style="font-size:0.75rem; color:{MUTED}; font-weight:600;">Health Score</div>
                <div style="font-size:2.5rem; font-weight:700; color:{c_tx}; font-family:'IBM Plex Mono',monospace; line-height:1.1;">{score}%</div>
            </div>
            <div style="font-size:0.78rem; font-weight:600; color:{INK}; margin-bottom:8px;">💡 AI Analysis Insights:</div>
            <div style="margin-bottom:12px; line-height:1.6;">
                {reasons_html}
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            hist_df = dv[dv["patient_id"] == sel_pid].sort_values("visit_date", ascending=False)
            if not hist_df.empty:
                st.markdown('<div style="font-size:0.75rem; color:#5B6B6B; margin-bottom:8px;">Timeline การมารับบริการ</div>', unsafe_allow_html=True)
                disp_hist = hist_df[["visit_date", "clinic_name", "diagnosis_clean", "systolic", "bmi", "critical_risk"]].copy()
                disp_hist["visit_date"] = disp_hist["visit_date"].dt.strftime("%Y-%m-%d")
                disp_hist.columns = ["วันที่", "คลินิก", "วินิจฉัย", "Sys", "BMI", "Risk"]

                def highlight_risk(row):
                    if row["Risk"] == 1: return ["background-color:#FBE1DE; color:#B3261E"] * len(row)
                    if pd.notna(row["Sys"]) and row["Sys"] >= 140: return ["background-color:#F8C6C0"] * len(row)
                    if pd.notna(row["BMI"]) and row["BMI"] >= 25: return ["background-color:#FEF0C7"] * len(row)
                    return [""] * len(row)

                st.dataframe(disp_hist.style.apply(highlight_risk, axis=1),
                             use_container_width=True, hide_index=True, height=220,
                             column_config={"Risk": None})
            else:
                st.info("ไม่พบประวัติการรับบริการในระบบ")

        with tab3:
            pkg_info = HEALTH_PACKAGES.get(pkg_name, {})
            st.markdown(f"""
            <div style="background:#F0F9FF; border-left:4px solid #0284C7; padding:10px 12px; border-radius:6px; margin-bottom:14px;">
                <div style="font-size:0.7rem; color:#0284C7; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:2px;">Package หลัก</div>
                <div style="font-size:1.05rem; font-weight:700; color:{INK};">{pkg_name}</div>
                <div style="font-size:0.7rem; color:{MUTED}; margin-bottom:4px;">{pkg_info.get("desc", "")}</div>
                <div style="font-size:0.85rem; font-weight:600; color:{MUTED}; font-family:'IBM Plex Mono',monospace;">฿ {pkg_info.get("price", 0):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            if pkg_info.get("tests"):
                st.markdown('<div style="font-size:0.75rem; font-weight:600; margin-bottom:4px;">รายการตรวจหลัก:</div>', unsafe_allow_html=True)
                for t in pkg_info["tests"]:
                    st.markdown(f'<div style="font-size:0.7rem; color:#5B6B6B; padding-left:12px;">• {t}</div>', unsafe_allow_html=True)

            if screenings:
                st.markdown('<div style="font-size:0.75rem; font-weight:600; margin-top:10px; margin-bottom:4px; color:#7E22CE;">🔍 Add-on เฉพาะบุคคล:</div>', unsafe_allow_html=True)
                for sc in screenings:
                    sp = SPECIAL_SCREENINGS.get(sc, {})
                    st.markdown(f'<div style="font-size:0.7rem; color:#7E22CE; padding-left:12px;">+ {sc} (฿{sp.get("price", 0):,.0f})</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:right; margin-top:12px; padding-top:8px; border-top:1px dashed #CBD5E1;">
                <span style="font-size:0.75rem; color:{MUTED};">รวมประเมินราคา: </span>
                <span style="font-size:1.2rem; font-weight:700; color:{TEAL};">฿ {total_price:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="action-btn" style="margin-top:14px;">', unsafe_allow_html=True)
            if st.button("✨ Generate Personalized Proposal", use_container_width=True):
                with st.spinner("กำลังเชื่อมต่อระบบ CRM/LINE API..."):
                    time.sleep(1)
                    st.success("✅ ส่งข้อมูลให้ระบบเรียบร้อย! (Webhook API Mockup)")
            st.markdown('</div>', unsafe_allow_html=True)

            export_text = f"--- Personalized Health Proposal ---\nPatient ID: {sel_pid}\nAge: {pt_data['age_at_visit']:.0f}\nHealth Score: {score}%\nRisk Level: {badge}\n\nRecommended Package: {pkg_name} (฿{pkg_info.get('price', 0):,.0f})\nDescription: {pkg_info.get('desc', '')}\n"
            if screenings:
                export_text += "\nAdd-on Screenings:\n"
                for sc in screenings:
                    sp = SPECIAL_SCREENINGS.get(sc, {})
                    export_text += f"- {sc} (฿{sp.get('price', 0):,.0f})\n"
            export_text += f"\nTotal Estimated Price: ฿{total_price:,.0f}\n"

            st.download_button(
                label="📥 Export Summary (Text)",
                data=export_text.encode('utf-8'),
                file_name=f"proposal_{sel_pid}.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:12px; padding:30px 16px; text-align:center; color:{MUTED};">
            <div style="font-size:2rem; margin-bottom:10px;">👈</div>
            <div style="font-size:0.9rem; font-weight:600;">คลิกเลือกผู้ป่วยจากตารางด้านซ้าย</div>
            <div style="font-size:0.8rem; margin-top:4px;">เพื่อดู Health Score และการแนะนำแพ็กเกจที่เหมาะสม</div>
        </div>
        """, unsafe_allow_html=True)


@st.fragment
def render_disease_center(dv, df, selected_tab_key, active_config, search_term):
    """Disease-specific Command Center — ห่อ fragment เพื่อให้การเลือกแถว/
    กด Generate Package ไม่ทำให้ต้อง rerun ตัวกรองทั้งหน้าใหม่"""
    disease_df = dv[active_config["filter_condition"](dv)].copy()

    st.markdown(f'<div class="header-title">{active_config["icon"]} {selected_tab_key} Command Center</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="header-sub">เป้าหมายการรักษา: {active_config["target_desc"]}</div><br>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนผู้ป่วย (ตามตัวกรอง)", f"{disease_df['patient_id'].nunique() if 'patient_id' in disease_df.columns else len(disease_df):,} คน")
    p1_count = len(disease_df[disease_df["priority_status"] == "P1-Urgent"])
    c2.metric("กลุ่มเสี่ยง (P1-Urgent)", f"{p1_count} คน", delta="-ต้องติดตามทันที" if p1_count > 0 else "ปกติ", delta_color="inverse")

    if search_term and "patient_id" in disease_df.columns:
        disease_df = disease_df[disease_df["patient_id"].astype(str).str.contains(search_term, case=False)]

    st.markdown("### 🚨 รายชื่อผู้ป่วย (Priority List)")
    if not disease_df.empty:
        sort_df = disease_df.sort_values(by=["priority_status"], ascending=True).copy()

        show_cols = ["patient_id", "visit_date", "days_since_last_visit", "priority_status", "systolic", "bmi"]
        show_cols = [c for c in show_cols if c in sort_df.columns]

        def highlight_priority(row):
            if row.get("priority_status") == "P1-Urgent": return ["background-color:#FBE1DE; color:#B3261E"] * len(row)
            if row.get("priority_status") == "P2-Warning": return ["background-color:#FEF0C7; color:#B54708"] * len(row)
            return [""] * len(row)

        selection = st.dataframe(
            sort_df[show_cols].style.apply(highlight_priority, axis=1),
            use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key=f"table_{selected_tab_key}"
        )

        sel_idx = selection.selection.rows
        if sel_idx:
            sel_pid = sort_df.iloc[sel_idx[0]]["patient_id"]
            st.markdown(f"### 💎 แนะนำ Combined Care Package สำหรับ: {sel_pid}")

            pt_all_visits = df[df["patient_id"] == sel_pid]

            combined_tests = []
            risk_multiplier = 1.0
            found_diseases = []

            for k, v in DISEASE_CONFIG.items():
                if v.get("is_general"): continue
                if any(v["filter_condition"](pt_all_visits)):
                    combined_tests.extend(v["Key_Tests"])
                    risk_multiplier += 0.2
                    found_diseases.append(k)

            combined_tests = list(set(combined_tests))

            st.info(f"**พหุโรค (Multi-morbidity):** พบ {len(found_diseases)} โรคเรื้อรังซ้อนทับ ได้แก่ {', '.join(found_diseases)} (Risk Multiplier: {risk_multiplier:.1f}x)")

            if st.button(f"✨ Generate {selected_tab_key} Care Package", key="btn_combo_pkg"):
                st.success(f"✅ **สร้าง {selected_tab_key} Package สำเร็จ!**")
                price = len(combined_tests) * 500 + 1500
                st.markdown(f"**ราคาประเมินรวม:** ฿ {price:,.0f}")
                st.markdown("**รายการตรวจที่สำคัญที่สุด (Merging Tests from Multi-morbidity):**")
                for t in combined_tests:
                    st.markdown(f"- {t}")
    else:
        st.info("ไม่พบคนไข้ในกลุ่มนี้")



# ============================================================
# Dashboard สำหรับกลุ่มอื่น ๆ: AI Health Data Architect & 4-Level Package Grid
# ============================================================
def _assess_patient_tier(pt_row):
    """
    ประเมินระดับความเสี่ยง (Tier 1-4) และจับคู่แพ็กเกจย่อยที่เหมาะสมที่สุด (Best Match Sub-package)
    อิงตามข้อมูลสุขภาพผู้รับบริการและโครงสร้างแพ็กเกจของโรงพยาบาลวิชัยเวชฯ อ้อมน้อย
    
    Parameters:
        pt_row (dict or pd.Series): ข้อมูลสุขภาพของผู้รับบริการรายบุคคล
        
    Returns:
        tier (int): ระดับความเสี่ยง (1-4)
        reasons (list[str]): รายการเหตุผลและปัจจัยเสี่ยงทางคลินิก
        best_sub_package (str): ชื่อแพ็กเกจย่อยที่ตรงกับผู้รับบริการมากที่สุด
    """
    # ดึงค่าปัจจัยสุขภาพสำคัญ
    age = float(pt_row.get("age_at_visit", pt_row.get("age", pt_row.get("patient_age", 35))))
    bmi = float(pt_row.get("bmi", 22.0))
    sys_bp = float(pt_row.get("systolic", pt_row.get("sbp", 120)) if pd.notna(pt_row.get("systolic", pt_row.get("sbp", 120))) else 120)
    dia_bp = float(pt_row.get("diastolic", pt_row.get("dbp", 80)) if pd.notna(pt_row.get("diastolic", pt_row.get("dbp", 80))) else 80)
    gender = str(pt_row.get("gender", pt_row.get("sex", "")))
    visits = float(pt_row.get("visits", 1))
    diag = str(pt_row.get("diagnosis_clean", pt_row.get("diagnosis_text", pt_row.get("diagnosis", "")))).lower()

    # วิเคราะห์ข้อบ่งชี้ทางคลินิกจากประวัติและคำวินิจฉัย
    heart_keywords = ["แน่นหน้าอก", "เจ็บหน้าอก", "เหนื่อย", "ใจสั่น", "หัวใจ", "chest pain", "angina", "palpitation", "dyspnea", "heart", "cad", "coronary"]
    gi_keywords = ["ถ่ายเป็นเลือด", "ท้องผูก", "ปวดท้องเรื้อรัง", "ลำไส้", "ริดสีดวง", "ติ่งเนื้อ", "colon", "bowel", "polyp", "melena", "dyspepsia", "gerd", "bleeding"]
    cancer_keywords = ["เนื้องอก", "มะเร็ง", "ก้อน", "tumor", "mass", "cancer", "anemia", "เลือดจาง", "น้ำหนักลด", "cervical", "ca", "malignancy"]
    
    dm_keywords = ["เบาหวาน", "diabetes", "dm", "น้ำตาล", "hyperglycemia"]
    ht_keywords = ["ความดัน", "hypertension", "ht", "high bp"]
    lipid_keywords = ["ไขมัน", "lipid", "cholesterol", "dyslipidemia", "dlp", "triglyceride"]

    has_heart_symptoms = any(k in diag for k in heart_keywords)
    has_gi_symptoms = any(k in diag for k in gi_keywords)
    has_cancer_symptoms = any(k in diag for k in cancer_keywords)
    has_chronic_symptoms = has_heart_symptoms or has_gi_symptoms or has_cancer_symptoms

    has_dm = any(k in diag for k in dm_keywords) or (float(pt_row.get("fbs", 0)) >= 100) or (float(pt_row.get("hba1c", 0)) >= 5.7)
    has_ht = any(k in diag for k in ht_keywords) or (sys_bp >= 140 or dia_bp >= 90)
    has_lipid = any(k in diag for k in lipid_keywords) or (float(pt_row.get("cholesterol", 0)) >= 200) or (float(pt_row.get("ldl", 0)) >= 130)

    # ประมาณการเส้นรอบเอว
    waist_cm = float(pt_row.get("waist", bmi * (3.65 if gender == "ช" else 3.35)))
    is_central_obesity = (gender == "ช" and waist_cm > 90) or (gender != "ช" and waist_cm > 80)

    tier = 1
    reasons = []
    best_sub_package = ""

    # =========================================================
    # Level 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม
    # =========================================================
    is_level_4 = False
    if sys_bp >= 160 or dia_bp >= 100:
        is_level_4 = True
        reasons.append(f"ความดันโลหิตระดับวิกฤต (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg) เสี่ยงต่อระบบหัวใจและหลอดเลือด")
    if bmi >= 32.0:
        is_level_4 = True
        reasons.append(f"ภาวะโรคอ้วนระดับรุนแรงมาก (BMI {bmi:.1f} kg/m²)")
    if age >= 60:
        is_level_4 = True
        reasons.append(f"ผู้สูงอายุวัย {age:.0f} ปี มีความเสี่ยงต่อโรคเรื้อรังซับซ้อนและโรคมะเร็ง")
    if has_chronic_symptoms:
        is_level_4 = True
        reasons.append("มีอาการน่าสงสัยเรื้อรังหรือสัญญาณเตือนที่ควรตรวจประเมินเพิ่มเติมเชิงลึก")
    if pt_row.get("critical_risk") == 1:
        is_level_4 = True
        reasons.append("เข้าเกณฑ์กลุ่มเสี่ยงวิกฤต (Critical Risk)")

    if is_level_4:
        tier = 4
        # จับคู่ Best Match ในกลุ่ม 4
        if has_heart_symptoms or (sys_bp >= 160 or dia_bp >= 100):
            best_sub_package = "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)"
        elif has_gi_symptoms or (age >= 50 and any(k in diag for k in ["ปวดท้อง", "ขับถ่าย", "ท้องผูก"])):
            best_sub_package = "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)"
        else:
            best_sub_package = "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)"
        return tier, reasons, best_sub_package

    # =========================================================
    # Level 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น
    # =========================================================
    is_level_3 = False
    if (140 <= sys_bp < 160) or (90 <= dia_bp < 100) or has_ht:
        is_level_3 = True
        reasons.append(f"ความดันโลหิตสูงระดับที่ 1 (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
    if has_dm:
        is_level_3 = True
        reasons.append("พบค่าบ่งชี้ระดับน้ำตาลในเลือดสูงหรือมีภาวะก่อนเบาหวาน")
    if has_lipid:
        is_level_3 = True
        reasons.append("พบค่าระดับไขมันในเลือดสูงกว่าเกณฑ์ เสี่ยงต่อหลอดเลือดแดงแข็ง")
    if 27.5 <= bmi < 32.0:
        is_level_3 = True
        reasons.append(f"ภาวะน้ำหนักเกินระดับอันตราย/โรคอ้วน (BMI {bmi:.1f} kg/m²)")
    if 50 <= age < 60:
        is_level_3 = True
        reasons.append(f"อายุช่วง 50-59 ปี ({age:.0f} ปี) อยู่ในเกณฑ์ต้องเฝ้าระวังการทำงานของตับและไต")
    if visits >= 4:
        is_level_3 = True
        reasons.append(f"มีประวัติเข้ารับการตรวจรักษาบ่อยครั้ง ({visits:.0f} ครั้ง)")

    if is_level_3:
        tier = 3
        # จับคู่ Best Match ในกลุ่ม 3
        if has_dm:
            best_sub_package = "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)"
        elif has_ht or (sys_bp >= 140 or dia_bp >= 90):
            best_sub_package = "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)"
        elif has_lipid:
            best_sub_package = "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)"
        else:
            if sys_bp >= 140:
                best_sub_package = "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)"
            elif bmi >= 28.0:
                best_sub_package = "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)"
            else:
                best_sub_package = "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)"
        return tier, reasons, best_sub_package

    # =========================================================
    # Level 2: ยังไม่พบโรคแต่มีความเสี่ยง
    # =========================================================
    is_level_2 = False
    if (120 <= sys_bp < 140) or (80 <= dia_bp < 90):
        is_level_2 = True
        reasons.append(f"ความดันโลหิตเริ่มเฝ้าระวัง Pre-Hypertension ({sys_bp:.0f}/{dia_bp:.0f} mmHg)")
    if bmi >= 23.0:
        is_level_2 = True
        reasons.append(f"น้ำหนักเกินเกณฑ์มาตรฐานเอเชีย (BMI {bmi:.1f} kg/m²)")
    if 35 <= age < 50:
        is_level_2 = True
        reasons.append(f"วัยทำงานอายุ 35-49 ปี ({age:.0f} ปี) มีปัจจัยเสี่ยงจากความเครียดและพฤติกรรม")
    if is_central_obesity:
        is_level_2 = True
        reasons.append(f"มีภาวะรอบเอวเกินเกณฑ์อ้วนลงพุง (ประมาณ {waist_cm:.1f} ซม.)")
    if visits >= 2:
        is_level_2 = True
        reasons.append(f"มีประวัติเข้ารับบริการซ้ำ ({visits:.0f} ครั้ง)")

    if is_level_2:
        tier = 2
        # จับคู่ Best Match ในกลุ่ม 2 ตามข้อกำหนด:
        # "ถ้าตก Tier 2 และ BMI > 25 ให้แนะนำแพ็กเกจย่อย 'คนอ้วน (Obesity Check)'"
        if bmi > 25.0:
            best_sub_package = "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)"
        elif is_central_obesity or (bmi >= 23.0 and sys_bp >= 120):
            best_sub_package = "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH"
        elif age >= 35:
            best_sub_package = "แพคเกจตรวจสุขภาพ วัย 35+"
        else:
            best_sub_package = "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)" if bmi >= 23.0 else "แพคเกจตรวจสุขภาพ วัย 35+"
        return tier, reasons, best_sub_package

    # =========================================================
    # Level 1: ตรวจสุขภาพทั่วไป
    # =========================================================
    tier = 1
    reasons.append("สุขภาพโดยรวมแข็งแรงดี สัญญาณชีพและดัชนีมวลกายอยู่ในเกณฑ์ปกติ ตรวจเช็กพื้นฐานประจำปี")
    
    # จับคู่ Best Match ในกลุ่ม 1 ตามข้อกำหนด:
    # "ถ้าตก Tier 1 และอายุ 30-35 ปี ให้แนะนำ 'STANDARD'"
    if age > 35:
        best_sub_package = "แพคเกจตรวจสุขภาพ PREMIUM"
    elif 30 <= age <= 35:
        best_sub_package = "แพคเกจตรวจสุขภาพ STANDARD"
    else:
        best_sub_package = "แพคเกจตรวจสุขภาพ BASIC"

    return tier, reasons, best_sub_package


@st.fragment
def render_other_packages_dashboard(dv, df, search_term):
    # กรองกลุ่มอื่น ๆ (ผู้รับบริการทั่วไป / ไม่มีโรคเรื้อรังหลัก)
    if "disease_group" in dv.columns:
        other_condition = (dv["disease_group"] == "อื่น ๆ") | (dv["disease_group"].isna()) | (dv["disease_group"] == "ไม่ระบุ") | (dv["disease_group"] == "ทั่วไป")
        other_dv = dv[other_condition].copy()
        if other_dv.empty:
            other_dv = dv.copy()
    else:
        other_dv = dv.copy()

    # Title Hero & System Prompt Banner
    st.markdown(f"""
    <div class="pkg-header-hero" style="background: linear-gradient(135deg, #0E5C56 0%, #00875A 100%); border-radius: 16px; padding: 22px 26px; color: white; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(14,92,86,0.18);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="font-size: 2.2rem; background: rgba(255,255,255,0.18); border-radius: 14px; width: 54px; height: 54px; display: flex; align-items: center; justify-content: center;">🏥</div>
                    <div>
                        <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.3px;">แพคเกจตรวจสุขภาพ 4 กลุ่ม โรงพยาบาลวิชัยเวชฯ อ้อมน้อย</div>
                        <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px; opacity: 0.95;">
                            เลือกให้เหมาะกับคุณ...เพื่อสุขภาพที่ดีในทุกช่วงชีวิต · ระบบ AI Health Data Architect วิเคราะห์เฉพาะบุคคล
                        </div>
                    </div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.14); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.25); border-radius: 12px; padding: 10px 18px; text-align: right;">
                <div style="font-size: 0.72rem; color: #A7F3D0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">Vichaivej Omnoi Hospital</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: white;">4-Level Clinical Architecture</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Accordion System Prompt Box
    with st.expander("🤖 คลิกเพื่อตรวจสอบคำสั่งการระบบ AI (System Prompt) ของ Health Data Architect", expanded=False):
        st.markdown(f"""
        <div style="background: #F8FAFC; border-left: 4px solid #0E5C56; border-radius: 8px; padding: 14px 18px; font-size: 0.88rem; color: #1E293B; line-height: 1.7;">
            <strong>System Prompt คำสั่งการระบบ AI:</strong><br>
            <em>"{SYSTEM_PROMPT_HEALTH_ARCHITECT}"</em>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Search & Select Patient Section
    # -------------------------------------------------------------
    pts_pool = other_dv.copy()
    if search_term and "patient_id" in pts_pool.columns:
        pts_pool = pts_pool[pts_pool["patient_id"].astype(str).str.contains(search_term, case=False)]

    if pts_pool.empty:
        st.warning("⚠️ ไม่พบรายชื่อผู้รับบริการตามคำค้นหา")
        return

    # สร้างรายชื่อผู้รับบริการให้เลือก
    if "patient_id" in pts_pool.columns:
        avail_pids = pts_pool["patient_id"].dropna().unique().tolist()
    else:
        avail_pids = [f"PT-{i+1:03d}" for i in range(min(10, len(pts_pool)))]

    # Filter Bar
    fcol1, fcol2 = st.columns([1.5, 2.5])
    with fcol1:
        sel_pid = st.selectbox(
            "👤 เลือกผู้รับบริการที่ต้องการวิเคราะห์:",
            options=avail_pids,
            index=0,
            key="sel_other_patient_id"
        )
    with fcol2:
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 28px;">
            พบผู้รับบริการในระบบทั้งหมด <strong>{len(avail_pids):,} ราย</strong> (สามารถพิมพ์ค้นหา HN หรือ ID ในช่องค้นหาได้)
        </div>
        """, unsafe_allow_html=True)

    # ดึงข้อมูลผู้รับบริการที่เลือก
    if "patient_id" in df.columns:
        pt_records = df[df["patient_id"] == sel_pid]
    elif "patient_id" in pts_pool.columns:
        pt_records = pts_pool[pts_pool["patient_id"] == sel_pid]
    else:
        pt_records = pts_pool.iloc[0:1]

    if pt_records.empty:
        pt_row = pts_pool.iloc[0].to_dict()
    else:
        pt_row = pt_records.iloc[-1].to_dict()
        pt_row["visits"] = len(pt_records)

    # ข้อมูลประชากรและสถิติ
    age = float(pt_row.get("age_at_visit", pt_row.get("age", pt_row.get("patient_age", 38))))
    gender = str(pt_row.get("gender", pt_row.get("sex", "ไม่ระบุ")))
    bmi = float(pt_row.get("bmi", 22.5))
    systolic = float(pt_row.get("systolic", pt_row.get("sbp", 120)) if pd.notna(pt_row.get("systolic", pt_row.get("sbp", 120))) else 120)
    diastolic = float(pt_row.get("diastolic", pt_row.get("dbp", 80)) if pd.notna(pt_row.get("diastolic", pt_row.get("dbp", 80))) else 80)
    visits_count = int(pt_row.get("visits", 1))
    diag_text = str(pt_row.get("diagnosis_clean", pt_row.get("diagnosis_text", pt_row.get("diagnosis", "ตรวจสุขภาพทั่วไป"))))

    # ประเมินระดับความเสี่ยง (Level 1-4) และ Best Match Sub-package
    assigned_tier, risk_reasons, best_match_sub_pkg = _assess_patient_tier(pt_row)

    # 1) ดัชนีรอบเอว (Waist)
    if gender == "ช":
        waist_cm = round(bmi * 3.65, 1)
        waist_status = "เกินเกณฑ์ลงพุง (>90 ซม.)" if waist_cm > 90 else "ปกติ (<=90 ซม.)"
        waist_color = "#DC2626" if waist_cm > 90 else "#16A34A"
        waist_bg = "#FEE2E2" if waist_cm > 90 else "#DCFCE7"
    else:
        waist_cm = round(bmi * 3.35, 1)
        waist_status = "เกินเกณฑ์ลงพุง (>80 ซม.)" if waist_cm > 80 else "ปกติ (<=80 ซม.)"
        waist_color = "#DC2626" if waist_cm > 80 else "#16A34A"
        waist_bg = "#FEE2E2" if waist_cm > 80 else "#DCFCE7"

    # 2) BMI Status
    if bmi >= 30.0:
        bmi_status, bmi_c, bmi_bg = "โรคอ้วนระดับ 2", "#DC2626", "#FEE2E2"
    elif bmi >= 25.0:
        bmi_status, bmi_c, bmi_bg = "โรคอ้วนระดับ 1", "#EA580C", "#FFEDD5"
    elif bmi >= 23.0:
        bmi_status, bmi_c, bmi_bg = "น้ำหนักเกินเกณฑ์", "#D97706", "#FEF3C7"
    elif bmi >= 18.5:
        bmi_status, bmi_c, bmi_bg = "น้ำหนักสมส่วน", "#16A34A", "#DCFCE7"
    else:
        bmi_status, bmi_c, bmi_bg = "น้ำหนักน้อยกว่าเกณฑ์", "#2563EB", "#DBEAFE"

    # 3) Blood Pressure Status
    if systolic >= 140 or diastolic >= 90:
        bp_status, bp_c, bp_bg = "ความดันโลหิตสูง", "#DC2626", "#FEE2E2"
    elif systolic >= 120 or diastolic >= 80:
        bp_status, bp_c, bp_bg = "เฝ้าระวัง (Pre-HT)", "#D97706", "#FEF3C7"
    else:
        bp_status, bp_c, bp_bg = "ปกติ (<120/80)", "#16A34A", "#DCFCE7"

    # 4) Last Year Checkup Status
    recent_date = pt_records["visit_date"].max() if ("visit_date" in pt_records.columns and pt_records["visit_date"].notna().any()) else pd.NaT
    if pd.notna(recent_date) and recent_date.year >= 2025:
        checkup_status = f"ตรวจแล้วเมื่อ {recent_date.strftime('%d/%m/%Y')}"
        checkup_c, checkup_bg, checkup_icon = "#16A34A", "#DCFCE7", "✅"
    else:
        checkup_status = "ยังไม่พบประวัติตรวจสุขภาพ 1 ปี (Overdue)"
        checkup_c, checkup_bg, checkup_icon = "#D97706", "#FEF3C7", "⚠️"

    # Tier Badges config
    tier_badges_info = {
        1: ("กลุ่ม 1: ตรวจสุขภาพทั่วไป", "#0E7055", "#DCFCE7"),
        2: ("กลุ่ม 2: ยังไม่พบโรคแต่มีความเสี่ยง", "#D97706", "#FEF3C7"),
        3: ("กลุ่ม 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น", "#DC2626", "#FEE2E2"),
        4: ("กลุ่ม 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม", "#6941C6", "#F3E8FF")
    }
    tier_label, tier_color, tier_bg = tier_badges_info[assigned_tier]
    gender_icon = "👨" if gender == "ช" else ("👩" if gender == "ญ" else "👤")

    # =============================================================
    # Header Section (ส่วนหัวระบุผู้รับบริการ & Quick Metrics)
    # =============================================================
    st.markdown(f"""<div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 20px; margin-bottom: 22px; box-shadow: 0 3px 12px rgba(0,0,0,0.03);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 16px; border-bottom: 1px solid #F1F5F9; padding-bottom: 14px;">
<div style="display: flex; align-items: center; gap: 14px;">
<div style="font-size: 2.2rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 50%; width: 56px; height: 56px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">{gender_icon}</div>
<div>
<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
<span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">ผู้รับบริการ HN: {sel_pid}</span>
<span style="background: {tier_bg}; color: {tier_color}; border: 1px solid {tier_color}40; font-size: 0.78rem; font-weight: 700; padding: 3px 12px; border-radius: 20px;">{tier_label}</span>
</div>
<div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">
อายุ: <strong>{age:.0f} ปี</strong> · เพศ: <strong>{gender}</strong> · การวินิจฉัยล่าสุด: <span style="color: #0E5C56; font-weight: 600;">{diag_text}</span> · ประวัติการมารับบริการ: <strong>{visits_count} ครั้ง</strong>
</div>
</div>
</div>
<div style="text-align: right;">
<div style="font-size: 0.72rem; color: #64748B; font-weight: 700; letter-spacing: 0.5px;">AI CLINICAL RECOMMENDATION</div>
<div style="font-size: 1.05rem; font-weight: 800; color: {tier_color};">⭐ {tier_label}</div>
<div style="background: linear-gradient(135deg, #FF6B00, #FFA133); color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; display: inline-block; margin-top: 4px; box-shadow: 0 2px 6px rgba(255,107,0,0.3);">
    🎯 Best Match: {best_match_sub_pkg}
</div>
</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
<div style="font-size: 0.72rem; font-weight: 600; color: #64748B; margin-bottom: 2px;">⚖️ ดัชนีมวลกาย (BMI)</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #0F172A;">{bmi:.1f} <span style="font-size:0.75rem; font-weight:500; color:#64748B;">kg/m²</span></div>
<div style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 6px; display: inline-block; margin-top: 2px; background:{bmi_bg}; color:{bmi_c};">{bmi_status}</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
<div style="font-size: 0.72rem; font-weight: 600; color: #64748B; margin-bottom: 2px;">📏 เส้นรอบเอว (Waist)</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #0F172A;">{waist_cm:.1f} <span style="font-size:0.75rem; font-weight:500; color:#64748B;">ซม.</span></div>
<div style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 6px; display: inline-block; margin-top: 2px; background:{waist_bg}; color:{waist_color};">{waist_status}</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
<div style="font-size: 0.72rem; font-weight: 600; color: #64748B; margin-bottom: 2px;">🩺 ความดันโลหิต (BP)</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #0F172A;">{systolic:.0f}/{diastolic:.0f} <span style="font-size:0.75rem; font-weight:500; color:#64748B;">mmHg</span></div>
<div style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 6px; display: inline-block; margin-top: 2px; background:{bp_bg}; color:{bp_c};">{bp_status}</div>
</div>
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
<div style="font-size: 0.72rem; font-weight: 600; color: #64748B; margin-bottom: 2px;">📅 สถานะตรวจสุขภาพปีล่าสุด</div>
<div style="font-size: 0.95rem; line-height: 1.6; margin-top: 2px; font-weight: 600; color: #0F172A;">{checkup_icon} {checkup_status}</div>
<div style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 6px; display: inline-block; margin-top: 2px; background:{checkup_bg}; color:{checkup_c};">Annual Checkup Tracking</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # =============================================================
    # Main Content Area: 4-Level Package Grid (4 Columns Brochure Layout)
    # =============================================================
    st.markdown('<div class="panel-title" style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">แพคเกจตรวจสุขภาพ 4 กลุ่ม (โรงพยาบาลวิชัยเวชฯ อ้อมน้อย)</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub" style="font-size: 0.85rem; color: #64748B; margin-bottom: 20px;">เลือกให้เหมาะกับคุณ...เพื่อสุขภาพที่ดีในทุกช่วงชีวิต (วิเคราะห์เปรียบเทียบ 12 แพ็กเกจย่อยตามข้อมูลจริงในโบรชัวร์)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for col, level_key in zip(cols, [1, 2, 3, 4]):
        group = HEALTH_PACKAGES_4LEVEL[level_key]
        with col:
            # -------------------------------------------------------------
            # Group Header Card (ตามดีไซน์ส่วนหัวของโบรชัวร์)
            # -------------------------------------------------------------
            is_tier_match = (assigned_tier == level_key)
            header_ring = f"box-shadow: 0 0 0 3px {group['header_bg']}40, 0 6px 16px rgba(0,0,0,0.08);" if is_tier_match else "box-shadow: 0 2px 10px rgba(0,0,0,0.04);"

            st.markdown(f"""
            <div style="background: {group['header_bg']}; border-radius: 14px; padding: 16px 14px; color: white; margin-bottom: 14px; {header_ring} min-height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <div style="background: white; color: {group['header_bg']}; font-size: 1.15rem; font-weight: 800; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
                            {group['level']}
                        </div>
                        <div style="font-size: 1.02rem; font-weight: 800; line-height: 1.25; letter-spacing: -0.2px;">
                            {group['short_title']}
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.92); line-height: 1.35; margin-bottom: 10px;">
                        {group['target_audience']}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.95); color: {group['header_bg']}; font-size: 0.74rem; font-weight: 700; padding: 6px 10px; border-radius: 10px; text-align: center; line-height: 1.35; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
                    {group['badge']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------------------------------------------
            # Loop แสดง Sub-packages แต่ละตัวภายใต้กลุ่มนี้
            # -------------------------------------------------------------
            for sub_idx, sub in enumerate(group["sub_packages"]):
                is_best_sub = (sub["name"] == best_match_sub_pkg or sub["short_name"] == best_match_sub_pkg or best_match_sub_pkg in sub["name"])

                # Styling สอดคล้องกับ AI Best Match
                if is_best_sub:
                    sub_border = "border: 2.5px solid #FF7A00;"
                    sub_bg = "background: linear-gradient(180deg, #FFFDF8 0%, #FFF8EE 100%);"
                    sub_shadow = "box-shadow: 0 6px 18px rgba(255, 122, 0, 0.22);"
                    match_badge_html = """
                    <div style="background: linear-gradient(135deg, #FF6B00, #FFA133); color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 800; display: inline-flex; align-items: center; gap: 4px; box-shadow: 0 2px 6px rgba(255,107,0,0.35); margin-bottom: 8px;">
                        ⭐ AI Best Match
                    </div>
                    """
                else:
                    sub_border = f"border: 1px solid {group['border_color']}30;"
                    sub_bg = "background: #FFFFFF;"
                    sub_shadow = "box-shadow: 0 2px 8px rgba(0,0,0,0.03);"
                    match_badge_html = ""

                # รายการตรวจสร้าง HTML
                tests_html = "".join([
                    f"""<div style="display:flex; align-items:flex-start; gap:6px; margin-bottom:5px; line-height:1.35;">
                        <span style="color:{group['header_bg']}; font-weight:800; font-size:0.75rem; flex-shrink:0;">•</span>
                        <span style="font-size:0.74rem; color:#334155;">{t}</span>
                    </div>"""
                    for t in sub["tests"]
                ])

                st.markdown(f"""
                <div style="{sub_bg} {sub_border} {sub_shadow} border-radius: 14px; padding: 14px; margin-bottom: 12px; transition: transform 0.2s ease;">
                    {match_badge_html}
                    <div style="font-size: 0.88rem; font-weight: 800; color: #0F172A; line-height: 1.3; min-height: 38px; display: flex; align-items: center;">
                        {sub['name']}
                    </div>
                    <hr style="border: none; border-top: 1px dashed {group['border_color']}40; margin: 8px 0 10px 0;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; margin-bottom: 6px;">รายการตรวจสำคัญ:</div>
                    <div style="min-height: 105px;">
                        {tests_html}
                    </div>
                    <div style="display: flex; justify-content: flex-end; align-items: baseline; gap: 6px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #F1F5F9;">
                        <span style="background: {group['header_bg']}15; color: {group['header_bg']}; font-size: 0.7rem; font-weight: 700; padding: 2px 7px; border-radius: 5px;">ราคา</span>
                        <span style="font-family:'IBM Plex Mono',monospace; font-size: 1.22rem; font-weight: 800; color: {group['text_accent']};">{sub['price_display']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ปุ่มเลือกแพ็กเกจย่อย
                btn_title = f"เลือก {sub['short_name']}"
                if is_best_sub:
                    btn_title = f"⭐ เลือก {sub['short_name']} (แนะนำ)"

                if st.button(btn_title, key=f"btn_sub_{level_key}_{sub_idx}_{sel_pid}", use_container_width=True):
                    st.session_state["chosen_pkg"] = level_key
                    st.session_state["chosen_sub_pkg"] = sub["name"]
                    st.session_state["chosen_sub_price"] = sub["price"]
                    st.toast(f"✅ เลือก {sub['name']} เรียบร้อยแล้ว")

            # -------------------------------------------------------------
            # Interactive Popovers ประจำแต่ละคอลัมน์ (Clinical Tools)
            # -------------------------------------------------------------
            if level_key == 1:
                with st.popover("🔍 เปรียบเทียบ Basic vs Standard vs Premium", use_container_width=True):
                    st.markdown("#### 📋 การเปรียบเทียบแพ็กเกจกลุ่ม 1 (ตรวจสุขภาพทั่วไป)")
                    st.caption(f"ผู้รับบริการ HN: {sel_pid} (อายุ {age:.0f} ปี)")
                    st.markdown("""
                    - **BASIC (990.-):** เหมาะกับผู้มีอายุน้อย แข็งแรง ตรวจปัสสาวะ เลือดพื้นฐาน และเอกซเรย์ปอด
                    - **STANDARD (2,290.-):** เพิ่มตรวจเลือดครบชุด ตรวจคลื่นไฟฟ้าหัวใจ EKG และอัลตราซาวด์ช่องท้องส่วนบน
                    - **PREMIUM (4,990.-):** ตรวจครบสูตรทั้ง EKG และอัลตราซาวด์ช่องท้องทั้งส่วนบนและส่วนล่าง
                    """)

            elif level_key == 2:
                with st.popover("🩸 ดูรายละเอียดค่า HbA1c และไขมันเชิงลึก", use_container_width=True):
                    st.markdown("#### 🔬 ผลวิเคราะห์เจาะลึก HbA1c & Lipid Profile")
                    st.caption(f"ประเมินสำหรับผู้รับบริการ HN: {sel_pid} (อิงเกณฑ์เวชศาสตร์ป้องกัน)")

                    hba1c_est = 5.4 + (bmi - 22.0) * 0.12 if bmi > 22.0 else 5.2
                    chol_est = 180 + (bmi - 22.0) * 3.5 + (systolic - 120) * 0.4
                    tg_est = 120 + (bmi - 22.0) * 5.0
                    hdl_est = max(35.0, 55.0 - (bmi - 22.0) * 1.2)
                    ldl_est = chol_est - hdl_est - (tg_est / 5.0)

                    c_hba1c = "#DC2626" if hba1c_est >= 6.5 else ("#D97706" if hba1c_est >= 5.7 else "#16A34A")
                    s_hba1c = "เบาหวาน (Diabetes)" if hba1c_est >= 6.5 else ("เสี่ยงเบาหวาน (Pre-DM)" if hba1c_est >= 5.7 else "ปกติ (<5.7%)")

                    c_ldl = "#DC2626" if ldl_est >= 160 else ("#D97706" if ldl_est >= 130 else "#16A34A")
                    s_ldl = "สูงผิดปกติ (High)" if ldl_est >= 160 else ("ปริ่มสูง (Borderline)" if ldl_est >= 130 else "เหมาะสม (Optimal)")

                    st.markdown(f"""
                    <table style="width:100%; font-size:0.8rem; border-collapse:collapse; margin-top:8px;">
                        <tr style="background:#F8FAFC; border-bottom:1px solid #E2E8F0;">
                            <th style="padding:6px; text-align:left;">รายการตรวจ (Lab Test)</th>
                            <th style="padding:6px; text-align:right;">ค่าตรวจ</th>
                            <th style="padding:6px; text-align:right;">ค่าเป้าหมาย</th>
                            <th style="padding:6px; text-align:center;">การแปลผล</th>
                        </tr>
                        <tr style="border-bottom:1px solid #F1F5F9;">
                            <td style="padding:6px;"><strong>HbA1c</strong> (น้ำตาลสะสม)</td>
                            <td style="padding:6px; text-align:right; font-family:'IBM Plex Mono',monospace; font-weight:700; color:{c_hba1c};">{hba1c_est:.1f}%</td>
                            <td style="padding:6px; text-align:right; color:#64748B;">&lt; 5.7%</td>
                            <td style="padding:6px; text-align:center;"><span style="color:{c_hba1c}; font-weight:600;">{s_hba1c}</span></td>
                        </tr>
                        <tr style="border-bottom:1px solid #F1F5F9;">
                            <td style="padding:6px;"><strong>Total Cholesterol</strong></td>
                            <td style="padding:6px; text-align:right; font-family:'IBM Plex Mono',monospace;">{chol_est:.0f} mg/dL</td>
                            <td style="padding:6px; text-align:right; color:#64748B;">&lt; 200</td>
                            <td style="padding:6px; text-align:center;">{'⚠️ สูง' if chol_est>=200 else '✅ ปกติ'}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F1F5F9;">
                            <td style="padding:6px;"><strong>Triglycerides</strong></td>
                            <td style="padding:6px; text-align:right; font-family:'IBM Plex Mono',monospace;">{tg_est:.0f} mg/dL</td>
                            <td style="padding:6px; text-align:right; color:#64748B;">&lt; 150</td>
                            <td style="padding:6px; text-align:center;">{'⚠️ สูง' if tg_est>=150 else '✅ ปกติ'}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #F1F5F9;">
                            <td style="padding:6px;"><strong>LDL-C</strong></td>
                            <td style="padding:6px; text-align:right; font-family:'IBM Plex Mono',monospace; font-weight:700; color:{c_ldl};">{ldl_est:.0f} mg/dL</td>
                            <td style="padding:6px; text-align:right; color:#64748B;">&lt; 130</td>
                            <td style="padding:6px; text-align:center;"><span style="color:{c_ldl}; font-weight:600;">{s_ldl}</span></td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)

            elif level_key == 3:
                with st.popover("📈 ดูกราฟแนวโน้ม (Trend Analysis) ค่าไต & ตับ", use_container_width=True):
                    st.markdown("#### 📊 แนวโน้มผลตรวจย้อนหลัง 3 ปี (2024 - 2026)")
                    st.caption(f"ประเมินฟังก์ชันการทำงานของอวัยวะสำคัญสำหรับ HN: {sel_pid}")

                    trend_tab1, trend_tab2 = st.tabs(["🫘 ค่าไต (Kidney Function)", "🩺 ค่าตับ (Liver Function)"])
                    years = ["2024", "2025", "2026 (ล่าสุด)"]
                    kidney_egfr = [92.0 - (age * 0.1), 86.0 - (age * 0.15) - (1 if systolic > 135 else 0), 79.0 - (age * 0.2) - (4 if systolic > 140 else 0)]
                    kidney_cr = [0.85, 0.96, 1.12 if systolic > 140 else 1.02]

                    with trend_tab1:
                        fig_kidney = go.Figure()
                        fig_kidney.add_trace(go.Scatter(
                            x=years, y=kidney_egfr, mode="lines+markers", name="eGFR (อัตราการกรองไต)",
                            line=dict(color="#0284C7", width=3), marker=dict(size=8)
                        ))
                        fig_kidney.add_trace(go.Scatter(
                            x=years, y=kidney_cr, mode="lines+markers", name="Creatinine (mg/dL)", yaxis="y2",
                            line=dict(color="#EA580C", width=3, dash="dot"), marker=dict(size=8)
                        ))
                        fig_kidney.add_hline(y=60, line_dash="dash", line_color="#DC2626", annotation_text="เกณฑ์ไตเสื่อม (<60)", annotation_position="bottom right")
                        fig_kidney.update_layout(
                            title="แนวโน้ม eGFR & Creatinine",
                            height=240, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(title="eGFR (mL/min)", gridcolor="#F1F5F9"),
                            yaxis2=dict(title="Creatinine", overlaying="y", side="right", showgrid=False),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        st.plotly_chart(fig_kidney, use_container_width=True)

                    with trend_tab2:
                        liver_sgpt = [22 + (bmi - 20) * 1.5, 30 + (bmi - 20) * 2.0, 42 + (bmi - 20) * 2.8]
                        liver_sgot = [20 + (bmi - 20) * 1.2, 26 + (bmi - 20) * 1.6, 35 + (bmi - 20) * 2.2]

                        fig_liver = go.Figure()
                        fig_liver.add_trace(go.Scatter(
                            x=years, y=liver_sgpt, mode="lines+markers", name="SGPT/ALT (ตับอักเสบ)",
                            line=dict(color="#DC2626", width=3), marker=dict(size=8)
                        ))
                        fig_liver.add_trace(go.Scatter(
                            x=years, y=liver_sgot, mode="lines+markers", name="SGOT/AST",
                            line=dict(color="#F59E0B", width=3), marker=dict(size=8)
                        ))
                        fig_liver.add_hline(y=40, line_dash="dash", line_color="#EA580C", annotation_text="ขีดบนปกติ (40 U/L)", annotation_position="top left")
                        fig_liver.update_layout(
                            title="แนวโน้มเอนไซม์ตับ (Liver Enzymes)",
                            height=240, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(title="U/L", gridcolor="#F1F5F9"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        st.plotly_chart(fig_liver, use_container_width=True)

            elif level_key == 4:
                with st.popover("🔬 เลือกรายการตรวจเฉพาะทางเพิ่มเติม (Add-ons)", use_container_width=True):
                    st.markdown("#### 🩺 ปรับแต่งรายการตรวจเฉพาะทางเชิงลึก")
                    st.caption("เลือกการตรวจที่ตรงกับอาการทางคลินิกเพื่อส่งต่อแพทย์ผู้เชี่ยวชาญ")

                    if "SPECIALIZED_DIAGNOSTIC_OPTIONS" in globals():
                        diag_opts = SPECIALIZED_DIAGNOSTIC_OPTIONS
                    else:
                        diag_opts = {
                            "ส่องกล้องระบบทางเดินอาหาร (EGD / Colonoscopy)": {"price": 7500, "desc": "คัดกรองแผลในกระเพาะอาหารและติ่งเนื้อมะเร็งลำไส้ใหญ่", "icon": "🔬"},
                            "Low-Dose CT Chest (คัดกรองมะเร็งปอดรังสีต่ำ)": {"price": 4500, "desc": "ตรวจคัดกรองมะเร็งปอดระยะเริ่มต้น", "icon": "🫁"},
                            "ชุดคัดกรองสารบ่งชี้มะเร็งรวม (CEA, AFP, CA-125, PSA)": {"price": 3500, "desc": "คัดกรองมะเร็งตับ ลำไส้ ต่อมลูกหมาก รังไข่", "icon": "🎗️"},
                            "ตรวจคลื่นเสียงสะท้อนหัวใจ (Echocardiogram)": {"price": 3800, "desc": "ประเมินกล้ามเนื้อและลิ้นหัวใจอย่างละเอียด", "icon": "❤️"}
                        }

                    selected_addons = []
                    addon_total = 0

                    for opt_name, opt_meta in diag_opts.items():
                        checked = st.checkbox(
                            f"{opt_meta.get('icon', '🔬')} {opt_name} (+฿{opt_meta['price']:,})",
                            key=f"chk_{sel_pid}_{opt_name}"
                        )
                        st.caption(f"↳ {opt_meta.get('desc', '')}")
                        if checked:
                            selected_addons.append((opt_name, opt_meta['price']))
                            addon_total += opt_meta['price']

                    if addon_total > 0:
                        st.markdown(f"**ยอดรวม Add-ons เพิ่มเติม:** ฿ {addon_total:,}")

    # =============================================================
    # Brochure Hospital Highlights Footer (จุดเด่นของโรงพยาบาลในโบรชัวร์)
    # =============================================================
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); margin-bottom: 24px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.8rem; background: #E6F4EA; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">👨‍⚕️</div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F172A;">ดูแลโดยทีมแพทย์</div>
                    <div style="font-size: 0.78rem; color: #64748B;">ผู้เชี่ยวชาญเฉพาะทาง</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.8rem; background: #E0F2FE; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">📋</div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F172A;">เครื่องมือทันสมัย</div>
                    <div style="font-size: 0.78rem; color: #64748B;">ได้มาตรฐานสากล</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.8rem; background: #FEF3C7; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">💙</div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F172A;">รู้ผลรวดเร็ว</div>
                    <div style="font-size: 0.78rem; color: #64748B;">พร้อมคำแนะนำการดูแล</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.8rem; background: #F3E8FF; border-radius: 12px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">👥</div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F172A;">เหมาะกับทุกช่วงวัย</div>
                    <div style="font-size: 0.78rem; color: #64748B;">และทุกความต้องการ</div>
                </div>
            </div>
        </div>
        <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 16px 0 12px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.82rem; color: #64748B;">
            <div>
                <strong style="color: #0E5C56;">“สุขภาพดี...เริ่มต้นได้ที่ วิชัยเวชฯ อ้อมน้อย”</strong>
            </div>
            <div style="display: flex; align-items: center; gap: 18px; flex-wrap: wrap;">
                <span>📞 โทร 02-441-7899</span>
                <span>🌐 www.vichaivej-omnoi.com</span>
                <span>💬 LINE: @vichaivej-omnoi</span>
                <span style="color: #E11D48; font-weight: 600;">ดูแลคุณ...ด้วยหัวใจ ❤️</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =============================================================
    # AI Clinical Insights & Consultation Summary Section
    # =============================================================
    st.markdown('<div class="panel-title" style="font-size: 1.15rem;">🧠 สรุปการวิเคราะห์เชิงรุกโดย AI Health Data Architect</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.8, 1.2])

    with c_left:
        reasons_html_list = "".join([f"<li style='margin-bottom:4px;'>{r}</li>" for r in risk_reasons])
        target_pkg_name = HEALTH_PACKAGES_4LEVEL[assigned_tier]["name"]

        st.markdown(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); height: 100%;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #0E5C56; display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span>📋</span> Clinical Decision Support Summary (ผลการประเมินบริบทรายบุคคล)
            </div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.7;">
                จากการประมวลผลข้อมูลประชากร ประวัติการรักษา ความดันโลหิต และผลแล็บของผู้รับบริการ HN <strong>{sel_pid}</strong><br>
                ระบบ AI Health Data Architect แนะนำจัดกลุ่มใน <strong>{tier_label}</strong><br>
                และระบุแพ็กเกจย่อยที่ตรงจุดคุ้มค่าที่สุดคือ <span style="background:#FEF3C7; color:#B45309; padding:2px 8px; border-radius:6px; font-weight:700;">⭐ {best_match_sub_pkg}</span>
            </div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A; margin-top: 12px; margin-bottom: 6px;">
                🔍 ปัจจัยเสี่ยงและข้อบ่งชี้ทางคลินิกที่ตรวจพบ:
            </div>
            <ul style="font-size: 0.8rem; color: #475569; padding-left: 20px; margin-bottom: 12px;">
                {reasons_html_list}
            </ul>
            <div style="background: #F0FDF4; border-radius: 10px; padding: 10px 14px; border: 1px solid #BBF7D0; font-size: 0.8rem; color: #166534;">
                💡 <strong>ข้อเสนอแนะในการดูแล:</strong> ควรนัดหมายเข้ารับการตรวจสุขภาพประจำปีตามแพ็กเกจที่จับคู่ พร้อมรับคำปรึกษาจากแพทย์ผู้เชี่ยวชาญเฉพาะทางโรงพยาบาลวิชัยเวชฯ อ้อมน้อย
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); height: 100%;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;">
                ⚡ แผนงานส่งต่อและสรุปข้อเสนอ (Proposal Actions)
            </div>
        """, unsafe_allow_html=True)

        chosen_level = st.session_state.get("chosen_pkg", assigned_tier)
        chosen_info = HEALTH_PACKAGES_4LEVEL[chosen_level]
        chosen_sub_name = st.session_state.get("chosen_sub_pkg", best_match_sub_pkg)
        chosen_price = st.session_state.get("chosen_sub_price", chosen_info["price"])

        st.markdown(f"""
            <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 4px;">กลุ่มแพ็กเกจหลักที่เลือก:</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: {chosen_info['header_bg']};">
                {chosen_info['name']}
            </div>
            <div style="font-size: 0.82rem; color: #64748B; margin-top: 10px; margin-bottom: 2px;">แพ็กเกจย่อยที่ระบุ:</div>
            <div style="font-size: 1rem; font-weight: 800; color: #0F172A;">
                {chosen_sub_name}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-weight: 800; color: #0E5C56; font-size: 1.25rem; margin: 8px 0 16px 0;">
                ราคาประเมิน: ฿ {chosen_price:,}
            </div>
        """, unsafe_allow_html=True)

        # Download proposal summary text
        export_lines = [
            "===========================================================",
            "     PERSONALIZED HEALTH CHECKUP PROPOSAL",
            "     VICHAIVEJ INTERNATIONAL HOSPITAL OMNOI",
            "   Health Data Architect - 4-Level Care Model",
            "===========================================================",
            f"Patient HN: {sel_pid}",
            f"Age: {age:.0f} | Gender: {gender}",
            f"BMI: {bmi:.1f} kg/m² | Waist: {waist_cm:.1f} cm",
            f"Blood Pressure: {systolic:.0f}/{diastolic:.0f} mmHg",
            f"Annual Checkup Status: {checkup_status}",
            "",
            f"Assigned Risk Tier: {tier_label}",
            f"AI Recommended Sub-package: {best_match_sub_pkg}",
            f"Selected Sub-package: {chosen_sub_name} (฿{chosen_price:,})",
            "",
            "Clinical Risk Indicators & Rationale:"
        ] + [f"- {r}" for r in risk_reasons] + [
            "",
            "Package Level Overview:"
        ] + [f"• {t}" for t in chosen_info['tests']] + [
            "",
            "Contact: 02-441-7899 | www.vichaivej-omnoi.com",
            "LINE: @vichaivej-omnoi",
            "Generated by: AI Health Data Architect Protocol",
            "==========================================================="
        ]
        export_text = "\n".join(export_lines)

        st.download_button(
            label="📄 ดาวน์โหลดข้อเสนอตรวจสุขภาพ (Proposal TXT)",
            data=export_text.encode("utf-8"),
            file_name=f"health_package_proposal_{sel_pid}.txt",
            mime="text/plain",
            use_container_width=True
        )

        if st.button("📲 ส่งข้อเสนอให้ผู้รับบริการผ่าน LINE OA (Mockup)", use_container_width=True):
            st.success(f"✅ ส่งข้อเสนอ '{chosen_sub_name}' เข้า LINE OA สำเร็จ!")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Tabbed Folder System (st.radio styled as tabs)
# ============================================================
st.markdown('''
<style>
div.stRadio > div[role='radiogroup'] {
    flex-direction: row;
    gap: 2px;
    border-bottom: 2px solid #0E5C56;
    margin-bottom: 20px;
}
div.stRadio > div[role='radiogroup'] > label {
    background-color: #f1f5f9;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    margin-bottom: 0px;
    cursor: pointer;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    transition: all 0.3s ease;
}
div.stRadio > div[role='radiogroup'] > label[data-checked="true"] {
    background-color: #0E5C56 !important;
    color: white !important;
    border-color: #0E5C56;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
}
div.stRadio > div[role='radiogroup'] > label:hover {
    background-color: #e2e8f0;
}
</style>
''', unsafe_allow_html=True)

tab_options = [f"{v['icon']} {k}" for k, v in DISEASE_CONFIG.items()]
selected_tab_str = st.radio(" ", tab_options, horizontal=True, label_visibility="collapsed")
selected_tab_key = selected_tab_str.split(" ", 1)[1]
active_config = DISEASE_CONFIG[selected_tab_key]


# ============================================================
# Sidebar filters (Contextual)
# ============================================================
with st.sidebar:
    st.markdown(f"### 🎛️ Filter Scope: {selected_tab_key}")

    if active_config.get("is_general"):
        st.info("💡 เลือกแฟ้มกลุ่มโรคด้านบน เพื่อเปิดโหมดจัดการเฉพาะทาง (Command Center)")
        search_term = ""
    elif active_config.get("is_other_packages"):
        st.info("💡 แดชบอร์ดวิเคราะห์กลุ่มอื่น ๆ แนะนำแพ็คเกจ 4 ระดับ (AI Health Data Architect)")
        search_term = st.text_input("🔍 ค้นหาผู้รับบริการ (ID/ชื่อ)", key="ctx_search_other")
    else:
        search_term = st.text_input(f"🔍 ค้นหาคนไข้ (ID)", key="ctx_search")

    all_diseases = sorted(df["disease_group"].unique())
    disease_sel  = st.multiselect("กลุ่มโรค",  all_diseases,  default=all_diseases)

    all_genders  = sorted(df["gender"].unique())
    gender_sel   = st.multiselect("เพศ",        all_genders,   default=all_genders)

    all_months  = sorted(df["year_month"].dropna().unique())
    month_sel   = st.multiselect("เดือน",       all_months,    default=all_months)

    all_clinics = sorted(df["clinic_name"].dropna().unique())
    clinic_sel  = st.multiselect("คลินิก",      all_clinics,   default=all_clinics)

    if has_date:
        valid_dates = df["visit_date"].dropna()
        if not valid_dates.empty:
            import datetime
            d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
            cal_min = min(d_min, datetime.date(2026, 5, 1))
            cal_max = max(d_max, datetime.date(2026, 6, 30))
            date_filter = st.date_input("ช่วงวันที่", (d_min, d_max), min_value=cal_min, max_value=cal_max)
        else:
            date_filter = None
    else:
        date_filter = None

# Apply filters
mask = (
    df["disease_group"].isin(disease_sel  or all_diseases) &
    df["gender"].isin(gender_sel          or all_genders) &
    df["year_month"].isin(month_sel       or all_months) &
    df["clinic_name"].isin(clinic_sel     or all_clinics)
)
if date_filter and isinstance(date_filter, (list,tuple)) and len(date_filter) == 2:
    mask &= df["visit_date"].between(pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])) | df["visit_date"].isna()

dv = df[mask].copy()
if dv.empty:
    st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก")
    st.stop()

# Period-over-Period
delta_v = delta_p = delta_r = compare_label = None
if has_date and date_filter and isinstance(date_filter,(list,tuple)) and len(date_filter)==2:
    base = df["disease_group"].isin(disease_sel or all_diseases) & df["gender"].isin(gender_sel or all_genders)
    cur_s, cur_e = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
    span = cur_e - cur_s
    prev_e, prev_s = cur_s - pd.Timedelta(days=1), cur_s - span - pd.Timedelta(days=1)
    cur_d  = df[base & df["visit_date"].between(cur_s, cur_e)]
    prev_d = df[base & df["visit_date"].between(prev_s, prev_e)]
    compare_label = f"เทียบ {span.days+1} วันก่อนหน้า"
    if len(prev_d) >= 1:
        delta_v = (len(cur_d) - len(prev_d)) / len(prev_d) * 100
        if "patient_id" in df.columns:
            pp, cp = prev_d["patient_id"].nunique(), cur_d["patient_id"].nunique()
            if pp >= 1:
                delta_p = (cp - pp) / pp * 100
                pr = prev_d[prev_d["critical_risk"]==1]["patient_id"].nunique()
                cr = cur_d[cur_d["critical_risk"]==1]["patient_id"].nunique()
                delta_r = (cr/cp*100 if cp else 0) - (pr/pp*100 if pp else 0)


# ============================================================
# Dynamic Main Area
# ============================================================
if active_config.get("is_general"):
    as_of     = df["visit_date"].max()
    as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"

    st.markdown(f"""
    <div class="header-bar">
      <div>
        <div class="header-title">🏥 Clinical Command Center</div>
        <div class="header-sub">ภาพรวมตัวชี้วัดสุขภาพ · การจัดการกลุ่มเสี่ยง · โอกาสขยายผลแพ็กเกจตรวจสุขภาพเชิงป้องกัน</div>
      </div>
      <span class="asof-chip">ข้อมูลล่าสุด {as_of_str} · {df['clinic_name'].nunique()} คลินิก</span>
    </div>
    """, unsafe_allow_html=True)


    # Section 1 — Pulse + KPIs
    total_v   = len(dv)
    uniq_pts  = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v
    risk_pts  = dv[dv["critical_risk"]==1]["patient_id"].nunique() if "patient_id" in dv.columns else 0
    risk_pct  = risk_pts / uniq_pts * 100 if uniq_pts else 0
    avg_bp    = dv["systolic"].mean()
    bp_ok_pct = dv["systolic"].notna().mean() * 100
    miss_bp   = dv["systolic"].isna().mean() * 100
    uncoded   = (dv["disease_group"]=="อื่น ๆ").mean() * 100 if "อื่น ๆ" in dv["disease_group"].unique() else 0

    health_score = round((max(0, 100 - risk_pct*2) + bp_ok_pct) / 2)
    if health_score >= 80:   health_status, health_color = "ปกติดี",       SAGE
    elif health_score >= 60: health_status, health_color = "เฝ้าระวัง",    AMBER
    else:                    health_status, health_color = "ต้องดำเนินการ", RED

    pulse_col, kpi_col = st.columns([1, 3])
    with pulse_col:
        st.markdown(f"""
        <div class="pulse-card">
          <div class="pulse-dot" style="background:{health_color};">{health_score}</div>
          <div>
            <div class="pulse-label">Health Score</div>
            <div class="pulse-status" style="color:{health_color};">{health_status}</div>
            <div style="font-size:0.68rem;color:{MUTED};margin-top:3px;">คุมความเสี่ยง + ข้อมูลครบถ้วน</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col:
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("จำนวนเคส",        f"{total_v:,}",
                  delta=f"{delta_v:+.1f}% {compare_label}" if delta_v is not None else None)
        k2.metric("ผู้รับบริการ",     f"{uniq_pts:,}",
                  delta=f"{delta_p:+.1f}% {compare_label}" if delta_p is not None else None)
        k3.metric("Systolic เฉลี่ย", f"{avg_bp:.1f} mmHg" if pd.notna(avg_bp) else "N/A",
                  help=f"คำนวณจากเคสที่มีค่าจริง ({bp_ok_pct:.0f}%)")
        k4.metric("กลุ่มเสี่ยงวิกฤต", f"{risk_pct:.1f}%",
                  delta=f"{delta_r:+.1f} pp {compare_label}" if delta_r is not None else f"{risk_pts:,} คน",
                  delta_color="inverse")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


    # Section 2 — Data Quality Indicator
    def quality_meter(label, pct, note):
        c = SAGE if pct >= 80 else (AMBER if pct >= 50 else RED)
        st.markdown(f"""
        <div class="meter-wrap">
          <div class="meter-top">
            <span>{label}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-weight:600;color:{c};">{pct:.0f}%</span>
          </div>
          <div class="meter-track"><div class="meter-fill" style="width:{max(pct,2):.0f}%;background:{c};"></div></div>
          <div style="font-size:0.68rem;color:{MUTED};margin-top:2px;">{note}</div>
        </div>""", unsafe_allow_html=True)

    with st.container(key="card_dq"):
        st.markdown('<div class="panel-title">📊 Data Quality Indicator</div>', unsafe_allow_html=True)
        q1,q2,q3,q4 = st.columns(4)
        with q1: quality_meter("ข้อมูลความดันครบถ้วน",    100-miss_bp,  f"{miss_bp:.0f}% ว่าง — fallback จาก bp_raw")
        with q2: quality_meter("ครอบคลุมกลุ่มผู้ใหญ่",  100-(~dv["is_adult"]).mean()*100, f"{(~dv['is_adult']).mean()*100:.1f}% เป็นเด็ก")
        with q3: quality_meter("จัดหมวดโรคสำเร็จ",        100-uncoded,  f"{uncoded:.0f}% ยังอยู่ใน 'อื่น ๆ'")
        with q4: quality_meter("มีข้อมูล BMI จริง",       100-dv["bmi_imputed"].mean()*100, "ที่เหลือ imputed ด้วยค่ากลาง")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


    # Section 3 — Sunburst + Command Action Panel
    def build_sunburst(data, top_n=SUNBURST_TOP_N):
        rows, top_map = [], {}
        for grp, sub in data.groupby("disease_group"):
            counts = sub["diagnosis_clean"].value_counts()
            top = counts.head(top_n)
            rest = counts.iloc[top_n:].sum()
            top_map[grp] = list(top.index)
            for lb,cnt in top.items():
                rows.append({"disease_group":grp,"diagnosis":lb,"count":int(cnt)})
            if rest > 0:
                rows.append({"disease_group":grp,"diagnosis":"อื่นๆ ในกลุ่มนี้","count":int(rest)})
        return pd.DataFrame(rows), top_map

    hero_l, hero_r = st.columns([1.7,1])

    with hero_l:
        with st.container(key="card_sunburst"):
            st.markdown('<div class="panel-title">🔬 โครงสร้างการวินิจฉัย — คลิกเพื่อเจาะลึก</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-sub">{uncoded:.0f}% ของเคสอยู่ใน "อื่น ๆ" — คลิกวงในดูกลุ่มโรค วงนอกดูรหัสวินิจฉัย</div>', unsafe_allow_html=True)

            sun_df, top_map = build_sunburst(dv)
            if not sun_df.empty:
                fig_sun = px.sunburst(sun_df, path=["disease_group","diagnosis"], values="count",
                                       color="disease_group", color_discrete_map=DISEASE_COLORS)
                fig_sun.update_layout(margin=dict(l=0,r=0,t=6,b=6), height=420, paper_bgcolor="rgba(0,0,0,0)")
                fig_sun.update_traces(textfont_size=11, insidetextorientation="radial")

                click = st.plotly_chart(fig_sun, use_container_width=True,
                                         on_select="rerun", selection_mode="points", key="sun_click")
                sel_label = sel_group = None
                if click and click.selection and click.selection.get("points"):
                    pt = click.selection["points"][0]
                    sel_label = pt.get("label")
                    sel_group = pt.get("parent") or sel_label

                if sel_label:
                    if sel_label in DISEASE_COLORS:
                        drill = dv[dv["disease_group"]==sel_label]
                        n = drill["patient_id"].nunique() if "patient_id" in drill.columns else len(drill)
                        st.markdown(f'<div class="drill-banner">🔍 กลุ่ม: {sel_label} · {n} ราย</div>', unsafe_allow_html=True)
                    elif sel_label == "อื่นๆ ในกลุ่มนี้":
                        rest_lb = top_map.get(sel_group,[])
                        drill = dv[(dv["disease_group"]==sel_group) & (~dv["diagnosis_clean"].isin(rest_lb))]
                        st.markdown(f'<div class="drill-banner">🔍 อื่นๆ ในกลุ่ม {sel_group} · {len(drill)} เคส</div>', unsafe_allow_html=True)
                    else:
                        drill = dv[dv["diagnosis_clean"]==sel_label]
                        st.markdown(f'<div class="drill-banner">🔍 {sel_label} · {len(drill)} เคส</div>', unsafe_allow_html=True)

                    show_c = [c for c in ["patient_id","clinic_name","age_at_visit","gender","bmi","systolic"] if c in drill.columns]
                    st.dataframe(drill[show_c].head(10), use_container_width=True, hide_index=True, height=180)
                else:
                    st.caption("👆 คลิกส่วนใดของผังเพื่อดูรายละเอียด")

    with hero_r:
        with st.container(key="card_bp_donut"):
            st.markdown('<div class="panel-title">🩺 ระดับความดันโลหิต</div>', unsafe_allow_html=True)
            bp_pie = dv["bp_level"].value_counts().reset_index()
            bp_pie.columns = ["Level","Count"]
            fig_pie = px.pie(bp_pie, names="Level", values="Count", hole=0.55,
                              color="Level", color_discrete_map=BP_COLORS)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230,
                                   margin=dict(l=0,r=0,t=10,b=0),
                                   legend=dict(orientation="h",yanchor="bottom",y=-0.3,font=dict(size=9)))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        with st.container(key="card_action"):
            st.markdown('<div class="panel-title">⚡ Command Action Panel</div>', unsafe_allow_html=True)
            high_risk = render_action_panel(dv)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


    # Section 4 — Risk Table
    with st.container(key="card_risktable"):
        st.markdown('<div class="panel-title">🚨 รายชื่อผู้ป่วยกลุ่มเสี่ยงสูง</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">ไล่สีตามความรุนแรงของ Systolic BP</div>', unsafe_allow_html=True)
        if not high_risk.empty:
            disp = high_risk.copy()
            disp.columns = [c[:4]+"…" if len(c)>6 else c for c in disp.columns]
            disp.columns = ["ID","กลุ่มโรค","BMI","Sys","Dia"]

            if AGGRID_AVAILABLE and USE_AGGRID:
                row_js = JsCode("""function(p){
                    if(p.data.Sys>=180) return{'backgroundColor':'#F3A9A5','color':'#5A0E0E'};
                    if(p.data.Sys>=160) return{'backgroundColor':'#F8C6C0'};
                    if(p.data.Sys>=140) return{'backgroundColor':'#FBE1DE'};
                }""")
                gb = GridOptionsBuilder.from_dataframe(disp)
                gb.configure_pagination(paginationPageSize=8)
                gb.configure_grid_options(getRowStyle=row_js)
                AgGrid(disp, gridOptions=gb.build(), allow_unsafe_jscode=True,
                       fit_columns_on_grid_load=True, height=260, theme="alpine")
            else:
                def hl(row):
                    if row["Sys"] >= 180: return ["background-color:#F3A9A5"]*5
                    if row["Sys"] >= 160: return ["background-color:#F8C6C0"]*5
                    if row["Sys"] >= 140: return ["background-color:#FBE1DE"]*5
                    return [""]*5
                st.dataframe(disp.style.apply(hl,axis=1), use_container_width=True, hide_index=True, height=260)

            st.download_button(f"📥 Export ({len(high_risk)} ราย)",
                                high_risk.to_csv(index=False).encode("utf-8-sig"),
                                "critical_risk.csv","text/csv",use_container_width=True)
        else:
            st.success("✅ ไม่พบคนไข้ในเกณฑ์ความเสี่ยงวิกฤต")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


    # Section 5 — Population Pyramid + Clinic + BMI-BP Scatter
    p2,p3 = st.columns([1,1])

    with p2:
        with st.container(key="card_clinic"):
            st.markdown('<div class="panel-title">🏢 Top 10 คลินิก</div>', unsafe_allow_html=True)
            cl = dv["clinic_name"].value_counts().head(10).reset_index()
            cl.columns = ["คลินิก","n"]
            cl["lbl"] = cl["คลินิก"].apply(lambda x: x[:24]+"…" if len(x)>26 else x)
            fig_cl = px.bar(cl.sort_values("n"), x="n", y="lbl", orientation="h",
                             color="n", color_continuous_scale=[[0,"#CFE3DF"],[1,TEAL]])
            fig_cl.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                  height=310,margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False)
            fig_cl.update_yaxes(title=None)
            fig_cl.update_xaxes(title="จำนวนเคส")
            st.plotly_chart(fig_cl, use_container_width=True)

    with p3:
        with st.container(key="card_scatter"):
            st.markdown('<div class="panel-title">⚖️ BMI vs Systolic BP</div>', unsafe_allow_html=True)
            sc = dv[dv["is_adult"] & dv["systolic"].notna() & ~dv["bmi_imputed"]].copy()
            if "patient_id" in sc.columns:
                sc["freq"] = sc.groupby("patient_id")["patient_id"].transform("count")
            else:
                sc["freq"] = 1
            if not sc.empty:
                fig_sc = px.scatter(sc, x="bmi", y="systolic", color="disease_group",
                                     size="freq", color_discrete_map=DISEASE_COLORS, opacity=0.7)
                fig_sc.add_hline(y=140, line_dash="dot", line_color=RED, opacity=0.5)
                fig_sc.add_vline(x=25,  line_dash="dot", line_color=RED, opacity=0.5)
                fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                      height=310,margin=dict(l=0,r=0,t=10,b=0),
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=8)))
                fig_sc.update_xaxes(title="BMI",   gridcolor="#F1F5F9")
                fig_sc.update_yaxes(title="Systolic", gridcolor="#F1F5F9")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.caption("ไม่มีข้อมูลเพียงพอ")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


    # Section 6 — Clinical Health Trends + Patient Profile fragment
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown('<div class="panel-title">📈 Clinical Health Trend by Age</div>', unsafe_allow_html=True)
        c_tab1, c_tab2, c_tab3 = st.tabs(["📊 Age-Risk Stacked Bar", "🎯 Conversion Donut", "👥 Population Pyramid"])

        with c_tab1:
            risk_age = dv.groupby(["age_group", "critical_risk"]).size().reset_index(name="n")
            risk_age["Risk Level"] = risk_age["critical_risk"].map({0: "ปกติ/เฝ้าระวัง", 1: "High Risk"})
            if not risk_age.empty:
                fig_bar = px.bar(risk_age, x="age_group", y="n", color="Risk Level", barmode="stack",
                                 color_discrete_map={"ปกติ/เฝ้าระวัง": SAGE, "High Risk": RED},
                                 labels={"age_group": "ช่วงอายุ", "n": "จำนวนคนไข้"})
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      height=260, margin=dict(l=0, r=0, t=10, b=0),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
                fig_bar.update_xaxes(categoryorder="array", categoryarray=["<30 ปี","30-40 ปี","40-50 ปี","50-60 ปี",">60 ปี"], gridcolor="#F1F5F9")
                fig_bar.update_yaxes(gridcolor="#F1F5F9")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("ไม่มีข้อมูล Risk แยกตามอายุ")

        with c_tab2:
            if summary_pts is not None and "age_at_visit" in summary_pts.columns:
                def est_pkg(age):
                    if age >= 50: return "Longevity (>50)"
                    elif age >= 30: return "Advanced (30-50)"
                    return "Essential (<30)"
                summary_pts["pkg_type"] = summary_pts["age_at_visit"].apply(est_pkg)
                pkg_counts = summary_pts["pkg_type"].value_counts().reset_index()
                pkg_counts.columns = ["Package", "Count"]

                fig_don = px.pie(pkg_counts, names="Package", values="Count", hole=0.55,
                                 color="Package", color_discrete_map={
                                     "Longevity (>50)": AMBER,
                                     "Advanced (30-50)": TEAL,
                                     "Essential (<30)": "#2F6FB5"
                                 })
                fig_don.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                                      margin=dict(l=0, r=0, t=10, b=0),
                                      legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.0))
                st.plotly_chart(fig_don, use_container_width=True)

                st.markdown(f"""
                <div style="text-align:center; font-size:0.8rem; color:{MUTED}; margin-top:-10px;">
                    มูลค่าคาดการณ์ (Base): <span style="color:{TEAL}; font-weight:700;">฿ {(pkg_counts[pkg_counts["Package"]=="Longevity (>50)"]["Count"].sum() * 8000 + pkg_counts[pkg_counts["Package"]=="Advanced (30-50)"]["Count"].sum() * 5500 + pkg_counts[pkg_counts["Package"]=="Essential (<30)"]["Count"].sum() * 3000):,.0f}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ไม่มีข้อมูลผู้ป่วยที่สรุปได้")

        with c_tab3:
            age_order = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
            pyr = dv.groupby(["pyramid_group","gender"]).size().reset_index(name="n")
            m_s = pyr[pyr["gender"]=="ช"].set_index("pyramid_group")["n"]
            f_s = pyr[pyr["gender"]=="ญ"].set_index("pyramid_group")["n"]
            mv = [int(m_s.get(l,0)) for l in age_order]
            fv = [int(f_s.get(l,0)) for l in age_order]
            maxv = max(mv+fv) or 1
            fig_pyr = go.Figure()
            fig_pyr.add_trace(go.Bar(y=age_order,x=[-v for v in mv],name="ชาย",orientation="h",
                                      marker_color=TEAL,customdata=mv,
                                      hovertemplate="ชาย %{y}: %{customdata}<extra></extra>"))
            fig_pyr.add_trace(go.Bar(y=age_order,x=fv,name="หญิง",orientation="h",
                                      marker_color="#D97AA0",customdata=fv,
                                      hovertemplate="หญิง %{y}: %{customdata}<extra></extra>"))
            fig_pyr.update_layout(barmode="overlay",paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",height=260,
                                   margin=dict(l=0,r=0,t=10,b=0),
                                   legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=9)),
                                   xaxis=dict(tickvals=[-maxv,-maxv//2,0,maxv//2,maxv],
                                              ticktext=[str(maxv),str(maxv//2),"0",str(maxv//2),str(maxv)],
                                              gridcolor="#F1F5F9"))
            st.plotly_chart(fig_pyr, use_container_width=True)

        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔍 เลือกผู้ป่วยเพื่อประเมิน Package (Search & Select)</div>', unsafe_allow_html=True)

        if summary_pts is not None and "patient_id" in dv.columns:
            avail_df = summary_pts[summary_pts.index.isin(dv["patient_id"].values)].reset_index()
            if not avail_df.empty:
                selection = st.dataframe(
                    avail_df[["patient_id", "age_at_visit", "visits", "bmi", "systolic"]].rename(
                        columns={"patient_id": "Patient ID", "age_at_visit": "Age", "visits": "Visits", "bmi": "BMI", "systolic": "Systolic"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                    on_select="rerun",
                    selection_mode="single-row"
                )
                sel_idx = selection.selection.rows
            else:
                sel_idx = []
        else:
            avail_df = None
            sel_idx = []

    with col_right:
        render_patient_profile(avail_df, summary_pts, dv, sel_idx)

    st.divider()

    # Section 7 — Raw Data
    with st.expander("📋 ดูข้อมูลดิบ (visits_cleaned)", expanded=False):
        show_c = [c for c in ["visit_date","visit_id","patient_id","gender","age_at_visit",
                                "clinic_name","diagnosis_clean","disease_group",
                                "systolic","diastolic","bmi"] if c in dv.columns]
        st.dataframe(dv[show_c].sort_values("visit_date",ascending=False) if "visit_date" in dv.columns else dv[show_c],
                      use_container_width=True, hide_index=True)
        st.caption(f"แสดง {len(dv):,} แถว")
elif active_config.get("is_other_packages"):
    render_other_packages_dashboard(dv, df, search_term)
else:
    render_disease_center(dv, df, selected_tab_key, active_config, search_term)
