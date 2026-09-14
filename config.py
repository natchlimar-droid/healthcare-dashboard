"""
Application Configurations, Design Tokens, and Clinical Catalogs
"""

# ============================================================
# Design Tokens & Theme Colors
# ============================================================
INK     = "#0B1B2B"
BG      = "#F7F9F8"
SURFACE = "#FFFFFF"
TEAL    = "#0E5C56"
SAGE    = "#4F7C5B"
AMBER   = "#C87F0A"
RED     = "#B3261E"
MUTED   = "#5B6B6B"

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

# ============================================================
# Standard Health Packages (3 Packages based on Age)
# ============================================================
HEALTH_PACKAGES = {
    "Essential Package": {
        "price": 3000,
        "tests": [
            "CBC (ความสมบูรณ์ของเม็ดเลือด)",
            "FBS (น้ำตาลในเลือด)",
            "Lipid Profile (ไขมันในเลือด)",
            "Uric Acid (กรดยูริก)",
            "CXR (เอกซเรย์ปอด)",
            "EKG (คลื่นไฟฟ้าหัวใจ)"
        ],
        "desc": "เหมาะสำหรับวัยเริ่มต้นทำงานและผู้ที่ไม่มีความเสี่ยงหรือโรคประจำตัว (อายุ <30 ปี)"
    },
    "Advanced Package": {
        "price": 5500,
        "tests": [
            "Essential Tests +",
            "Liver Function (การทำงานของตับ)",
            "Kidney Function (การทำงานของไต)",
            "HbA1c (น้ำตาลสะสม)",
            "Urine Examination (ปัสสาวะ)",
            "Ultrasound Whole Abdomen (อัลตราซาวด์ช่องท้อง)"
        ],
        "desc": "เหมาะสำหรับวัยทำงานที่มีความเครียดสะสม พักผ่อนน้อย หรือเริ่มมีความเสี่ยง (อายุ 30-50 ปี)"
    },
    "Longevity Package": {
        "price": 8000,
        "tests": [
            "Advanced Tests +",
            "Thyroid Function (ไทรอยด์)",
            "Bone Densitometry (มวลกระดูก)",
            "Tumor Markers (สารบ่งชี้มะเร็งพื้นฐาน)",
            "ABI (การตีบตันของหลอดเลือด)"
        ],
        "desc": "เหมาะสำหรับผู้สูงอายุ หรือผู้ที่มีความเสี่ยงโรคเรื้อรัง ต้องการดูแลอย่างใกล้ชิด (อายุ >50 ปี)"
    }
}

SPECIAL_SCREENINGS = {
    "Mammogram": {"price": 2000, "desc": "คัดกรองมะเร็งเต้านม"},
    "PSA (มะเร็งต่อมลูกหมาก)": {"price": 2000, "desc": "คัดกรองมะเร็งต่อมลูกหมาก"}
}

# ============================================================
# AI Health Data Architect System Prompt
# ============================================================
SYSTEM_PROMPT_HEALTH_ARCHITECT = (
    "คุณคือ Senior Health Data Architect และผู้เชี่ยวชาญด้านเวชศาสตร์ป้องกันและระบบคัดกรองสุขภาพอัจฉริยะ "
    "ประจำโรงพยาบาลวิชัยเวช อินเตอร์เนชั่นแนล อ้อมน้อย (Vichaivej International Hospital Omnoi) "
    "หน้าที่ของคุณคือการวิเคราะห์ข้อมูลผู้รับบริการ (อายุ, เพศ, ดัชนีมวลกาย BMI, เส้นรอบเอว, ความดันโลหิต, ประวัติโรค, อาการ และผลแล็บ) "
    "เพื่อประเมินระดับความเสี่ยง 4 กลุ่ม (Level 1: ตรวจสุขภาพทั่วไป, Level 2: ยังไม่พบโรคแต่มีความเสี่ยง, "
    "Level 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น, Level 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม) "
    "และคัดเลือกแพ็กเกจย่อยที่เหมาะสมที่สุดเฉพาะบุคคล (Personalized Best Match Sub-package) "
    "ตามมาตรฐานแพ็กเกจตรวจสุขภาพจริงของโรงพยาบาลวิชัยเวชฯ อ้อมน้อย ได้อย่างแม่นยำ ปลอดภัย คุ้มค่า และเกิดประโยชน์สูงสุด"
)

# ============================================================
# 4-Level Clinical Health Packages (รพ.วิชัยเวช อินเตอร์เนชั่นแนล อ้อมน้อย)
# ============================================================
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

# ============================================================
# Navigation & Center Routing Configuration
# ============================================================
DISEASE_CONFIG = {
    "🏥 General Dashboard (หน้าแรก)": {
        "icon": "🏥",
        "is_general": True
    },
    "🔮 AI Forecast (พยากรณ์ 2568-2569)": {
        "icon": "🔮",
        "is_forecast": True,
        "target_desc": "พยากรณ์ปริมาณผู้รับบริการและแนวโน้มกลุ่มเสี่ยงวิกฤตล่วงหน้า 24 เดือน"
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
