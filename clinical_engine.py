"""
Clinical Risk Scoring, Multi-tier Risk Assessment, and Health Package Recommendations
"""

import pandas as pd
from modules.config import HEALTH_PACKAGES, SPECIAL_SCREENINGS


def recommend_package(row):
    """
    Evaluates age and gender to recommend base health package and special screenings.
    Returns: (pkg_name, base_price, screenings, add_on_price)
    """
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

    # Female (gender_code > 0.5) age >= 40 -> Mammogram
    if gender_code > 0.5 and age >= 40:
        screenings.append("Mammogram")
        add_on_price += SPECIAL_SCREENINGS.get("Mammogram", {}).get("price", 2000)

    # Male (gender_code < 0.5) age >= 50 -> PSA
    if gender_code < 0.5 and age >= 50:
        screenings.append("PSA (มะเร็งต่อมลูกหมาก)")
        add_on_price += SPECIAL_SCREENINGS.get("PSA (มะเร็งต่อมลูกหมาก)", {}).get("price", 2000)

    return pkg_name, base_price, screenings, add_on_price


def analyze_patient_risk(row):
    """
    Calculates overall Health Score (0-100%), extracts risk rationale badges,
    and pairs with recommended base package and personal add-ons.
    """
    score = 100
    reasons = []

    sys_val = row.get("systolic", 0)
    bmi_val = row.get("bmi", 22)
    visits_val = row.get("visits", 1)

    # Systolic blood pressure deductions
    if sys_val >= 160:
        score -= 40
        reasons.append(
            f"<span style='background:#FEE2E2; color:#B91C1C; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันวิกฤต ({sys_val:.0f})</span>"
        )
    elif sys_val >= 140:
        score -= 25
        reasons.append(
            f"<span style='background:#FEE2E2; color:#B91C1C; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 ความดันสูง ({sys_val:.0f})</span>"
        )
    elif sys_val >= 130:
        score -= 10
        reasons.append(
            f"<span style='background:#FEF3C7; color:#B45309; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🫀 เฝ้าระวังความดัน ({sys_val:.0f})</span>"
        )

    # BMI deductions
    if bmi_val >= 30:
        score -= 20
        reasons.append(
            f"<span style='background:#FEE2E2; color:#B91C1C; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 โรคอ้วน ({bmi_val:.1f})</span>"
        )
    elif bmi_val >= 25:
        score -= 10
        reasons.append(
            f"<span style='background:#FEF3C7; color:#B45309; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>📈 น้ำหนักเกิน ({bmi_val:.1f})</span>"
        )

    # Visit frequency deductions
    if visits_val >= 5:
        score -= 15
        reasons.append(
            f"<span style='background:#FEE2E2; color:#B91C1C; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มารพ. บ่อยผิดปกติ ({visits_val:.0f} ครั้ง)</span>"
        )
    elif visits_val >= 3:
        score -= 5
        reasons.append(
            f"<span style='background:#FEF3C7; color:#B45309; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🏥 มีประวัติมาซ้ำ ({visits_val:.0f} ครั้ง)</span>"
        )

    pkg_name, base_price, screenings, add_on_price = recommend_package(row)
    if screenings:
        score -= 5
        for sc in screenings:
            reasons.append(
                f"<span style='background:#F3E8FF; color:#7E22CE; border:1px solid #D8B4FE; padding:4px 8px; border-radius:12px; font-size:0.7rem; margin-right:4px; display:inline-block; margin-bottom:4px;'>🎗️ แนะนำ {sc}</span>"
            )

    if not reasons:
        reasons.append(
            "<span style='background:#ECFDF5; color:#065F46; padding:4px 8px; border-radius:12px; font-size:0.7rem; display:inline-block; margin-bottom:4px;'>✅ สุขภาพอยู่ในเกณฑ์ปกติ</span>"
        )

    return max(0, score), "".join(reasons), pkg_name, base_price + add_on_price, screenings


def assess_patient_tier(pt_row):
    """
    Classifies a patient into clinical tiers 1-4 and matches the most optimal
    hospital sub-package based on Vichaivej Hospital clinical architecture standards.
    """
    age = float(pt_row.get("age_at_visit", pt_row.get("age", pt_row.get("patient_age", 35))))
    bmi = float(pt_row.get("bmi", 22.0))
    sys_bp = float(pt_row.get("systolic", pt_row.get("sbp", 120)) if pd.notna(pt_row.get("systolic", pt_row.get("sbp", 120))) else 120)
    dia_bp = float(pt_row.get("diastolic", pt_row.get("dbp", 80)) if pd.notna(pt_row.get("diastolic", pt_row.get("dbp", 80))) else 80)
    gender = str(pt_row.get("gender", pt_row.get("sex", "")))
    diag = str(pt_row.get("diagnosis_clean", pt_row.get("diagnosis_text", pt_row.get("diagnosis", "")))).lower()

    # Clinical indications from diagnosis text
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

    # Estimate waist circumference
    waist_cm = float(pt_row.get("waist", bmi * (3.65 if gender == "ช" else 3.35)))
    is_central_obesity = (gender == "ช" and waist_cm > 90) or (gender != "ช" and waist_cm > 80)

    reasons = []

    # Level 4: Complex chronic symptoms or high acuity
    if (sys_bp >= 160 or dia_bp >= 100) or (bmi >= 32.0) or (age >= 60) or has_chronic_symptoms or (pt_row.get("critical_risk") == 1):
        tier = 4
        if sys_bp >= 160 or dia_bp >= 100:
            reasons.append(f"ความดันโลหิตระดับวิกฤต (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg) เสี่ยงต่อระบบหัวใจและหลอดเลือด")
        if bmi >= 32.0:
            reasons.append(f"ภาวะโรคอ้วนระดับรุนแรงมาก (BMI {bmi:.1f} kg/m²)")
        if age >= 60:
            reasons.append(f"ผู้สูงอายุวัย {age:.0f} ปี มีความเสี่ยงต่อโรคเรื้อรังซับซ้อนและโรคมะเร็ง")
        if has_chronic_symptoms:
            reasons.append("พบอาการสงสัยหรือประวัติโรคเรื้อรังที่ควรได้รับการตรวจวินิจฉัยเชิงลึก")

        if sys_bp >= 160 or dia_bp >= 100 or has_heart_symptoms:
            best_sub_package = "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)"
        elif has_gi_symptoms:
            best_sub_package = "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)"
        else:
            best_sub_package = "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)"
        return tier, reasons, best_sub_package

    # Level 3: Early-stage disease or significant biomarker elevation
    if (140 <= sys_bp < 160 or 90 <= dia_bp < 100) or (has_ht or has_dm or has_lipid) or (27.5 <= bmi < 32.0) or (50 <= age < 60):
        tier = 3
        if has_dm:
            reasons.append("พบค่าบ่งชี้ระดับน้ำตาลในเลือดสูง / ความเสี่ยงเบาหวาน")
        if 140 <= sys_bp < 160 or 90 <= dia_bp < 100 or has_ht:
            reasons.append(f"ความดันโลหิตสูงระดับที่ 1 (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        if has_lipid:
            reasons.append("ระดับไขมันในเลือดสูงกว่าเกณฑ์มาตรฐาน")
        if 27.5 <= bmi < 32.0:
            reasons.append(f"ภาวะน้ำหนักเกินระดับอันตราย (BMI {bmi:.1f} kg/m²)")

        if has_dm:
            best_sub_package = "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)"
        elif has_ht or (140 <= sys_bp < 160):
            best_sub_package = "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)"
        else:
            best_sub_package = "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)"
        return tier, reasons, best_sub_package

    # Level 2: Risk factors / early metabolic warning
    if (120 <= sys_bp < 140 or 80 <= dia_bp < 90) or (bmi >= 23.0) or (35 <= age < 50) or is_central_obesity:
        tier = 2
        if 120 <= sys_bp < 140 or 80 <= dia_bp < 90:
            reasons.append(f"ความดันโลหิตระยะก่อนความดันสูง Pre-hypertension (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        if bmi >= 23.0:
            reasons.append(f"ดัชนีมวลกายเกินเกณฑ์มาตรฐานคนเอเชีย (BMI {bmi:.1f} kg/m²)")
        if is_central_obesity:
            reasons.append(f"ภาวะอ้วนลงพุง (รอบเอวประมาณการ {waist_cm:.1f} cm)")
        if 35 <= age < 50:
            reasons.append(f"อยู่ในช่วงวัยทำงานตอนกลาง ({age:.0f} ปี) มีความเครียดสะสมและพฤติกรรมเสี่ยง")

        if is_central_obesity or (120 <= sys_bp < 140 and bmi >= 23.0):
            best_sub_package = "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH"
        elif bmi >= 25.0:
            best_sub_package = "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)"
        else:
            best_sub_package = "แพคเกจตรวจสุขภาพ วัย 35+"
        return tier, reasons, best_sub_package

    # Level 1: General Healthy Checkup
    tier = 1
    reasons.append("สุขภาพโดยรวมแข็งแรงดี สัญญาณชีพและดัชนีมวลกายอยู่ในเกณฑ์มาตรฐาน")
    if age > 35:
        best_sub_package = "แพคเกจตรวจสุขภาพ PREMIUM"
    elif 30 <= age <= 35:
        best_sub_package = "แพคเกจตรวจสุขภาพ STANDARD"
    else:
        best_sub_package = "แพคเกจตรวจสุขภาพ BASIC"

    return tier, reasons, best_sub_package
