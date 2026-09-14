import os
import random
import uuid
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Page Config (Must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Vichaivej Omnoi Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Optional Dependencies
# ============================================================
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    import shap
    XGB_SHAP_AVAILABLE = True
except ImportError:
    XGB_SHAP_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

USE_AGGRID = False

# Master switch: only used as a last-resort fallback inside load_clinical_df()
# if no real visits_cleaned.csv/.zip file can be found on disk.
USE_DEMO_DATA_FALLBACK = True

# ============================================================
# Design Tokens & Configurations
# ============================================================
# --- Clinical Command Center palette (from the analytics dashboard) ---
INK = "#0B1B2B"
BG = "#F7F9F8"
SURFACE = "#FFFFFF"
TEAL = "#0E5C56"
SAGE = "#4F7C5B"
AMBER = "#C87F0A"
RED = "#B3261E"
MUTED = "#5B6B6B"

DISEASE_COLORS = {
    "ความดันโลหิตสูง": TEAL,
    "เบาหวาน": "#2F6FB5",
    "ไขมันในเลือดสูง": AMBER,
    "อื่น ๆ": "#9AA6A0",
}
BP_COLORS = {
    "ปกติ (<120)": SAGE,
    "เฝ้าระวัง (120-139)": AMBER,
    "สูง (>=140)": RED,
    "ไม่มีข้อมูล": "#C7CFCC",
}

MIN_SAMPLE = 5
SUNBURST_TOP_N = 5

# --- Sales / CRM palette (from the demo Command Center) ---
COLORS = {
    "Navy": "#16325C",
    "Teal": "#009A9A",
    "Green": "#16A34A",
    "Amber": "#F59E0B",
    "Red": "#DC2626",
    "Purple": "#7C3AED",
    "Background": "#F8FAFC",
}

HEALTH_PACKAGES = {
    "Essential Package": {
        "price": 3000,
        "tests": ["CBC", "FBS", "Lipid Profile", "Uric Acid", "CXR", "EKG"],
        "desc": "เหมาะสำหรับวัยเริ่มต้นทำงานและผู้ที่ไม่มีความเสี่ยง (อายุ <30 ปี)"
    },
    "Advanced Package": {
        "price": 5500,
        "tests": ["Essential Tests +", "Liver Function", "Kidney Function", "HbA1c", "Urine Examination", "Ultrasound Whole Abdomen"],
        "desc": "เหมาะสำหรับวัยทำงานที่เริ่มมีความเสี่ยง (อายุ 30-50 ปี)"
    },
    "Longevity Package": {
        "price": 8000,
        "tests": ["Advanced Tests +", "Thyroid Function", "Bone Densitometry", "Tumor Markers", "ABI"],
        "desc": "เหมาะสำหรับผู้สูงอายุหรือมีความเสี่ยงโรคเรื้อรัง (อายุ >50 ปี)"
    }
}

HEALTH_PACKAGES_4LEVEL = {
    1: {
        "level": 1, "name": "กลุ่ม 1: ตรวจสุขภาพทั่วไป", "short_title": "ตรวจสุขภาพทั่วไป",
        "target_audience": "สำหรับคนที่ใส่ใจสุขภาพ อยากรู้พื้นฐานของตัวเอง",
        "badge_color": "#0E7055", "badge_bg": "#DCFCE7", "header_bg": "#0E7055",
        "price": 990,
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ BASIC", "short_name": "BASIC", "price": 990},
            {"name": "แพคเกจตรวจสุขภาพ STANDARD", "short_name": "STANDARD", "price": 2290},
            {"name": "แพคเกจตรวจสุขภาพ PREMIUM", "short_name": "PREMIUM", "price": 4990},
        ]
    },
    2: {
        "level": 2, "name": "กลุ่ม 2: ยังไม่พบโรคแต่มีความเสี่ยง", "short_title": "ยังไม่พบโรคแต่มีความเสี่ยง",
        "target_audience": "สำหรับผู้ที่มีปัจจัยเสี่ยงจากอายุ น้ำหนัก พฤติกรรม หรือประวัติครอบครัว",
        "badge_color": "#B45309", "badge_bg": "#FEF3C7", "header_bg": "#D97706",
        "price": 2990,
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ 9.9 METABOLIC HEALTH MONTH", "short_name": "9.9 METABOLIC", "price": 2990},
            {"name": "แพคเกจตรวจสุขภาพ คนอ้วน (Obesity Check)", "short_name": "คนอ้วน", "price": 2990},
            {"name": "แพคเกจตรวจสุขภาพ วัย 35+", "short_name": "วัย 35+", "price": 3990},
        ]
    },
    3: {
        "level": 3, "name": "กลุ่ม 3: เริ่มมีความผิดปกติ/โรคระยะเริ่มต้น", "short_title": "เริ่มมีความผิดปกติ",
        "target_audience": "สำหรับผู้ที่มีค่าผลตรวจเริ่มผิดปกติ หรือได้รับการวินิจฉัยโรคระยะเริ่มต้น",
        "badge_color": "#B91C1C", "badge_bg": "#FEE2E2", "header_bg": "#DC2626",
        "price": 2990,
        "sub_packages": [
            {"name": "แพคเกจตรวจสุขภาพ เบาหวาน (Diabetes Check)", "short_name": "เบาหวาน", "price": 3290},
            {"name": "แพคเกจตรวจสุขภาพ ความดันโลหิตสูง (Hypertension Check)", "short_name": "ความดันโลหิตสูง", "price": 2990},
            {"name": "แพคเกจตรวจสุขภาพ ไขมันในเลือดสูง (Lipid Check)", "short_name": "ไขมันในเลือดสูง", "price": 2990},
        ]
    },
    4: {
        "level": 4, "name": "กลุ่ม 4: มีอาการซ้ำ ๆ / ควรประเมินเพิ่มเติม", "short_title": "มีอาการซ้ำๆ",
        "target_audience": "สำหรับผู้ที่มีอาการผิดปกติซ้ำๆ หรือมีความกังวลและต้องการตรวจเชิงลึก",
        "badge_color": "#5B21B6", "badge_bg": "#F3E8FF", "header_bg": "#6941C6",
        "price": 4990,
        "sub_packages": [
            {"name": "แพคเกจตรวจคัดกรอง มะเร็ง (Cancer Screening)", "short_name": "คัดกรองมะเร็ง", "price": 4990},
            {"name": "แพคเกจตรวจสุขภาพ หัวใจ (Heart Check)", "short_name": "หัวใจ", "price": 4990},
            {"name": "แพคเกจตรวจคัดกรอง มะเร็งลำไส้ใหญ่ (Colonoscopy)", "short_name": "Colonoscopy", "price": 8900},
        ]
    }
}

DISEASE_CONFIG = {
    "🏥 General Dashboard (หน้าแรก)": {"icon": "🏥", "is_general": True},
    "🔮 AI Forecast (พยากรณ์ 2568-2569)": {"icon": "🔮", "is_forecast": True, "target_desc": "พยากรณ์ล่วงหน้า 24 เดือน"},
    "ล้างไต (Dialysis)": {"icon": "🩺", "filter_condition": lambda df: df["is_dialysis"] == True, "target_desc": "BP <130/80"},
    "เบาหวาน (Diabetes)": {"icon": "🩸", "filter_condition": lambda df: df["disease_group"] == "เบาหวาน", "target_desc": "HbA1c <7.0"},
    "ความดันโลหิตสูง (Hypertension)": {"icon": "🫀", "filter_condition": lambda df: df["disease_group"] == "ความดันโลหิตสูง", "target_desc": "BP <140/90"},
    "ไขมันในเลือดสูง (Dyslipidemia)": {"icon": "🧈", "filter_condition": lambda df: df["disease_group"] == "ไขมันในเลือดสูง", "target_desc": "LDL <100"},
    "กลุ่มอื่น ๆ (Health Packages)": {"icon": "🧬", "filter_condition": lambda df: (df["disease_group"] == "อื่น ๆ") | (df["disease_group"].isna()) | (df["disease_group"] == "ไม่ระบุ") | (df["disease_group"] == "ทั่วไป"), "target_desc": "คัดกรองสุขภาพเชิงรุก", "is_other_packages": True}
}

# ============================================================
# Global CSS & UI Helpers (from the CRM demo dashboard)
# ============================================================
def inject_css():
    st.markdown(f"""
    <style>
    .kpi-card {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid {COLORS['Teal']};
        margin-bottom: 15px;
    }}
    .kpi-title {{ color: #64748b; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }}
    .kpi-value {{ color: {COLORS['Navy']}; font-size: 1.8rem; font-weight: 700; }}
    .kpi-delta.positive {{ color: {COLORS['Green']}; font-size: 0.9rem; font-weight: 600; }}
    .kpi-delta.negative {{ color: {COLORS['Red']}; font-size: 0.9rem; font-weight: 600; }}

    .badge {{
        padding: 4px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; display: inline-block;
    }}
    .badge-success {{ background-color: #dcfce7; color: #166534; }}
    .badge-warning {{ background-color: #fef9c3; color: #854d0e; }}
    .badge-danger {{ background-color: #fee2e2; color: #991b1b; }}
    .badge-info {{ background-color: #e0f2fe; color: #075985; }}

    .recommendation-card {{
        background-color: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 10px;
    }}
    .recommendation-card.best-match {{
        border: 2px solid {COLORS['Purple']}; background-color: #f5f3ff;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_custom_html(html_str):
    clean_str = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_str, unsafe_allow_html=True)


def format_currency(val):
    return f"฿{val:,.0f}"


def format_percent(val):
    return f"{val:.1f}%"


def get_badge_html(text, status_type="info"):
    return f'<span class="badge badge-{status_type}">{text}</span>'


def kpi_card(title, value, delta=None, delta_desc="vs last month", invert_color=False):
    delta_html = ""
    if delta is not None:
        if delta > 0:
            c = "negative" if invert_color else "positive"
            delta_html = f'<div class="kpi-delta {c}">▲ {delta:.1f}% {delta_desc}</div>'
        elif delta < 0:
            c = "positive" if invert_color else "negative"
            delta_html = f'<div class="kpi-delta {c}">▼ {abs(delta):.1f}% {delta_desc}</div>'
        else:
            delta_html = f'<div class="kpi-delta" style="color:#64748b;">- 0.0% {delta_desc}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# Audit Log
if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = []


def log_audit(action, details, user_role):
    st.session_state["audit_log"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": user_role,
        "action": action,
        "details": details
    })


# ============================================================
# ML Caching Functions (Clinical Command Center)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_business_forecast_model(df_hist, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
    if not PROPHET_AVAILABLE or df_hist is None or df_hist.empty:
        return None, 0.0
    try:
        pdf = pd.DataFrame()
        if "ym" in df_hist.columns:
            pdf["ds"] = df_hist["ym"].dt.to_timestamp()
        else:
            pdf["ds"] = pd.to_datetime(df_hist.index)
        pdf["y"] = df_hist["total_cases"].values if "total_cases" in df_hist.columns else df_hist.iloc[:, 1].values

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        m.fit(pdf)

        forecast = m.predict(pdf)
        actual = pdf["y"].values
        predicted = forecast["yhat"].values
        non_zero = actual != 0
        if non_zero.any():
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0.0

        return m, mape
    except Exception as e:
        st.error(f"Prophet Error: {e}")
        return None, 0.0


@st.cache_resource(show_spinner=False)
def get_upsell_risk_models(df_train, max_depth=6, learning_rate=0.3, n_estimators=100):
    if not XGB_SHAP_AVAILABLE or not SKLEARN_AVAILABLE or df_train is None or df_train.empty:
        return None, None, {}
    try:
        features = ["age_at_visit", "bmi", "systolic", "gender_code", "visits"]
        available_features = [f for f in features if f in df_train.columns]
        if not available_features or "target" not in df_train.columns:
            return None, None, {}

        X = df_train[available_features].fillna(0)
        y = df_train["target"].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred) * 100,
            "roc_auc": roc_auc_score(y_test, y_prob) * 100 if len(np.unique(y_test)) > 1 else 0.0,
            "precision": precision_score(y_test, y_pred, zero_division=0) * 100,
            "recall": recall_score(y_test, y_pred, zero_division=0) * 100
        }

        explainer = shap.TreeExplainer(model)
        return model, explainer, metrics
    except Exception as e:
        st.error(f"XGBoost/SHAP Error: {e}")
        return None, None, {}


# ============================================================
# Clinical Data Loading & Preprocessing
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


def clean_and_enrich_clinical(df):
    """Shared cleaning/feature-engineering pipeline used by BOTH the real
    visits_cleaned.csv/.zip loader and the demo-data fallback, so that the
    Clinical Command Center works identically regardless of data source."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

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

    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    df["gender"] = df["gender"].fillna("ไม่ระบุ") if "gender" in df.columns else "ไม่ระบุ"
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)
    df["bmi_imputed"] = df["bmi"].isna() if "bmi" in df.columns else True
    df["bmi"] = df["bmi"].fillna(df["bmi"].median() if "bmi" in df.columns and pd.notna(df["bmi"].median()) else 22.0)

    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()
    if has_age:
        df["age_at_visit"] = df["age_at_visit"].fillna(df["age_at_visit"].median() if pd.notna(df["age_at_visit"].median()) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(df["age_at_visit"], bins=[0, 29, 39, 49, 59, 120], labels=["<30 ปี", "30-40 ปี", "40-50 ปี", "50-60 ปี", ">60 ปี"]).astype(str).replace("nan", "ไม่ระบุ")
        df["pyramid_group"] = pd.cut(df["age_at_visit"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False, labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]).astype(str)
    else:
        df["age_at_visit"], df["is_adult"], df["age_group"], df["pyramid_group"] = 35.0, True, "ไม่ระบุ", "ไม่ระบุ"

    bp_cat = pd.cut(df["systolic"], bins=[-1, 120, 139, 300], labels=["ปกติ (<120)", "เฝ้าระวัง (120-139)", "สูง (>=140)"]).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat
    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    df["diagnosis_clean"] = df["diagnosis_text"].astype(str).str.strip().replace({"": "ไม่ระบุ", "-": "ไม่ระบุ", ":": "ไม่ระบุ"}).where(df["diagnosis_text"].notna(), "ไม่ระบุ") if "diagnosis_text" in df.columns else "ไม่ระบุ"
    if "disease_group" not in df.columns: df["disease_group"] = "ทั่วไป"
    if "clinic_name" not in df.columns: df["clinic_name"] = "ไม่ระบุ"

    return df, has_date, has_age


def _map_diagnosis_to_disease_group(dx_text):
    dx = str(dx_text).lower()
    if "hypertension" in dx or "ความดัน" in dx:
        return "ความดันโลหิตสูง"
    if "dm" in dx or "diabetes" in dx or "เบาหวาน" in dx:
        return "เบาหวาน"
    if "dyslipidemia" in dx or "ไขมัน" in dx:
        return "ไขมันในเลือดสูง"
    if "ไต" in dx or "kidney" in dx or "dialysis" in dx or "ckd" in dx:
        return "ไต"
    return "อื่น ๆ"


def adapt_demo_to_clinical_schema(customers, clinical_visits):
    """Builds a visits_cleaned-style dataframe out of the CRM demo data so
    the full Clinical Command Center can still be demoed when no real
    visits_cleaned.csv/.zip file is present on disk."""
    if clinical_visits is None or clinical_visits.empty:
        return pd.DataFrame()

    cv = clinical_visits.merge(
        customers[["patient_id", "age", "gender", "preferred_branch"]],
        on="patient_id", how="left"
    )

    out = pd.DataFrame()
    out["patient_id"] = cv["patient_id"]
    out["visit_date"] = cv["visit_date"]
    out["age_at_visit"] = cv["age"]
    out["gender"] = cv["gender"].map({"Male": "ช", "Female": "ญ"}).fillna("ไม่ระบุ")
    out["height_cm"] = cv.get("height_cm", 165)
    out["weight_kg"] = cv.get("weight_kg", np.nan)
    out["bmi"] = cv["bmi"]
    out["systolic_bp"] = cv["systolic_bp"]
    out["diastolic_bp"] = cv["diastolic_bp"]
    out["clinic_name"] = cv["preferred_branch"]
    out["diagnosis_text"] = cv["diagnosis_text"]
    out["disease_group"] = cv["diagnosis_text"].apply(_map_diagnosis_to_disease_group)
    out.loc[out["disease_group"] == "ไต", "disease_group"] = "อื่น ๆ"  # dialysis flag handled separately
    return out


@st.cache_data(show_spinner="กำลังโหลดข้อมูลคลินิก...")
def load_clinical_df(_customers, _clinical_visits_demo):
    """Tries to load a real visits_cleaned.csv/.zip file first; falls back
    to the CRM demo dataset (adapted to the same schema) if none is found."""
    main_file = find_data_file([
        "visits_cleaned.csv", "visits_cleaned.zip",
        "visits_with_monthly_count.csv", "visits_with_monthly_count.zip"
    ])
    if main_file is not None:
        try:
            raw = read_csv_or_zip(main_file, preferred_csv_names=["visits_cleaned.csv", "visits_with_monthly_count.csv"])
            df, has_date, has_age = clean_and_enrich_clinical(raw)
            return df, "real"
        except Exception as e:
            st.warning(f"⚠️ อ่านไฟล์ข้อมูลหลักไม่สำเร็จ ({e}) — ใช้ข้อมูลจำลองแทน")

    if not USE_DEMO_DATA_FALLBACK:
        return None, "none"

    demo_raw = adapt_demo_to_clinical_schema(_customers, _clinical_visits_demo)
    if demo_raw.empty:
        return None, "none"
    df, has_date, has_age = clean_and_enrich_clinical(demo_raw)
    return df, "demo"


@st.cache_data(show_spinner=False)
def process_patient_data(dataframe):
    df_proc = dataframe.copy()
    today = df_proc["visit_date"].max() if pd.notna(df_proc["visit_date"].max()) else pd.Timestamp.today().normalize()

    if "diagnosis_text" in df_proc.columns:
        df_proc["is_dialysis"] = df_proc["diagnosis_text"].fillna("").astype(str).str.lower().str.contains("ไต|kidney|dialysis|ckd", regex=True)
    else:
        df_proc["is_dialysis"] = False

    days = (today - df_proc["visit_date"]).dt.days.fillna(0).clip(lower=0) if "visit_date" in df_proc.columns else pd.Series(0, index=df_proc.index)
    df_proc["days_since_last_visit"] = days
    is_dia = df_proc["is_dialysis"]
    is_diabetes = df_proc["disease_group"].astype(str).str.contains("เบาหวาน")

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
    agg_kwargs = {
        "bmi": ("bmi", "mean"),
        "systolic": ("systolic", "mean"),
        "gender_code": ("gender_code", "first"),
        "age_at_visit": ("age_at_visit", "max"),
    }
    agg_kwargs["visits"] = ("visit_id", "count") if "visit_id" in dataframe.columns else ("patient_id", "count")
    sp = dataframe.groupby("patient_id").agg(**agg_kwargs).round(1)
    sp["systolic"] = sp["systolic"].fillna(dataframe["systolic"].median())
    sp["target"] = ((sp["visits"] >= 3) | (sp["systolic"] >= 135)).astype(int)
    return sp


# ============================================================
# Clinical Business Logic Helpers
# ============================================================
def _recommend_package(row):
    age = row.get("age_at_visit", 35)
    gender_code = row.get("gender_code", 0.5)
    if age >= 50:
        pkg_name, base_price = "Longevity Package", 8000
    elif age >= 30:
        pkg_name, base_price = "Advanced Package", 5500
    else:
        pkg_name, base_price = "Essential Package", 3000
    screenings, add_on_price = [], 0
    if gender_code > 0.5 and age >= 40:
        screenings.append("Mammogram")
        add_on_price += 2000
    if gender_code < 0.5 and age >= 50:
        screenings.append("PSA (มะเร็งต่อมลูกหมาก)")
        add_on_price += 2000
    return pkg_name, base_price, screenings, add_on_price


def _analyze_patient_risk(row):
    score, reasons = 100, []
    sys_val = row.get("systolic", 0)
    bmi_val = row.get("bmi", 22)
    visits_val = row.get("visits", 1)

    if sys_val >= 160:
        score -= 40; reasons.append(f"ความดันวิกฤต ({sys_val:.0f})")
    elif sys_val >= 140:
        score -= 25; reasons.append(f"ความดันสูง ({sys_val:.0f})")
    elif sys_val >= 130:
        score -= 10; reasons.append(f"เฝ้าระวังความดัน ({sys_val:.0f})")
    if bmi_val >= 30:
        score -= 20; reasons.append(f"โรคอ้วน ({bmi_val:.1f})")
    elif bmi_val >= 25:
        score -= 10; reasons.append(f"น้ำหนักเกิน ({bmi_val:.1f})")
    if visits_val >= 5:
        score -= 15; reasons.append(f"มารพ.บ่อยผิดปกติ ({visits_val:.0f} ครั้ง)")
    elif visits_val >= 3:
        score -= 5; reasons.append(f"มีประวัติมาซ้ำ ({visits_val:.0f} ครั้ง)")

    pkg_name, base_price, screenings, add_on_price = _recommend_package(row)
    if screenings:
        score -= 5
    for sc in screenings:
        reasons.append(f"แนะนำ {sc}")
    if not reasons:
        reasons.append("สุขภาพอยู่ในเกณฑ์ปกติ")

    return max(0, score), " | ".join(reasons), pkg_name, base_price + add_on_price, screenings


def _assess_patient_tier(pt_row):
    age = float(pt_row.get("age_at_visit", pt_row.get("age", 35)))
    bmi = float(pt_row.get("bmi", 22.0))
    sys_bp_raw = pt_row.get("systolic", 120)
    sys_bp = float(sys_bp_raw if pd.notna(sys_bp_raw) else 120)
    dia_bp_raw = pt_row.get("diastolic", 80)
    dia_bp = float(dia_bp_raw if pd.notna(dia_bp_raw) else 80)
    gender = str(pt_row.get("gender", ""))
    diag = str(pt_row.get("diagnosis_clean", pt_row.get("diagnosis_text", ""))).lower()

    has_chronic = any(k in diag for k in ["แน่นหน้าอก", "เหนื่อย", "ใจสั่น", "หัวใจ", "chest pain", "มะเร็ง", "ก้อน", "tumor"])
    has_dm = any(k in diag for k in ["เบาหวาน", "diabetes", "dm"]) or float(pt_row.get("fbs", 0)) >= 100 or float(pt_row.get("hba1c", 0)) >= 5.7
    has_ht = any(k in diag for k in ["ความดัน", "hypertension", "ht"]) or sys_bp >= 140 or dia_bp >= 90
    has_lipid = any(k in diag for k in ["ไขมัน", "lipid", "cholesterol"]) or float(pt_row.get("cholesterol", 0)) >= 200
    waist_cm = float(pt_row.get("waist", bmi * (3.65 if gender == "ช" else 3.35)))
    is_central_obesity = (gender == "ช" and waist_cm > 90) or (gender != "ช" and waist_cm > 80)

    reasons = []
    if sys_bp >= 160 or dia_bp >= 100 or bmi >= 32.0 or age >= 60 or has_chronic or pt_row.get("critical_risk") == 1:
        if sys_bp >= 160 or dia_bp >= 100:
            reasons.append(f"ความดันโลหิตระดับวิกฤต (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        if bmi >= 32.0:
            reasons.append(f"โรคอ้วนรุนแรง (BMI {bmi:.1f})")
        best_sub = "หัวใจ (Heart Check)" if (sys_bp >= 160 or dia_bp >= 100) else "คัดกรองมะเร็ง"
        return 4, reasons, best_sub

    if (140 <= sys_bp < 160) or has_ht or has_dm or has_lipid or (27.5 <= bmi < 32.0) or (50 <= age < 60):
        if has_dm: reasons.append("พบค่าน้ำตาลในเลือดสูง")
        if 140 <= sys_bp < 160: reasons.append(f"ความดันโลหิตสูงระดับ 1 (BP {sys_bp:.0f}/{dia_bp:.0f})")
        best_sub = "เบาหวาน (Diabetes Check)" if has_dm else "ความดันโลหิตสูง (Hypertension Check)"
        return 3, reasons, best_sub

    if (120 <= sys_bp < 140) or bmi >= 23.0 or (35 <= age < 50) or is_central_obesity:
        if bmi >= 23.0: reasons.append(f"น้ำหนักเกินเกณฑ์ (BMI {bmi:.1f})")
        best_sub = "คนอ้วน (Obesity Check)" if bmi > 25.0 else "วัย 35+"
        return 2, reasons, best_sub

    reasons.append("สุขภาพโดยรวมแข็งแรงดี")
    best_sub = "PREMIUM" if age > 35 else ("STANDARD" if age >= 30 else "BASIC")
    return 1, reasons, best_sub


# ============================================================
# Clinical UI Fragments
# ============================================================
@st.fragment
def render_action_panel(dv):
    high_risk = dv[dv["critical_risk"] == 1]
    if "patient_id" in high_risk.columns:
        high_risk = (
            high_risk[["patient_id", "disease_group", "bmi", "systolic", "diastolic"]]
            .drop_duplicates("patient_id")
            .sort_values(["systolic", "bmi"], ascending=False)
        )

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
    st.markdown("#### 📋 Patient Profile & Recommendation")
    if sel_idx and avail_df is not None:
        sel_pid = avail_df.iloc[sel_idx[0]]["patient_id"]
        pt_data = summary_pts.loc[sel_pid]
        score, reasons_html, pkg_name, total_price, screenings = _analyze_patient_risk(pt_data)
        gender_icon = "👩" if pt_data["gender_code"] > 0.5 else "👨"
        if score <= 60:
            c_tx, badge = "#B3261E", "🚨 High Risk"
        elif score <= 80:
            c_tx, badge = "#B54708", "⚠️ Medium Risk"
        else:
            c_tx, badge = "#065F46", "🌱 Low Risk"
        st.markdown(f"{gender_icon} **{sel_pid}** — อายุ: {pt_data['age_at_visit']:.0f} ปี | :{badge}:")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 ข้อมูลสุขภาพ", "🏥 ประวัติการวินิจฉัย", "💎 แผนการตรวจที่แนะนำ", "🤖 AI Predict (XGBoost)"])

        with tab1:
            st.metric("Health Score", f"{score}%")
            st.info(f"💡 AI Analysis: {reasons_html}")

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
            st.markdown(f"**Package หลัก:** {pkg_name} — ฿ {pkg_info.get('price', 0):,.0f}")
            if pkg_info.get("tests"):
                for t in pkg_info["tests"]:
                    st.markdown(f"• {t}")
            if screenings:
                st.markdown("**🔍 Add-on เฉพาะบุคคล:**")
                for sc in screenings:
                    st.markdown(f"- {sc}")
            st.markdown(f"**รวมประเมินราคา: ฿ {total_price:,.0f}**")

        with tab4:
            with st.expander("⚙️ ปรับแต่งพารามิเตอร์ (XGBoost Tuning)"):
                x_col1, x_col2, x_col3 = st.columns(3)
                with x_col1:
                    max_depth = st.slider("Max Depth", 1, 15, 6, 1)
                with x_col2:
                    lr = st.slider("Learning Rate", 0.01, 0.50, 0.30, 0.01)
                with x_col3:
                    n_est = st.slider("N Estimators", 50, 500, 100, 50)

            xgb_model, explainer, metrics = get_upsell_risk_models(summary_pts, max_depth, lr, n_est)
            if xgb_model and explainer:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
                m2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.1f}%")
                m3.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
                m4.metric("Recall", f"{metrics.get('recall', 0):.1f}%")

                features = ["age_at_visit", "bmi", "systolic", "gender_code", "visits"]
                pt_df = pd.DataFrame([pt_data])[features].fillna(0)

                prob = xgb_model.predict_proba(pt_df)[0][1] * 100

                st.markdown("#### 🎯 Upsell Probability Score")
                if prob >= 70:
                    st.error(f"🔥 Hot Lead: {prob:.1f}%")
                elif prob >= 40:
                    st.warning(f"⚡ Warm Lead: {prob:.1f}%")
                else:
                    st.success(f"🌱 Cold Lead: {prob:.1f}%")

                st.markdown("##### 🧠 SHAP Explainability (Why?)")
                shap_values = explainer(pt_df)
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            else:
                st.info("⚠️ XGBoost or SHAP models are currently unavailable (missing library or training data).")
    else:
        st.info("👈 คลิกเลือกผู้ป่วยจากตารางด้านซ้าย เพื่อดู Profile")


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
        mean_crit = float(monthly_stats["critical_cases"].mean()) if not monthly_stats.empty else mean_c * 0.20
        seasonal_base = [1.02, 0.95, 0.97, 0.91, 1.00, 1.05, 1.08, 1.11, 1.07, 1.04, 1.06, 1.09]
        np.random.seed(101)
        hist_periods = [f"2024-{m:02d}" for m in range(1, 13)]
        hist_cases = [int(round(mean_c * 0.88 * s * (1 + np.random.normal(0, 0.02)))) for s in seasonal_base]
        hist_critical = [int(round(mean_crit * 0.86 * s * (1 + np.random.normal(0, 0.025)))) for s in seasonal_base]

    forecast_periods = [f"{y}-{m:02d}" for y in [2025, 2026] for m in range(1, 13)]
    seasonal_pattern = {1: 1.06, 2: 0.94, 3: 0.96, 4: 0.89, 5: 1.01, 6: 1.07, 7: 1.10, 8: 1.13, 9: 1.08, 10: 1.05, 11: 1.08, 12: 1.12}

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        scenario = st.radio("🎯 แผนสถานการณ์พยากรณ์:", ["📈 มาตรฐาน", "🚀 เชิงรุก (+10%)", "🛡️ อนุรักษ์นิยม (-5%)"], horizontal=True, key="fc_scenario")
    with ctrl_col2:
        show_ci_band = st.checkbox("🛡️ แสดง Confidence Band", value=True, key="fc_show_ci")
    with ctrl_col3:
        show_crit_line = st.checkbox("⚠️ แสดงกลุ่มเสี่ยง NCDs", value=True, key="fc_show_crit")

    with st.expander("⚙️ ปรับแต่งพารามิเตอร์ (Prophet Tuning)"):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            cps = st.slider("Changepoint Prior Scale", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
        with p_col2:
            sps = st.slider("Seasonality Prior Scale", min_value=0.01, max_value=20.0, value=10.0, step=0.5)

    scenario_mult = 1.10 if "เชิงรุก" in scenario else (0.95 if "อนุรักษ์นิยม" in scenario else 1.00)

    forecast_data = []
    prophet_model, prophet_mape = get_business_forecast_model(monthly_stats, cps, sps)

    if prophet_model is not None:
        future_dates = pd.DataFrame({"ds": pd.to_datetime(forecast_periods)})
        forecast = prophet_model.predict(future_dates)

        mean_crit_ratio = np.mean(hist_critical) / np.mean(hist_cases) if np.mean(hist_cases) > 0 else 0.20

        for i, row in forecast.iterrows():
            y_int, m_int = row['ds'].year, row['ds'].month

            projected = int(round(row['yhat'] * scenario_mult))
            lower = int(round(row['yhat_lower'] * scenario_mult))
            upper = int(round(row['yhat_upper'] * scenario_mult))

            proj_crit = int(round(projected * mean_crit_ratio * (1.05 if m_int >= 7 else 0.97)))

            forecast_data.append({
                "period": f"{y_int}-{m_int:02d}", "year": y_int, "month": m_int,
                "projected_cases": max(100, projected),
                "lower_bound": max(50, lower),
                "upper_bound": upper,
                "projected_critical": max(10, proj_crit)
            })
    else:
        x_hist = np.arange(len(hist_cases))
        slope_tot, intercept_tot = np.polyfit(x_hist, np.array(hist_cases, dtype=float), 1)
        slope_crit, intercept_crit = np.polyfit(x_hist, np.array(hist_critical, dtype=float), 1)
        if slope_tot <= 0: slope_tot = np.mean(hist_cases) * 0.015
        if slope_crit <= 0: slope_crit = np.mean(hist_critical) * 0.018

        np.random.seed(42)
        base_step = len(hist_cases)
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

    st.markdown("### 🔮 พยากรณ์แนวโน้มสุขภาพ (AI Forecast 2025-2026)")
    if prophet_model is not None:
        st.info(f"💡 ความน่าเชื่อถือของโมเดล: ความแม่นยำเฉลี่ย {100-prophet_mape:.1f}% (คลาดเคลื่อน ±{prophet_mape:.1f}%)")
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 คาดการณ์ผู้รับบริการรวม (24 เดือน)", f"{tot_24m:,} เคส", f"ปี 68: {tot_2025:,} | ปี 69: {tot_2026:,}")
    c2.metric("📈 อัตราการเติบโตคาดการณ์", f"{((tot_2026 - tot_2025) / tot_2025 * 100):+.1f}% YoY")
    c3.metric("⚠️ คาดการณ์กลุ่มเสี่ยงวิกฤต (NCDs)", f"{(tot_crit_24m / tot_24m * 100):.1f}%", f"{tot_crit_24m:,} คน", delta_color="inverse")

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

    st.markdown("---")

    st.markdown("#### 🎯 บทวิเคราะห์เชิงกลยุทธ์ (Strategic Recommendations)")
    trend_desc = "เพิ่มขึ้น" if scenario_mult >= 1.0 else "ทรงตัว/ลดลง"
    promo_focus = "เน้นการเจาะตลาดกลุ่มใหม่และแพคเกจครอบครัว" if scenario_mult >= 1.0 else "เน้นโปรโมชั่นรักษาฐานลูกค้าเก่า"

    t1, t2, t3 = st.tabs(["🎁 แผนโปรโมชั่น", "👨‍⚕️ การจัดอัตรากำลัง (Staff/Lab)", "🛡️ การดูแลเชิงป้องกัน"])
    with t1:
        st.info(f"**แนวโน้มผู้ป่วย{trend_desc}:** {promo_focus}\n\n- เสนอ **Health Checkup Add-on** สำหรับกลุ่มเสี่ยง\n- จัดแคมเปญกระตุ้นยอดในเดือน Low Season")
    with t2:
        st.warning(f"**เตรียมพร้อมรับมือ:**\n\n- คาดการณ์ผู้ป่วยเฉลี่ย **{int(tot_24m/24):,} เคส/เดือน**\n- จัดสรรพยาบาล/แพทย์ให้สอดคล้องกับช่วงพีค\n- ห้อง Lab ควรเตรียม Resource ให้เพียงพอต่อผู้ป่วย NCDs ({tot_crit_24m:,} คน)")
    with t3:
        st.success(f"**ลดอัตราผู้ป่วยวิกฤต:**\n\n- สัดส่วนกลุ่มเสี่ยงอยู่ที่ **{(tot_crit_24m / tot_24m * 100):.1f}%**\n- สร้างโปรแกรม NCDs Clinic เพื่อติดตามอาการ\n- จัดกิจกรรมให้ความรู้เชิงป้องกัน (Preventive Education)")

    st.markdown("#### 📊 ตารางผลพยากรณ์")
    fc_df = pd.DataFrame(forecast_data)
    fc_df.columns = ["งวด (ปี-เดือน)", "ปี", "เดือน", "คาดการณ์ผู้ป่วยรวม", "ขอบเขตล่าง", "ขอบเขตบน", "กลุ่มเสี่ยงวิกฤต (NCDs)"]
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    csv_data = fc_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 ดาวน์โหลดผลพยากรณ์ (CSV)",
        data=csv_data,
        file_name="forecast_results.csv",
        mime="text/csv",
        use_container_width=True
    )


@st.fragment
def render_other_packages_dashboard(dv, df, search_term):
    if "disease_group" in dv.columns:
        other_condition = (dv["disease_group"] == "อื่น ๆ") | (dv["disease_group"].isna()) | (dv["disease_group"] == "ไม่ระบุ") | (dv["disease_group"] == "ทั่วไป")
    else:
        other_condition = pd.Series(True, index=dv.index)
    other_dv = dv[other_condition].copy()
    if search_term:
        other_dv = other_dv[other_dv.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)]

    tab_brochure, tab_raw = st.tabs(["🏥 แพคเกจตรวจสุขภาพ 4 กลุ่ม (AI Architect)", "📊 ข้อมูลผู้รับบริการแบบตาราง"])
    with tab_raw:
        st.dataframe(other_dv, use_container_width=True)

    with tab_brochure:
        st.markdown("### 🏥 แพคเกจตรวจสุขภาพ 4 กลุ่ม (Vichaivej Omnoi)")
        if other_dv.empty:
            st.warning("⚠️ ไม่พบรายชื่อผู้รับบริการ")
            return
        avail_pids = other_dv["patient_id"].dropna().unique().tolist() if "patient_id" in other_dv.columns else [f"PT-{i+1:03d}" for i in range(min(10, len(other_dv)))]
        sel_pid = st.selectbox("👤 เลือกผู้รับบริการที่ต้องการวิเคราะห์:", options=avail_pids, index=0)

        pt_records = df[df["patient_id"] == sel_pid] if "patient_id" in df.columns else other_dv.iloc[0:1]
        pt_row = pt_records.iloc[-1].to_dict() if not pt_records.empty else other_dv.iloc[0].to_dict()
        pt_row["visits"] = len(pt_records)

        assigned_tier, risk_reasons, best_match_sub_pkg = _assess_patient_tier(pt_row)
        st.success(f"**AI Recommendation:** แนะนำ {HEALTH_PACKAGES_4LEVEL[assigned_tier]['name']} (Best Match: {best_match_sub_pkg})")

        cols = st.columns(4)
        for col, level_key in zip(cols, [1, 2, 3, 4]):
            group = HEALTH_PACKAGES_4LEVEL[level_key]
            with col:
                is_tier_match = (assigned_tier == level_key)
                border_style = f"border: 3px solid {group['header_bg']};" if is_tier_match else ""
                st.markdown(f"<div style='{border_style} padding:8px; border-radius:8px;'><b>{group['short_title']}</b><br><small>{group['target_audience']}</small></div>", unsafe_allow_html=True)
                for sub in group["sub_packages"]:
                    if st.button(f"เลือก {sub['short_name']} ({sub['price']:,}฿)", key=f"btn_{level_key}_{sub['short_name']}_{sel_pid}", use_container_width=True):
                        st.toast(f"✅ เลือก {sub['name']} เรียบร้อย")


@st.fragment
def render_disease_center(dv, df, selected_tab_key, active_config, search_term):
    disease_df = dv[active_config["filter_condition"](dv)].copy()
    st.markdown(f"### {active_config['icon']} {selected_tab_key} Command Center")
    st.caption(f"เป้าหมายการรักษา: {active_config['target_desc']}")

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

        selection = st.dataframe(sort_df[show_cols].style.apply(highlight_priority, axis=1), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        sel_idx = selection.selection.rows
        if sel_idx:
            sel_pid = sort_df.iloc[sel_idx[0]]["patient_id"]
            st.markdown(f"### 💎 แนะนำ Combined Care Package สำหรับ: {sel_pid}")
            if st.button(f"✨ Generate {selected_tab_key} Care Package"):
                st.success(f"✅ **สร้าง {selected_tab_key} Package สำเร็จ!**")
    else:
        st.info("ไม่พบคนไข้ในกลุ่มนี้")


def render_clinical_command_center(df, summary_pts, filters, data_source):
    """Top-level router for the Clinical Command Center module (ported from
    the original standalone analytics dashboard)."""
    if data_source == "demo":
        st.caption("ℹ️ ไม่พบไฟล์ visits_cleaned.csv/.zip ในระบบ — กำลังแสดงผลด้วยข้อมูลจำลอง (Demo Data) เพื่อการสาธิต")

    tab_options = [f"{v['icon']} {k}" for k, v in DISEASE_CONFIG.items()]
    selected_tab_str = st.radio(" ", tab_options, horizontal=True, label_visibility="collapsed", key="clinical_tab_select")
    selected_tab_key = selected_tab_str.split(" ", 1)[1]
    active_config = DISEASE_CONFIG[selected_tab_key]

    search_term = filters.get("search_term", "") if filters else ""
    disease_sel = filters.get("disease_sel", sorted(df["disease_group"].unique())) if filters else sorted(df["disease_group"].unique())
    gender_sel = filters.get("gender_sel", sorted(df["gender"].unique())) if filters else sorted(df["gender"].unique())
    clinic_sel = filters.get("clinic_sel", sorted(df["clinic_name"].dropna().unique())) if filters else sorted(df["clinic_name"].dropna().unique())

    mask = (df["disease_group"].isin(disease_sel) & df["gender"].isin(gender_sel) & df["clinic_name"].isin(clinic_sel))
    dv = df[mask].copy()

    if dv.empty:
        st.warning("⚠️ ไม่มีข้อมูลตามตัวกรองที่เลือก")
        return

    if active_config.get("is_general"):
        as_of = df["visit_date"].max()
        as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "ไม่ระบุ"
        st.markdown(f"## 🏥 Clinical Command Center")
        st.caption(f"ข้อมูลล่าสุด {as_of_str}")

        total_v = len(dv)
        uniq_pts = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v
        st.markdown("### สรุปตัวชี้วัด (KPIs)")
        k1, k2, k3 = st.columns(3)
        k1.metric("จำนวนเคสรับบริการ", f"{total_v:,}")
        k2.metric("จำนวนผู้รับบริการ", f"{uniq_pts:,}")
        k3.metric("ความดันโลหิตเฉลี่ย", f"{dv['systolic'].mean():.1f} mmHg")

        st.divider()

        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("#### ⚡ Command Action Panel")
            render_action_panel(dv)
        with c2:
            st.markdown("#### 🔍 เลือกผู้ป่วยเพื่อประเมิน Package")
            if summary_pts is not None and "patient_id" in dv.columns:
                avail_df = summary_pts[summary_pts.index.isin(dv["patient_id"].values)].reset_index()
                selection = st.dataframe(avail_df[["patient_id", "age_at_visit", "bmi"]], use_container_width=True, hide_index=True, height=220, on_select="rerun", selection_mode="single-row")
                sel_idx = selection.selection.rows
            else:
                avail_df, sel_idx = None, []
            render_patient_profile(avail_df, summary_pts, dv, sel_idx)

    elif active_config.get("is_forecast"):
        render_forecast_dashboard(dv)

    elif active_config.get("is_other_packages"):
        render_other_packages_dashboard(dv, df, search_term)

    else:
        render_disease_center(dv, df, selected_tab_key, active_config, search_term)


# ============================================================
# CRM / Sales Demo Data Generation
# ============================================================
@st.cache_data
def generate_demo_crm_data():
    np.random.seed(42)
    random.seed(42)

    # 1. Package Catalog
    packages = pd.DataFrame([
        {"package_id": "PKG-01", "package_group_no": 1, "package_group_name": "ตรวจสุขภาพทั่วไป", "package_name": "BASIC", "price": 990, "description": "พื้นฐานสำหรับผู้เริ่มต้น", "suitable_for": "ทุกวัย", "included_tests": "CBC, FBS, Lipid", "active_flag": 1},
        {"package_id": "PKG-02", "package_group_no": 1, "package_group_name": "ตรวจสุขภาพทั่วไป", "package_name": "STANDARD", "price": 2290, "description": "ครอบคลุมเบื้องต้น", "suitable_for": "ทุกวัย", "included_tests": "CBC, FBS, Lipid, LFT, KFT", "active_flag": 1},
        {"package_id": "PKG-03", "package_group_no": 1, "package_group_name": "ตรวจสุขภาพทั่วไป", "package_name": "PREMIUM", "price": 4990, "description": "ตรวจสุขภาพประจำปีอย่างละเอียด", "suitable_for": "ทุกวัย", "included_tests": "CBC, FBS, Lipid, LFT, KFT, CXR, EKG, U/S", "active_flag": 1},

        {"package_id": "PKG-04", "package_group_no": 2, "package_group_name": "ยังไม่พบโรคแต่มีความเสี่ยง", "package_name": "9.9 METABOLIC", "price": 2990, "description": "กลุ่มเสี่ยง Metabolic", "suitable_for": "BMI>25 หรือ BP>130", "included_tests": "HbA1c, Lipid Profile, Uric", "active_flag": 1},
        {"package_id": "PKG-05", "package_group_no": 2, "package_group_name": "ยังไม่พบโรคแต่มีความเสี่ยง", "package_name": "คนอ้วน", "price": 2990, "description": "ประเมินสุขภาพคนอ้วน", "suitable_for": "BMI>25", "included_tests": "Lipid Profile, LFT, Thyroid", "active_flag": 1},
        {"package_id": "PKG-06", "package_group_no": 2, "package_group_name": "ยังไม่พบโรคแต่มีความเสี่ยง", "package_name": "วัย 35+", "price": 3990, "description": "เตรียมพร้อมวัยกลางคน", "suitable_for": "อายุ 35 ปีขึ้นไป", "included_tests": "Hormones, Bone Density", "active_flag": 1},

        {"package_id": "PKG-07", "package_group_no": 3, "package_group_name": "เริ่มมีความผิดปกติ", "package_name": "เบาหวาน", "price": 3290, "description": "ติดตามเบาหวาน", "suitable_for": "น้ำตาลสูง", "included_tests": "FBS, HbA1c, Microalbumin", "active_flag": 1},
        {"package_id": "PKG-08", "package_group_no": 3, "package_group_name": "เริ่มมีความผิดปกติ", "package_name": "ความดันโลหิตสูง", "price": 2990, "description": "ติดตามความดัน", "suitable_for": "BP>140/90", "included_tests": "EKG, KFT, Electrolyte", "active_flag": 1},
        {"package_id": "PKG-09", "package_group_no": 3, "package_group_name": "เริ่มมีความผิดปกติ", "package_name": "ไขมันในเลือดสูง", "price": 2990, "description": "ติดตามไขมัน", "suitable_for": "Chol>200", "included_tests": "Lipid Profile, LFT", "active_flag": 1},

        {"package_id": "PKG-10", "package_group_no": 4, "package_group_name": "มีอาการซ้ำ ๆ / เชิงลึก", "package_name": "คัดกรองมะเร็ง", "price": 4990, "description": "ตรวจสารบ่งชี้มะเร็ง", "suitable_for": "ประวัติครอบครัว", "included_tests": "Tumor Markers", "active_flag": 1},
        {"package_id": "PKG-11", "package_group_no": 4, "package_group_name": "มีอาการซ้ำ ๆ / เชิงลึก", "package_name": "หัวใจ", "price": 4990, "description": "สุขภาพหัวใจ", "suitable_for": "เจ็บหน้าอก, ใจสั่น", "included_tests": "EST, Echo", "active_flag": 1},
        {"package_id": "PKG-12", "package_group_no": 4, "package_group_name": "มีอาการซ้ำ ๆ / เชิงลึก", "package_name": "Colonoscopy", "price": 8900, "description": "ส่องกล้องลำไส้ใหญ่", "suitable_for": "อายุ>50, ขับถ่ายผิดปกติ", "included_tests": "Colonoscopy", "active_flag": 1},
    ])

    # 2. Customers
    n_customers = 500
    customers_data = []
    branches = ["อ้อมน้อย", "หนองแขม", "สมุทรสาคร", "แยกไฟฉาย"]
    channels = ["Walk-in", "Phone", "LINE", "Website", "Corporate", "Partner"]
    health_interests = ["หัวใจ", "เบาหวาน", "มะเร็ง", "สุขภาพผู้หญิง", "สุขภาพผู้ชาย", "ตรวจสุขภาพทั่วไป"]

    for i in range(n_customers):
        age = int(np.random.normal(45, 15))
        age = max(18, min(age, 85))
        c_id = f"CUS-{i+1000}"
        p_id = f"HN-{i+50000}" if random.random() > 0.3 else None

        customers_data.append({
            "customer_id": c_id,
            "patient_id": p_id,
            "full_name": f"Customer_{i}",
            "age": age,
            "gender": random.choice(["Male", "Female"]),
            "phone_masked": f"08{random.randint(0,9)}-XXX-{random.randint(1000,9999)}",
            "email_masked": f"c***{i}@email.com",
            "province": random.choice(["กรุงเทพมหานคร", "สมุทรสาคร", "นครปฐม", "นนทบุรี"]),
            "preferred_branch": random.choice(branches),
            "preferred_contact_channel": random.choice(channels),
            "health_interests": random.choice(health_interests),
            "budget_range": random.choice(["< 3,000", "3,000-5,000", "> 5,000"]),
            "consent_status": random.choice(["ยินยอมแล้ว", "รอตรวจสอบความยินยอม", "ไม่อนุญาตใช้เพื่อการตลาด"]),
            "marketing_consent": random.choice([True, False]),
            "created_at": datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        })
    customers = pd.DataFrame(customers_data)

    # 3. Sales Transactions
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 31)
    n_days = (end_date - start_date).days

    n_transactions = 3000
    sales_data = []

    for i in range(n_transactions):
        rand_days = random.randint(0, n_days)
        sale_date = start_date + timedelta(days=rand_days)
        month = sale_date.month
        year = sale_date.year

        cust = customers.iloc[random.randint(0, n_customers-1)]
        pkg = packages.iloc[random.randint(0, len(packages)-1)]

        discount = random.choice([0, 0, 0, 0.1, 0.15, 0.2]) * pkg["price"]

        sales_data.append({
            "transaction_id": f"TRX-{i+10000}",
            "sale_date": sale_date,
            "customer_id": cust["customer_id"],
            "branch_id": f"BR-{branches.index(cust['preferred_branch'])+1}",
            "branch_name": cust["preferred_branch"],
            "channel": random.choice(channels),
            "campaign_id": f"CMP-{year}-{month}",
            "campaign_name": f"Promo {year}-{month}",
            "package_id": pkg["package_id"],
            "package_name": pkg["package_name"],
            "package_group": pkg["package_group_name"],
            "list_price": pkg["price"],
            "discount_amount": discount,
            "net_amount": pkg["price"] - discount,
            "payment_status": random.choice(["Paid", "Paid", "Paid", "Pending"]),
            "sales_staff_id": f"STF-{random.randint(1,20)}",
            "sales_staff_name": f"Staff_{random.randint(1,20)}",
            "lead_status": random.choice(["Purchased", "Purchased", "Purchased", "Contacted", "Interested"]),
            "appointment_status": random.choice(["Scheduled", "Completed", "None"]),
            "package_usage_status": random.choice(["Used", "Unused"])
        })
    sales_transactions = pd.DataFrame(sales_data)

    # 4. Clinical Visits (CRM-side snapshot; used for Sales / Insight Card modules)
    clinical_data = []
    patients = customers[customers["patient_id"].notna()]

    for _, pt in patients.iterrows():
        n_visits = random.randint(1, 5)
        for v in range(n_visits):
            visit_date = pt["created_at"] + timedelta(days=random.randint(10, 700))
            if visit_date > end_date: continue

            sys_bp = int(np.random.normal(120 if pt["age"] < 40 else 135, 15))
            bmi = np.random.normal(23 if pt["age"] < 40 else 26, 4)

            clinical_data.append({
                "visit_id": f"VIS-{uuid.uuid4().hex[:6].upper()}",
                "patient_id": pt["patient_id"],
                "visit_date": visit_date,
                "branch_id": f"BR-{branches.index(pt['preferred_branch'])+1}",
                "appointment_id": f"APP-{uuid.uuid4().hex[:6].upper()}",
                "appointment_status": "Completed",
                "diagnosis_text": random.choice(["Essential Hypertension", "Type 2 DM", "Normal", "Dyslipidemia"]),
                "systolic_bp": sys_bp,
                "diastolic_bp": int(sys_bp * 0.6),
                "weight_kg": int(bmi * (1.65**2)),
                "height_cm": 165,
                "bmi": bmi,
                "waist_cm": int(bmi * 3.2),
                "screening_result_status": random.choice(["Normal", "Needs Review", "Referred", "Pending"]),
                "follow_up_date": visit_date + timedelta(days=random.choice([30, 90, 180])) if random.random() > 0.5 else None,
                "follow_up_status": random.choice(["Pending", "Completed", "Overdue"]),
                "referral_status": "None" if random.random() > 0.1 else "Referred to Specialist",
                "clinician_note": "Patient presents with normal vitals. Advised on diet."
            })
    clinical_visits = pd.DataFrame(clinical_data)

    # 5. Appointments & Recommendations (Empty init for demo)
    appointments = pd.DataFrame(columns=["appointment_id", "patient_id", "customer_id", "appointment_date", "branch_id", "appointment_type", "appointment_status", "no_show_flag", "package_id", "created_by"])
    package_recommendations = pd.DataFrame(columns=["recommendation_id", "created_at", "customer_id", "patient_id", "recommended_package_id", "recommendation_source", "recommendation_reason", "match_score", "sales_status", "next_action", "next_action_date", "owner_staff_id", "consent_checked"])

    # 6. Sales Targets
    targets_data = []
    for y in [2024, 2025, 2026]:
        for m in range(1, 13):
            for b in branches:
                targets_data.append({
                    "target_month": f"{y}-{m:02d}",
                    "branch_name": b,
                    "target_revenue": 50000 + random.randint(-10000, 20000),
                    "target_packages_sold": 20 + random.randint(-5, 10)
                })
    sales_targets = pd.DataFrame(targets_data)

    return packages, customers, sales_transactions, clinical_visits, appointments, package_recommendations, sales_targets


@st.cache_data
def load_crm_data():
    return generate_demo_crm_data()


def apply_global_filters(df, date_col, start_d, end_d, branch, channel):
    if df is None or df.empty or date_col not in df.columns:
        return df

    mask = (df[date_col].dt.date >= start_d) & (df[date_col].dt.date <= end_d)

    if branch != "All":
        if "branch_name" in df.columns:
            mask &= (df["branch_name"] == branch)
    if channel != "All":
        if "channel" in df.columns:
            mask &= (df["channel"] == channel)

    return df[mask]


# ============================================================
# CRM Recommendation Engine (Sales / Insight Card)
# ============================================================
def _recommend_packages_crm(customer_row, clinical_row_or_none, role, packages_df):
    results = []

    # 1. Consent Check
    consent = customer_row.get("consent_status", "")
    if consent == "ไม่อนุญาตใช้เพื่อการตลาด" and role == "Sales":
        return [{"pkg": None, "score": 0, "reason": "ลูกค้าไม่อนุญาตให้ใช้ข้อมูลเพื่อการตลาด (Consent Denied)"}]

    age = customer_row.get("age", 35)
    interests = str(customer_row.get("health_interests", ""))
    budget = str(customer_row.get("budget_range", ""))

    for _, pkg in packages_df.iterrows():
        score = 50  # Base score
        reasons_safe = []
        reasons_clin = []

        # 2. Health Interests
        if "มะเร็ง" in interests and "มะเร็ง" in pkg["package_name"]:
            score += 30; reasons_safe.append("ตรงกับความสนใจด้านโรคมะเร็ง")
        if "หัวใจ" in interests and "หัวใจ" in pkg["package_name"]:
            score += 30; reasons_safe.append("ตรงกับความสนใจด้านโรคหัวใจ")

        # 3. Demographics & Budget
        if age >= 35 and "35+" in pkg["package_name"]:
            score += 20; reasons_safe.append("เหมาะสมกับช่วงอายุ 35 ปีขึ้นไป")
        if budget == "< 3,000" and pkg["price"] < 3000:
            score += 10; reasons_safe.append("อยู่ในช่วงงบประมาณที่กำหนด")

        # 5. Clinical Risks (Role-based)
        if clinical_row_or_none is not None and role in ["Clinical", "Admin"]:
            sys_bp = clinical_row_or_none.get("systolic_bp", 120)
            bmi = clinical_row_or_none.get("bmi", 22)
            scr = clinical_row_or_none.get("screening_result_status", "")

            if sys_bp >= 140 and "ความดัน" in pkg["package_name"]:
                score += 40; reasons_clin.append("ค่าความดันโลหิตสูงกว่าเกณฑ์ เหมาะสำหรับการประเมินเพิ่มเติม")
            if bmi >= 25 and "อ้วน" in pkg["package_name"]:
                score += 40; reasons_clin.append("ดัชนีมวลกาย (BMI) สูง เหมาะสำหรับการประเมินปัจจัยสุขภาพเพิ่มเติม")
            if scr in ["Needs Review", "Referred"]:
                if pkg["package_group_no"] == 4:
                    score += 50; reasons_clin.append("ผลคัดกรองเบื้องต้นควรได้รับการประเมินเชิงลึกโดยแพทย์เฉพาะทาง")

        # Apply bounds
        score = min(100, max(0, score))

        match_level = "สูง" if score >= 80 else ("ปานกลาง" if score >= 60 else "พื้นฐาน")

        final_reasons = reasons_safe
        if role in ["Clinical", "Admin"]:
            final_reasons.extend(reasons_clin)

        if not final_reasons:
            final_reasons = ["เป็นแพ็กเกจตรวจสุขภาพมาตรฐานที่ครอบคลุม"]

        results.append({
            "pkg": pkg,
            "score": score,
            "match_level": match_level,
            "reasons": final_reasons
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3]  # Return top 3 (1 Best match, 2 alternatives)


# ============================================================
# Insight Card Module
# ============================================================
def render_insight_card(customer_id, role, customers, sales, clinical, appointments, packages):
    st.markdown("### 👤 Patient/Customer Insight Card")

    cust_row = customers[customers["customer_id"] == customer_id]
    if cust_row.empty:
        st.error("ไม่พบข้อมูลลูกค้า")
        return
    cust_row = cust_row.iloc[0]

    pt_id = cust_row.get("patient_id")
    clin_row = None
    if pt_id and pd.notna(pt_id) and not clinical[clinical["patient_id"] == pt_id].empty:
        clin_row = clinical[clinical["patient_id"] == pt_id].sort_values("visit_date", ascending=False).iloc[0]

    cust_sales = sales[sales["customer_id"] == customer_id]
    cust_apps = appointments[appointments["customer_id"] == customer_id]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"#### {cust_row['full_name']}")
        st.caption(f"CUS ID: {customer_id} | PT ID: {pt_id if pd.notna(pt_id) else 'ไม่มี'}")
    with c2:
        consent_color = "success" if cust_row["consent_status"] == "ยินยอมแล้ว" else ("danger" if "ไม่อนุญาต" in cust_row["consent_status"] else "warning")
        st.markdown(get_badge_html(cust_row["consent_status"], consent_color), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ภาพรวม", "แพ็กเกจแนะนำ", "การขายและการติดตาม", "นัดหมายและการใช้สิทธิ์", "ข้อมูลสุขภาพ"])

    with tab1:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("อายุ", f"{cust_row['age']} ปี")
        s2.metric("เพศ", cust_row['gender'])
        s3.metric("สาขาที่สะดวก", cust_row['preferred_branch'])
        s4.metric("งบประมาณ", cust_row['budget_range'])

        st.markdown(f"**เบอร์โทร (Masked):** {cust_row['phone_masked']}")
        st.markdown(f"**ความสนใจ:** {cust_row['health_interests']}")

        if role == "Sales":
            if clin_row is not None and pd.notna(clin_row.get('follow_up_date')):
                st.info("💡 ข้อบ่งชี้: ผู้รับบริการมีสถานะต้องติดตามสุขภาพ (อ้างอิงจากระบบคลินิก)")

    with tab2:
        st.markdown("#### 💡 แพ็กเกจที่แนะนำ (AI/Rule-based)")
        recs = _recommend_packages_crm(cust_row, clin_row, role, packages)
        if recs and recs[0]["pkg"] is not None:
            rcols = st.columns(3)
            for i, rec in enumerate(recs):
                with rcols[i]:
                    pkg = rec["pkg"]
                    css_class = "recommendation-card best-match" if i == 0 else "recommendation-card"
                    match_badge = get_badge_html(f"Match: {rec['match_level']}", "success" if i == 0 else "info")

                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="font-size: 0.8rem; color: #64748b;">กลุ่ม {pkg['package_group_no']}: {pkg['package_group_name']}</div>
                        <h4 style="margin: 5px 0; color: {COLORS['Navy']}">{pkg['package_name']}</h4>
                        <div style="font-weight: bold; color: {COLORS['Teal']}; font-size: 1.2rem; margin-bottom: 10px;">{format_currency(pkg['price'])}</div>
                        {match_badge}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**เหตุผล:**")
                    for reason in rec["reasons"]:
                        st.markdown(f"- {reason}")

                    if st.button("บันทึกความสนใจ", key=f"btn_interest_{pkg['package_id']}"):
                        log_audit("Save Interest", f"Package {pkg['package_id']} for {customer_id}", role)
                        st.success("บันทึกแล้ว!")

            st.caption("คำแนะนำนี้เป็นข้อมูลเพื่อประกอบการเลือกแพ็กเกจตรวจสุขภาพ ไม่ใช่การวินิจฉัยโรค")
        else:
            st.warning(recs[0]["reason"] if recs else "ไม่สามารถแนะนำแพ็กเกจได้")

    with tab3:
        st.markdown("#### ประวัติการติดต่อและยอดขาย")
        if not cust_sales.empty:
            st.dataframe(cust_sales[["sale_date", "package_name", "net_amount", "channel", "lead_status"]], use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีประวัติการซื้อ")

        st.markdown("#### บันทึกการติดต่อ (Log Note)")
        with st.form("contact_note_form"):
            note = st.text_area("บันทึกข้อความ")
            status = st.selectbox("สถานะ Lead", ["New", "Contacted", "Interested", "Appointed", "Declined"])
            sub = st.form_submit_button("บันทึก")
            if sub:
                log_audit("Update Sales Status", f"Status: {status}", role)
                st.success("บันทึกข้อมูลเรียบร้อย")

    with tab4:
        st.markdown("#### ประวัตินัดหมาย")
        if not cust_apps.empty:
            st.dataframe(cust_apps, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีประวัตินัดหมาย")

    with tab5:
        if role == "Sales":
            st.error("🚫 คุณไม่มีสิทธิ์เข้าถึงข้อมูลสุขภาพโดยละเอียด")
        else:
            st.markdown("#### ข้อมูลสุขภาพเชิงคลินิก")
            if clin_row is not None:
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("ความดัน (BP)", f"{clin_row['systolic_bp']}/{clin_row['diastolic_bp']}")
                cc2.metric("BMI", f"{clin_row['bmi']:.1f}")
                cc3.metric("ผลคัดกรอง", clin_row['screening_result_status'])

                st.markdown(f"**Diagnosis:** {clin_row['diagnosis_text']}")
                st.markdown(f"**Clinician Note:** {clin_row['clinician_note']}")
                st.markdown(f"**วันนัดติดตามผล:** {clin_row['follow_up_date'].strftime('%Y-%m-%d') if pd.notna(clin_row['follow_up_date']) else 'ไม่มี'}")
                st.caption("ข้อมูลเพื่อการติดตามโดยบุคลากรทางการแพทย์")

                log_audit("View Clinical Data", f"Viewed PT {pt_id}", role)
            else:
                st.info("ไม่พบประวัติการเข้ารับบริการ")


# ============================================================
# Sales Command Center Module
# ============================================================
def render_sales_dashboard(sales, targets, customers, packages):
    if sales.empty:
        st.warning("ไม่มีข้อมูลยอดขายในช่วงเวลาที่เลือก")
        return

    st.markdown("## 📈 Sales Command Center")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Overview", "Forecast & Scenario", "Campaign Performance",
        "Branch & Channel", "Package Analytics", "Sales Pipeline"
    ])

    total_rev = sales["net_amount"].sum()
    total_pkgs = len(sales)
    unique_cust = sales["customer_id"].nunique()
    aov = total_rev / total_pkgs if total_pkgs > 0 else 0

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("ยอดขายสุทธิ", format_currency(total_rev), 5.2)
        with c2: kpi_card("จำนวนแพ็กเกจ", f"{total_pkgs:,}", 3.1)
        with c3: kpi_card("ลูกค้าสั่งซื้อ", f"{unique_cust:,}")
        with c4: kpi_card("Average Order Value", format_currency(aov))

        sales['month_year'] = sales['sale_date'].dt.to_period('M').astype(str)
        monthly = sales.groupby('month_year')['net_amount'].sum().reset_index()
        fig_rev = px.line(monthly, x='month_year', y='net_amount', title="รายได้รายเดือน (Actual)", markers=True, color_discrete_sequence=[COLORS["Navy"]])

        top_pkg = sales.groupby('package_name')['net_amount'].sum().nlargest(10).reset_index()
        fig_top = px.bar(top_pkg, x='net_amount', y='package_name', orientation='h', title="Top 10 Packages", color_discrete_sequence=[COLORS["Teal"]])
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})

        r1, r2 = st.columns(2)
        r1.plotly_chart(fig_rev, use_container_width=True)
        r2.plotly_chart(fig_top, use_container_width=True)

    with tab2:
        st.markdown("### 🔮 Forecast & Scenario (Prophet)")
        if not PROPHET_AVAILABLE:
            st.warning("⚠️ Prophet library is not available. Falling back to linear projection (Not fully implemented in Demo).")

        fc1, fc2 = st.columns(2)
        horizon = fc1.selectbox("Forecast Horizon", [3, 6, 12], index=1)
        scenario = fc2.radio("Scenario", ["Conservative", "Standard", "Growth"], horizontal=True, index=1)

        mult = {"Conservative": 0.9, "Standard": 1.0, "Growth": 1.12}[scenario]

        if not monthly.empty and PROPHET_AVAILABLE:
            df_p = pd.DataFrame({'ds': pd.to_datetime(monthly['month_year']), 'y': monthly['net_amount']})
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=horizon, freq='M')
                forecast = m.predict(future)

                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat'] *= mult
                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat_lower'] *= mult
                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat_upper'] *= mult

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color=COLORS['Navy'])))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color=COLORS['Amber'], dash="dash")))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(245, 158, 11, 0.2)', name="Confidence Interval"))

                st.plotly_chart(fig_fc, use_container_width=True)

                train_fc = forecast[forecast['ds'] <= df_p['ds'].max()]
                mape = np.mean(np.abs((df_p['y'].values - train_fc['yhat'].values) / df_p['y'].values)) * 100
                st.info(f"💡 In-sample MAPE: {mape:.1f}%")
            except Exception as e:
                st.error(f"Forecast Error: {e}")

    with tab3:
        st.markdown("### 📢 Campaign Performance")
        camp = sales.groupby("campaign_name").agg(Revenue=("net_amount", "sum"), Packages=("transaction_id", "count")).reset_index()
        st.dataframe(camp.sort_values("Revenue", ascending=False), use_container_width=True)

    with tab4:
        st.markdown("### 🏢 Branch & Channel")
        b1, b2 = st.columns(2)
        branch_rev = sales.groupby("branch_name")["net_amount"].sum().reset_index()
        fig_b = px.pie(branch_rev, values="net_amount", names="branch_name", title="Revenue by Branch", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
        b1.plotly_chart(fig_b, use_container_width=True)

        channel_rev = sales.groupby("channel")["net_amount"].sum().reset_index()
        fig_c = px.bar(channel_rev, x="channel", y="net_amount", title="Revenue by Channel", color_discrete_sequence=[COLORS["Green"]])
        b2.plotly_chart(fig_c, use_container_width=True)

    with tab5:
        st.markdown("### 📦 Package Analytics")
        st.dataframe(sales.groupby("package_group").agg(Revenue=("net_amount", "sum"), Sold=("transaction_id", "count")).reset_index(), use_container_width=True)

    with tab6:
        st.markdown("### 🚦 Sales Pipeline")
        funnel = sales["lead_status"].value_counts().reset_index()
        funnel.columns = ["Status", "Count"]
        status_order = ["New", "Contacted", "Interested", "Appointed", "Purchased", "Declined"]
        funnel["Status"] = pd.Categorical(funnel["Status"], categories=status_order, ordered=True)
        funnel = funnel.sort_values("Status")

        fig_f = go.Figure(go.Funnel(y=funnel["Status"], x=funnel["Count"], marker={"color": [COLORS["Navy"], COLORS["Teal"], COLORS["Amber"], COLORS["Purple"], COLORS["Green"], COLORS["Red"]]}))
        st.plotly_chart(fig_f, use_container_width=True)


# ============================================================
# Main Application
# ============================================================
def main():
    inject_css()

    # CRM demo data — always available, powers Sales Command Center & Patient Insight Card
    crm_data = load_crm_data()
    if crm_data is None:
        st.error("Failed to load CRM demo data.")
        return
    packages, customers, sales_transactions, clinical_visits_demo, appointments, package_recommendations, sales_targets = crm_data

    # Clinical Command Center data — real visits_cleaned.csv/.zip if present,
    # otherwise falls back to the CRM demo data adapted to the same schema
    raw_clinical_df, clinical_data_source = load_clinical_df(customers, clinical_visits_demo)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
        st.markdown("### 🏥 Vichaivej Omnoi")
        st.markdown("#### Command Center")

        user_role = st.selectbox("🔑 เข้าสู่ระบบในฐานะ:", ["Sales", "Clinical", "Admin"])

        st.markdown("---")
        st.markdown("### 📍 Filters")
        date_range = st.date_input("ช่วงเวลา", [sales_transactions["sale_date"].min().date(), sales_transactions["sale_date"].max().date()])

        if len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = sales_transactions["sale_date"].min().date(), sales_transactions["sale_date"].max().date()

        branches_list = ["All"] + list(customers["preferred_branch"].unique())
        sel_branch = st.selectbox("สาขา", branches_list)

        channels_list = ["All"] + list(sales_transactions["channel"].unique())
        sel_channel = st.selectbox("ช่องทาง", channels_list)

        st.markdown("---")
        module = st.radio("📂 เลือกโมดูล", ["Sales Command Center", "Clinical Command Center", "Patient Insight Card"])

        clinical_filters = None
        if module == "Clinical Command Center" and raw_clinical_df is not None and not raw_clinical_df.empty:
            st.markdown("---")
            st.markdown(f"### 🎛️ Clinical Filters")
            search_term = st.text_input("🔍 ค้นหา (ID/ชื่อ)", key="ctx_search")
            all_diseases = sorted(raw_clinical_df["disease_group"].unique())
            disease_sel = st.multiselect("กลุ่มโรค", all_diseases, default=all_diseases)
            all_genders = sorted(raw_clinical_df["gender"].unique())
            gender_sel = st.multiselect("เพศ", all_genders, default=all_genders)
            all_clinics = sorted(raw_clinical_df["clinic_name"].dropna().unique())
            clinic_sel = st.multiselect("คลินิก", all_clinics, default=all_clinics)
            clinical_filters = {
                "search_term": search_term,
                "disease_sel": disease_sel,
                "gender_sel": gender_sel,
                "clinic_sel": clinic_sel,
            }

    # Apply global filters (date/branch/channel) to Sales data
    f_sales = apply_global_filters(sales_transactions, "sale_date", start_d, end_d, sel_branch, sel_channel)

    # ---------------- Routing ----------------
    if module == "Sales Command Center":
        render_sales_dashboard(f_sales, sales_targets, customers, packages)

    elif module == "Clinical Command Center":
        if user_role == "Sales":
            st.error("🚫 คุณไม่มีสิทธิ์เข้าถึง Clinical Command Center")
        elif raw_clinical_df is None or raw_clinical_df.empty:
            st.error("⚠️ ไม่พบข้อมูลสำหรับ Clinical Command Center กรุณาอัปโหลดไฟล์ visits_cleaned.csv")
        else:
            df_processed = process_patient_data(raw_clinical_df)
            summary_pts = build_summary_pts(df_processed) if "patient_id" in df_processed.columns else None
            render_clinical_command_center(df_processed, summary_pts, clinical_filters, clinical_data_source)

    elif module == "Patient Insight Card":
        st.markdown("## 🔍 ค้นหาข้อมูลผู้รับบริการ")
        search_q = st.text_input("กรอกชื่อ หรือ Customer ID (เช่น CUS-1000)")
        if search_q:
            res = customers[customers["customer_id"].str.contains(search_q, case=False) | customers["full_name"].str.contains(search_q, case=False)]
            if not res.empty:
                sel_cust = st.selectbox("เลือกลูกค้า", res["customer_id"].tolist(), format_func=lambda x: f"{x} - {res[res['customer_id']==x]['full_name'].values[0]}")
                if sel_cust:
                    render_insight_card(sel_cust, user_role, customers, f_sales, clinical_visits_demo, appointments, packages)
            else:
                st.warning("ไม่พบข้อมูล")


if __name__ == "__main__":
    main()
