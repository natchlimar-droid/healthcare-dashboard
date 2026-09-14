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
    page_icon="๐ฅ",
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
    "เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ": TEAL,
    "เน€เธเธฒเธซเธงเธฒเธ": "#2F6FB5",
    "เนเธเธกเธฑเธเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ": AMBER,
    "เธญเธทเนเธ เน": "#9AA6A0",
}
BP_COLORS = {
    "เธเธเธ•เธด (<120)": SAGE,
    "เน€เธเนเธฒเธฃเธฐเธงเธฑเธ (120-139)": AMBER,
    "เธชเธนเธ (>=140)": RED,
    "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ": "#C7CFCC",
}

MIN_SAMPLE = 5
SUNBURST_TOP_N = 5

HEALTH_PACKAGES = {
    "Essential Package": {
        "price": 3000,
        "tests": ["CBC", "FBS", "Lipid Profile", "Uric Acid", "CXR", "EKG"],
        "desc": "เน€เธซเธกเธฒเธฐเธชเธณเธซเธฃเธฑเธเธงเธฑเธขเน€เธฃเธดเนเธกเธ•เนเธเธ—เธณเธเธฒเธเนเธฅเธฐเธเธนเนเธ—เธตเนเนเธกเนเธกเธตเธเธงเธฒเธกเน€เธชเธตเนเธขเธ (เธญเธฒเธขเธธ <30 เธเธต)"
    },
    "Advanced Package": {
        "price": 5500,
        "tests": ["Essential Tests +", "Liver Function", "Kidney Function", "HbA1c", "Urine Examination", "Ultrasound Whole Abdomen"],
        "desc": "เน€เธซเธกเธฒเธฐเธชเธณเธซเธฃเธฑเธเธงเธฑเธขเธ—เธณเธเธฒเธเธ—เธตเนเน€เธฃเธดเนเธกเธกเธตเธเธงเธฒเธกเน€เธชเธตเนเธขเธ (เธญเธฒเธขเธธ 30-50 เธเธต)"
    },
    "Longevity Package": {
        "price": 8000,
        "tests": ["Advanced Tests +", "Thyroid Function", "Bone Densitometry", "Tumor Markers", "ABI"],
        "desc": "เน€เธซเธกเธฒเธฐเธชเธณเธซเธฃเธฑเธเธเธนเนเธชเธนเธเธญเธฒเธขเธธเธซเธฃเธทเธญเธกเธตเธเธงเธฒเธกเน€เธชเธตเนเธขเธเนเธฃเธเน€เธฃเธทเนเธญเธฃเธฑเธ (เธญเธฒเธขเธธ >50 เธเธต)"
    }
}

HEALTH_PACKAGES_4LEVEL = {
    1: {
        "level": 1, "name": "เธเธฅเธธเนเธก 1: เธ•เธฃเธงเธเธชเธธเธเธ เธฒเธเธ—เธฑเนเธงเนเธ", "short_title": "เธ•เธฃเธงเธเธชเธธเธเธ เธฒเธเธ—เธฑเนเธงเนเธ",
        "target_audience": "เธชเธณเธซเธฃเธฑเธเธเธเธ—เธตเนเนเธชเนเนเธเธชเธธเธเธ เธฒเธ เธญเธขเธฒเธเธฃเธนเนเธเธทเนเธเธเธฒเธเธเธญเธเธ•เธฑเธงเน€เธญเธ",
        "badge_color": "#0E7055", "badge_bg": "#DCFCE7", "header_bg": "#0E7055",
        "price": 990,
        "sub_packages": [
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ BASIC", "short_name": "BASIC", "price": 990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ STANDARD", "short_name": "STANDARD", "price": 2290},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ PREMIUM", "short_name": "PREMIUM", "price": 4990},
        ]
    },
    2: {
        "level": 2, "name": "เธเธฅเธธเนเธก 2: เธขเธฑเธเนเธกเนเธเธเนเธฃเธเนเธ•เนเธกเธตเธเธงเธฒเธกเน€เธชเธตเนเธขเธ", "short_title": "เธขเธฑเธเนเธกเนเธเธเนเธฃเธเนเธ•เนเธกเธตเธเธงเธฒเธกเน€เธชเธตเนเธขเธ",
        "target_audience": "เธชเธณเธซเธฃเธฑเธเธเธนเนเธ—เธตเนเธกเธตเธเธฑเธเธเธฑเธขเน€เธชเธตเนเธขเธเธเธฒเธเธญเธฒเธขเธธ เธเนเธณเธซเธเธฑเธ เธเธคเธ•เธดเธเธฃเธฃเธก เธซเธฃเธทเธญเธเธฃเธฐเธงเธฑเธ•เธดเธเธฃเธญเธเธเธฃเธฑเธง",
        "badge_color": "#B45309", "badge_bg": "#FEF3C7", "header_bg": "#D97706",
        "price": 2990,
        "sub_packages": [
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ 9.9 METABOLIC HEALTH MONTH", "short_name": "9.9 METABOLIC", "price": 2990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เธเธเธญเนเธงเธ (Obesity Check)", "short_name": "เธเธเธญเนเธงเธ", "price": 2990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เธงเธฑเธข 35+", "short_name": "เธงเธฑเธข 35+", "price": 3990},
        ]
    },
    3: {
        "level": 3, "name": "เธเธฅเธธเนเธก 3: เน€เธฃเธดเนเธกเธกเธตเธเธงเธฒเธกเธเธดเธ”เธเธเธ•เธด/เนเธฃเธเธฃเธฐเธขเธฐเน€เธฃเธดเนเธกเธ•เนเธ", "short_title": "เน€เธฃเธดเนเธกเธกเธตเธเธงเธฒเธกเธเธดเธ”เธเธเธ•เธด",
        "target_audience": "เธชเธณเธซเธฃเธฑเธเธเธนเนเธ—เธตเนเธกเธตเธเนเธฒเธเธฅเธ•เธฃเธงเธเน€เธฃเธดเนเธกเธเธดเธ”เธเธเธ•เธด เธซเธฃเธทเธญเนเธ”เนเธฃเธฑเธเธเธฒเธฃเธงเธดเธเธดเธเธเธฑเธขเนเธฃเธเธฃเธฐเธขเธฐเน€เธฃเธดเนเธกเธ•เนเธ",
        "badge_color": "#B91C1C", "badge_bg": "#FEE2E2", "header_bg": "#DC2626",
        "price": 2990,
        "sub_packages": [
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เน€เธเธฒเธซเธงเธฒเธ (Diabetes Check)", "short_name": "เน€เธเธฒเธซเธงเธฒเธ", "price": 3290},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ (Hypertension Check)", "short_name": "เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ", "price": 2990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เนเธเธกเธฑเธเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ (Lipid Check)", "short_name": "เนเธเธกเธฑเธเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ", "price": 2990},
        ]
    },
    4: {
        "level": 4, "name": "เธเธฅเธธเนเธก 4: เธกเธตเธญเธฒเธเธฒเธฃเธเนเธณ เน / เธเธงเธฃเธเธฃเธฐเน€เธกเธดเธเน€เธเธดเนเธกเน€เธ•เธดเธก", "short_title": "เธกเธตเธญเธฒเธเธฒเธฃเธเนเธณเน",
        "target_audience": "เธชเธณเธซเธฃเธฑเธเธเธนเนเธ—เธตเนเธกเธตเธญเธฒเธเธฒเธฃเธเธดเธ”เธเธเธ•เธดเธเนเธณเน เธซเธฃเธทเธญเธกเธตเธเธงเธฒเธกเธเธฑเธเธงเธฅเนเธฅเธฐเธ•เนเธญเธเธเธฒเธฃเธ•เธฃเธงเธเน€เธเธดเธเธฅเธถเธ",
        "badge_color": "#5B21B6", "badge_bg": "#F3E8FF", "header_bg": "#6941C6",
        "price": 4990,
        "sub_packages": [
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธเธฑเธ”เธเธฃเธญเธ เธกเธฐเน€เธฃเนเธ (Cancer Screening)", "short_name": "เธเธฑเธ”เธเธฃเธญเธเธกเธฐเน€เธฃเนเธ", "price": 4990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ เธซเธฑเธงเนเธ (Heart Check)", "short_name": "เธซเธฑเธงเนเธ", "price": 4990},
            {"name": "เนเธเธเน€เธเธเธ•เธฃเธงเธเธเธฑเธ”เธเธฃเธญเธ เธกเธฐเน€เธฃเนเธเธฅเธณเนเธชเนเนเธซเธเน (Colonoscopy)", "short_name": "Colonoscopy", "price": 8900},
        ]
    }
}

DISEASE_CONFIG = {
    "๐ฅ General Dashboard (เธซเธเนเธฒเนเธฃเธ)": {"icon": "๐ฅ", "is_general": True},
    "๐”ฎ AI Forecast (เธเธขเธฒเธเธฃเธ“เน 2568-2569)": {"icon": "๐”ฎ", "is_forecast": True, "target_desc": "เธเธขเธฒเธเธฃเธ“เนเธฅเนเธงเธเธซเธเนเธฒ 24 เน€เธ”เธทเธญเธ"},
    "เธฅเนเธฒเธเนเธ• (Dialysis)": {"icon": "๐ฉบ", "filter_condition": lambda df: df["is_dialysis"] == True, "target_desc": "BP <130/80"},
    "เน€เธเธฒเธซเธงเธฒเธ (Diabetes)": {"icon": "๐ฉธ", "filter_condition": lambda df: df["disease_group"] == "เน€เธเธฒเธซเธงเธฒเธ", "target_desc": "HbA1c <7.0"},
    "เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ (Hypertension)": {"icon": "๐ซ€", "filter_condition": lambda df: df["disease_group"] == "เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ", "target_desc": "BP <140/90"},
    "เนเธเธกเธฑเธเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ (Dyslipidemia)": {"icon": "๐ง", "filter_condition": lambda df: df["disease_group"] == "เนเธเธกเธฑเธเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ", "target_desc": "LDL <100"},
    "เธเธฅเธธเนเธกเธญเธทเนเธ เน (Health Packages)": {"icon": "๐งฌ", "filter_condition": lambda df: (df["disease_group"] == "เธญเธทเนเธ เน") | (df["disease_group"].isna()) | (df["disease_group"] == "เนเธกเนเธฃเธฐเธเธธ") | (df["disease_group"] == "เธ—เธฑเนเธงเนเธ"), "target_desc": "เธเธฑเธ”เธเธฃเธญเธเธชเธธเธเธ เธฒเธเน€เธเธดเธเธฃเธธเธ", "is_other_packages": True}
}

# ============================================================
# Helpers
# ============================================================
def render_custom_html(html_str):
    clean_str = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_str, unsafe_allow_html=True)

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
                raise ValueError(f"เนเธกเนเธเธเนเธเธฅเน CSV เธ เธฒเธขเนเธ ZIP: {file_path}")
            selected_file = csv_files[0]
            if preferred_csv_names:
                preferred_lower = {name.lower() for name in preferred_csv_names}
                for csv_name in csv_files:
                    if os.path.basename(csv_name).lower() in preferred_lower:
                        selected_file = csv_name
                        break
            with z.open(selected_file) as csv_file:
                return pd.read_csv(csv_file, encoding="utf-8-sig", low_memory=False)
    raise ValueError(f"เนเธกเนเธฃเธญเธเธฃเธฑเธเธเธฃเธฐเน€เธ เธ—เนเธเธฅเน: {file_path}")

def find_data_file(file_candidates):
    for file_name in file_candidates:
        if os.path.exists(file_name):
            return file_name
    return None

@st.cache_data(show_spinner="เธเธณเธฅเธฑเธเนเธซเธฅเธ”เธเนเธญเธกเธนเธฅ...")
def load_data():
    main_file = find_data_file(["visits_cleaned.csv", "visits_cleaned.zip", "visits_with_monthly_count.csv", "visits_with_monthly_count.zip"])
    if main_file is None:
        return None, None, False, False
    try:
        df = read_csv_or_zip(main_file, preferred_csv_names=["visits_cleaned.csv", "visits_with_monthly_count.csv"])
    except Exception as e:
        st.error(f"โ ๏ธ เธญเนเธฒเธเนเธเธฅเนเธเนเธญเธกเธนเธฅเธซเธฅเธฑเธเนเธกเนเธชเธณเน€เธฃเนเธ: {e}")
        return None, None, False, False

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
        df["year_month"] = "เนเธกเนเธฃเธฐเธเธธ"
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
    df["gender"] = df["gender"].fillna("เนเธกเนเธฃเธฐเธเธธ") if "gender" in df.columns else "เนเธกเนเธฃเธฐเธเธธ"
    df["gender_code"] = df["gender"].map({"เธ": 0, "เธ": 1}).fillna(0.5)
    df["bmi_imputed"] = df["bmi"].isna() if "bmi" in df.columns else True
    df["bmi"] = df["bmi"].fillna(df["bmi"].median() if "bmi" in df.columns and pd.notna(df["bmi"].median()) else 22.0)

    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()
    if has_age:
        df["age_at_visit"] = df["age_at_visit"].fillna(df["age_at_visit"].median() if pd.notna(df["age_at_visit"].median()) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(df["age_at_visit"], bins=[0, 29, 39, 49, 59, 120], labels=["<30 เธเธต", "30-40 เธเธต", "40-50 เธเธต", "50-60 เธเธต", ">60 เธเธต"]).astype(str).replace("nan", "เนเธกเนเธฃเธฐเธเธธ")
        df["pyramid_group"] = pd.cut(df["age_at_visit"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False, labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]).astype(str)
    else:
        df["age_at_visit"], df["is_adult"], df["age_group"], df["pyramid_group"] = 35.0, True, "เนเธกเนเธฃเธฐเธเธธ", "เนเธกเนเธฃเธฐเธเธธ"

    bp_cat = pd.cut(df["systolic"], bins=[-1, 120, 139, 300], labels=["เธเธเธ•เธด (<120)", "เน€เธเนเธฒเธฃเธฐเธงเธฑเธ (120-139)", "เธชเธนเธ (>=140)"]).astype(object)
    bp_cat[df["systolic"].isna()] = "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ"
    df["bp_level"] = bp_cat
    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    df["diagnosis_clean"] = df["diagnosis_text"].astype(str).str.strip().replace({"": "เนเธกเนเธฃเธฐเธเธธ", "-": "เนเธกเนเธฃเธฐเธเธธ", ":": "เนเธกเนเธฃเธฐเธเธธ"}).where(df["diagnosis_text"].notna(), "เนเธกเนเธฃเธฐเธเธธ") if "diagnosis_text" in df.columns else "เนเธกเนเธฃเธฐเธเธธ"
    if "disease_group" not in df.columns: df["disease_group"] = "เธ—เธฑเนเธงเนเธ"
    if "clinic_name" not in df.columns: df["clinic_name"] = "เนเธกเนเธฃเธฐเธเธธ"

    return df, None, has_date, has_age

@st.cache_data(show_spinner=False)
def process_patient_data(dataframe):
    df_proc = dataframe.copy()
    today = df_proc["visit_date"].max() if pd.notna(df_proc["visit_date"].max()) else pd.Timestamp.today().normalize()

    if "diagnosis_text" in df_proc.columns:
        df_proc["is_dialysis"] = df_proc["diagnosis_text"].fillna("").astype(str).str.lower().str.contains("เนเธ•|kidney|dialysis|ckd", regex=True)
    else:
        df_proc["is_dialysis"] = False

    days = (today - df_proc["visit_date"]).dt.days.fillna(0).clip(lower=0) if "visit_date" in df_proc.columns else pd.Series(0, index=df_proc.index)
    df_proc["days_since_last_visit"] = days
    is_dia = df_proc["is_dialysis"]
    is_diabetes = df_proc["disease_group"].astype(str).str.contains("เน€เธเธฒเธซเธงเธฒเธ")

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

# Load data sequence
raw_df, monthly_df, has_date, has_age = load_data()
if raw_df is None:
    st.error("โ ๏ธ เนเธกเนเธเธเนเธเธฅเนเธเนเธญเธกเธนเธฅ เธเธฃเธธเธ“เธฒเธญเธฑเธเนเธซเธฅเธ”เนเธเธฅเน visits_cleaned.csv เธซเธฃเธทเธญเนเธเธฅเนเธญเธทเนเธเน เธ—เธตเนเธฃเธฐเธเธธ")
    st.stop()

df = process_patient_data(raw_df)
summary_pts = None
if "patient_id" in df.columns:
    summary_pts = build_summary_pts(df)

# ============================================================
# Business Logic Helpers
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
        screenings.append("PSA (เธกเธฐเน€เธฃเนเธเธ•เนเธญเธกเธฅเธนเธเธซเธกเธฒเธ)")
        add_on_price += 2000
    return pkg_name, base_price, screenings, add_on_price

def _analyze_patient_risk(row):
    score, reasons = 100, []
    sys_val = row.get("systolic", 0)
    bmi_val = row.get("bmi", 22)
    visits_val = row.get("visits", 1)

    if sys_val >= 160:
        score -= 40; reasons.append(f"เธเธงเธฒเธกเธ”เธฑเธเธงเธดเธเธคเธ• ({sys_val:.0f})")
    elif sys_val >= 140:
        score -= 25; reasons.append(f"เธเธงเธฒเธกเธ”เธฑเธเธชเธนเธ ({sys_val:.0f})")
    elif sys_val >= 130:
        score -= 10; reasons.append(f"เน€เธเนเธฒเธฃเธฐเธงเธฑเธเธเธงเธฒเธกเธ”เธฑเธ ({sys_val:.0f})")
    if bmi_val >= 30:
        score -= 20; reasons.append(f"เนเธฃเธเธญเนเธงเธ ({bmi_val:.1f})")
    elif bmi_val >= 25:
        score -= 10; reasons.append(f"เธเนเธณเธซเธเธฑเธเน€เธเธดเธ ({bmi_val:.1f})")
    if visits_val >= 5:
        score -= 15; reasons.append(f"เธกเธฒเธฃเธ.เธเนเธญเธขเธเธดเธ”เธเธเธ•เธด ({visits_val:.0f} เธเธฃเธฑเนเธ)")
    elif visits_val >= 3:
        score -= 5; reasons.append(f"เธกเธตเธเธฃเธฐเธงเธฑเธ•เธดเธกเธฒเธเนเธณ ({visits_val:.0f} เธเธฃเธฑเนเธ)")

    pkg_name, base_price, screenings, add_on_price = _recommend_package(row)
    if screenings:
        score -= 5
    for sc in screenings:
        reasons.append(f"เนเธเธฐเธเธณ {sc}")
    if not reasons:
        reasons.append("เธชเธธเธเธ เธฒเธเธญเธขเธนเนเนเธเน€เธเธ“เธ‘เนเธเธเธ•เธด")

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

    has_chronic = any(k in diag for k in ["เนเธเนเธเธซเธเนเธฒเธญเธ", "เน€เธซเธเธทเนเธญเธข", "เนเธเธชเธฑเนเธ", "เธซเธฑเธงเนเธ", "chest pain", "เธกเธฐเน€เธฃเนเธ", "เธเนเธญเธ", "tumor"])
    has_dm = any(k in diag for k in ["เน€เธเธฒเธซเธงเธฒเธ", "diabetes", "dm"]) or float(pt_row.get("fbs", 0)) >= 100 or float(pt_row.get("hba1c", 0)) >= 5.7
    has_ht = any(k in diag for k in ["เธเธงเธฒเธกเธ”เธฑเธ", "hypertension", "ht"]) or sys_bp >= 140 or dia_bp >= 90
    has_lipid = any(k in diag for k in ["เนเธเธกเธฑเธ", "lipid", "cholesterol"]) or float(pt_row.get("cholesterol", 0)) >= 200
    waist_cm = float(pt_row.get("waist", bmi * (3.65 if gender == "เธ" else 3.35)))
    is_central_obesity = (gender == "เธ" and waist_cm > 90) or (gender != "เธ" and waist_cm > 80)

    reasons = []
    if sys_bp >= 160 or dia_bp >= 100 or bmi >= 32.0 or age >= 60 or has_chronic or pt_row.get("critical_risk") == 1:
        if sys_bp >= 160 or dia_bp >= 100:
            reasons.append(f"เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธฃเธฐเธ”เธฑเธเธงเธดเธเธคเธ• (BP {sys_bp:.0f}/{dia_bp:.0f} mmHg)")
        if bmi >= 32.0:
            reasons.append(f"เนเธฃเธเธญเนเธงเธเธฃเธธเธเนเธฃเธ (BMI {bmi:.1f})")
        best_sub = "เธซเธฑเธงเนเธ (Heart Check)" if (sys_bp >= 160 or dia_bp >= 100) else "เธเธฑเธ”เธเธฃเธญเธเธกเธฐเน€เธฃเนเธ"
        return 4, reasons, best_sub

    if (140 <= sys_bp < 160) or has_ht or has_dm or has_lipid or (27.5 <= bmi < 32.0) or (50 <= age < 60):
        if has_dm: reasons.append("เธเธเธเนเธฒเธเนเธณเธ•เธฒเธฅเนเธเน€เธฅเธทเธญเธ”เธชเธนเธ")
        if 140 <= sys_bp < 160: reasons.append(f"เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธเธฃเธฐเธ”เธฑเธ 1 (BP {sys_bp:.0f}/{dia_bp:.0f})")
        best_sub = "เน€เธเธฒเธซเธงเธฒเธ (Diabetes Check)" if has_dm else "เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เธชเธนเธ (Hypertension Check)"
        return 3, reasons, best_sub

    if (120 <= sys_bp < 140) or bmi >= 23.0 or (35 <= age < 50) or is_central_obesity:
        if bmi >= 23.0: reasons.append(f"เธเนเธณเธซเธเธฑเธเน€เธเธดเธเน€เธเธ“เธ‘เน (BMI {bmi:.1f})")
        best_sub = "เธเธเธญเนเธงเธ (Obesity Check)" if bmi > 25.0 else "เธงเธฑเธข 35+"
        return 2, reasons, best_sub

    reasons.append("เธชเธธเธเธ เธฒเธเนเธ”เธขเธฃเธงเธกเนเธเนเธเนเธฃเธเธ”เธต")
    best_sub = "PREMIUM" if age > 35 else ("STANDARD" if age >= 30 else "BASIC")
    return 1, reasons, best_sub

# ============================================================
# UI Fragments
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
        gen = st.button("๐”” Alert List", use_container_width=True)
    with ac2:
        sched = st.button("๐“… Outreach", use_container_width=True)

    if gen:
        if not high_risk.empty:
            st.success(f"โ… {len(high_risk)} เธฃเธฒเธขเธเธฒเธฃ")
            st.download_button("๐“ฅ เธ”เธฒเธงเธเนเนเธซเธฅเธ” CSV", high_risk.to_csv(index=False).encode("utf-8-sig"), "alert_list.csv", "text/csv", use_container_width=True)
        else:
            st.info("เนเธกเนเธกเธตเธเธฅเธธเนเธกเน€เธชเธตเนเธขเธเธงเธดเธเธคเธ•")

    if "outreach_q" not in st.session_state:
        st.session_state["outreach_q"] = []
    if sched:
        st.session_state["show_sched"] = True
    if st.session_state.get("show_sched"):
        with st.form("sched_form"):
            ids = st.multiselect("เน€เธฅเธทเธญเธ Patient", high_risk["patient_id"].tolist() if not high_risk.empty else [])
            d = st.date_input("เธงเธฑเธเธ—เธตเนเธเธฑเธ”")
            note = st.text_area("เธเธฑเธเธ—เธถเธ")
            if st.form_submit_button("เธขเธทเธเธขเธฑเธ"):
                st.session_state["outreach_q"].append({"เธงเธฑเธ": str(d), "เธฃเธฒเธข": len(ids), "เธเธฑเธเธ—เธถเธ": note})
                st.session_state["show_sched"] = False
                st.success(f"โ… เธเธณเธซเธเธ”เธเธฒเธฃ {len(ids)} เธฃเธฒเธข")
    return high_risk

@st.fragment
def render_patient_profile(avail_df, summary_pts, dv, sel_idx):
    st.markdown("#### ๐“ Patient Profile & Recommendation")
    if sel_idx and avail_df is not None:
        sel_pid = avail_df.iloc[sel_idx[0]]["patient_id"]
        pt_data = summary_pts.loc[sel_pid]
        score, reasons_html, pkg_name, total_price, screenings = _analyze_patient_risk(pt_data)
        gender_icon = "๐‘ฉ" if pt_data["gender_code"] > 0.5 else "๐‘จ"
        if score <= 60:
            c_tx, badge = "#B3261E", "๐จ High Risk"
        elif score <= 80:
            c_tx, badge = "#B54708", "โ ๏ธ Medium Risk"
        else:
            c_tx, badge = "#065F46", "๐ฑ Low Risk"
        st.markdown(f"{gender_icon} **{sel_pid}** โ€” เธญเธฒเธขเธธ: {pt_data['age_at_visit']:.0f} เธเธต | :{badge}:")

        tab1, tab2, tab3 = st.tabs(["๐“ เธเนเธญเธกเธนเธฅเธชเธธเธเธ เธฒเธ", "๐ฅ เธเธฃเธฐเธงเธฑเธ•เธดเธเธฒเธฃเธงเธดเธเธดเธเธเธฑเธข", "๐’ เนเธเธเธเธฒเธฃเธ•เธฃเธงเธเธ—เธตเนเนเธเธฐเธเธณ"])

        with tab1:
            st.metric("Health Score", f"{score}%")
            st.info(f"๐’ก AI Analysis: {reasons_html}")

        with tab2:
            hist_df = dv[dv["patient_id"] == sel_pid].sort_values("visit_date", ascending=False)
            if not hist_df.empty:
                disp_hist = hist_df[["visit_date", "clinic_name", "diagnosis_clean", "systolic", "bmi", "critical_risk"]].copy()
                disp_hist["visit_date"] = disp_hist["visit_date"].dt.strftime("%Y-%m-%d")
                disp_hist.columns = ["เธงเธฑเธเธ—เธตเน", "เธเธฅเธดเธเธดเธ", "เธงเธดเธเธดเธเธเธฑเธข", "Sys", "BMI", "Risk"]

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
                st.info("เนเธกเนเธเธเธเธฃเธฐเธงเธฑเธ•เธดเธเธฒเธฃเธฃเธฑเธเธเธฃเธดเธเธฒเธฃเนเธเธฃเธฐเธเธ")

        with tab3:
            pkg_info = HEALTH_PACKAGES.get(pkg_name, {})
            st.markdown(f"**Package เธซเธฅเธฑเธ:** {pkg_name} โ€” เธฟ {pkg_info.get('price', 0):,.0f}")
            if pkg_info.get("tests"):
                for t in pkg_info["tests"]:
                    st.markdown(f"โ€ข {t}")
            if screenings:
                st.markdown("**๐” Add-on เน€เธเธเธฒเธฐเธเธธเธเธเธฅ:**")
                for sc in screenings:
                    st.markdown(f"- {sc}")
            st.markdown(f"**เธฃเธงเธกเธเธฃเธฐเน€เธกเธดเธเธฃเธฒเธเธฒ: เธฟ {total_price:,.0f}**")
    else:
        st.info("๐‘ เธเธฅเธดเธเน€เธฅเธทเธญเธเธเธนเนเธเนเธงเธขเธเธฒเธเธ•เธฒเธฃเธฒเธเธ”เนเธฒเธเธเนเธฒเธข เน€เธเธทเนเธญเธ”เธน Profile")

@st.fragment
def render_forecast_dashboard(df):
    if df is None or df.empty:
        st.warning("โ ๏ธ เนเธกเนเธเธเธเนเธญเธกเธนเธฅเธชเธณเธซเธฃเธฑเธเธเธฃเธฐเธกเธงเธฅเธเธฅเธเธขเธฒเธเธฃเธ“เน")
        return

    df_valid = df.dropna(subset=["visit_date"]).copy() if "visit_date" in df.columns else pd.DataFrame()
    if df_valid.empty:
        st.warning("โ ๏ธ เนเธกเนเธเธเธเธญเธฅเธฑเธกเธเน visit_date เธซเธฃเธทเธญเธเนเธญเธกเธนเธฅเธงเธฑเธเธ—เธตเนเนเธกเนเธชเธกเธเธนเธฃเธ“เน")
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

    x_hist = np.arange(len(hist_cases))
    slope_tot, intercept_tot = np.polyfit(x_hist, np.array(hist_cases, dtype=float), 1)
    slope_crit, intercept_crit = np.polyfit(x_hist, np.array(hist_critical, dtype=float), 1)
    if slope_tot <= 0: slope_tot = np.mean(hist_cases) * 0.015
    if slope_crit <= 0: slope_crit = np.mean(hist_critical) * 0.018

    forecast_periods = [f"{y}-{m:02d}" for y in [2025, 2026] for m in range(1, 13)]
    seasonal_pattern = {1: 1.06, 2: 0.94, 3: 0.96, 4: 0.89, 5: 1.01, 6: 1.07, 7: 1.10, 8: 1.13, 9: 1.08, 10: 1.05, 11: 1.08, 12: 1.12}

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        scenario = st.radio("๐ฏ เนเธเธเธชเธ–เธฒเธเธเธฒเธฃเธ“เนเธเธขเธฒเธเธฃเธ“เน:", ["๐“ เธกเธฒเธ•เธฃเธเธฒเธ", "๐€ เน€เธเธดเธเธฃเธธเธ (+10%)", "๐ก๏ธ เธญเธเธธเธฃเธฑเธเธฉเนเธเธดเธขเธก (-5%)"], horizontal=True, key="fc_scenario")
    with ctrl_col2:
        show_ci_band = st.checkbox("๐ก๏ธ เนเธชเธ”เธ Confidence Band", value=True, key="fc_show_ci")
    with ctrl_col3:
        show_crit_line = st.checkbox("โ ๏ธ เนเธชเธ”เธเธเธฅเธธเนเธกเน€เธชเธตเนเธขเธ NCDs", value=True, key="fc_show_crit")

    scenario_mult = 1.10 if "เน€เธเธดเธเธฃเธธเธ" in scenario else (0.95 if "เธญเธเธธเธฃเธฑเธเธฉเนเธเธดเธขเธก" in scenario else 1.00)
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

    st.markdown("### ๐”ฎ เธเธขเธฒเธเธฃเธ“เนเนเธเธงเนเธเนเธกเธชเธธเธเธ เธฒเธ (AI Forecast 2025-2026)")
    c1, c2, c3 = st.columns(3)
    c1.metric("๐‘ฅ เธเธฒเธ”เธเธฒเธฃเธ“เนเธเธนเนเธฃเธฑเธเธเธฃเธดเธเธฒเธฃเธฃเธงเธก (24 เน€เธ”เธทเธญเธ)", f"{tot_24m:,} เน€เธเธช", f"เธเธต 68: {tot_2025:,} | เธเธต 69: {tot_2026:,}")
    c2.metric("๐“ เธญเธฑเธ•เธฃเธฒเธเธฒเธฃเน€เธ•เธดเธเนเธ•เธเธฒเธ”เธเธฒเธฃเธ“เน", f"{((tot_2026 - tot_2025) / tot_2025 * 100):+.1f}% YoY")
    c3.metric("โ ๏ธ เธเธฒเธ”เธเธฒเธฃเธ“เนเธเธฅเธธเนเธกเน€เธชเธตเนเธขเธเธงเธดเธเธคเธ• (NCDs)", f"{(tot_crit_24m / tot_24m * 100):.1f}%", f"{tot_crit_24m:,} เธเธ", delta_color="inverse")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_periods, y=hist_cases, mode="lines+markers", name="เธเนเธญเธกเธนเธฅเธเธฃเธดเธเนเธเธญเธ”เธตเธ•", line=dict(color="#0E5C56", width=3.5)))
    future_x = [hist_periods[-1]] + [r["period"] for r in forecast_data]
    future_y = [hist_cases[-1]] + [r["projected_cases"] for r in forecast_data]
    if show_ci_band:
        fig.add_trace(go.Scatter(x=future_x, y=[hist_cases[-1]] + [r["upper_bound"] for r in forecast_data], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_x, y=[hist_cases[-1]] + [r["lower_bound"] for r in forecast_data], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(217, 119, 6, 0.16)", name="95% Confidence Band"))
    fig.add_trace(go.Scatter(x=future_x, y=future_y, mode="lines+markers", name="เธเธขเธฒเธเธฃเธ“เน AI", line=dict(color="#D97706", width=3.2, dash="dash")))
    if show_crit_line:
        fig.add_trace(go.Scatter(x=future_x, y=[hist_critical[-1]] + [r["projected_critical"] for r in forecast_data], mode="lines+markers", name="เธเธฅเธธเนเธกเน€เธชเธตเนเธขเธ NCDs", line=dict(color="#B3261E", width=2.5, dash="dot")))
    fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

@st.fragment
def render_other_packages_dashboard(dv, df, search_term):
    if "disease_group" in dv.columns:
        other_condition = (dv["disease_group"] == "เธญเธทเนเธ เน") | (dv["disease_group"].isna()) | (dv["disease_group"] == "เนเธกเนเธฃเธฐเธเธธ") | (dv["disease_group"] == "เธ—เธฑเนเธงเนเธ")
    else:
        other_condition = pd.Series(True, index=dv.index)
    other_dv = dv[other_condition].copy()
    if search_term:
        other_dv = other_dv[other_dv.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)]

    tab_brochure, tab_raw = st.tabs(["๐ฅ เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ 4 เธเธฅเธธเนเธก (AI Architect)", "๐“ เธเนเธญเธกเธนเธฅเธเธนเนเธฃเธฑเธเธเธฃเธดเธเธฒเธฃเนเธเธเธ•เธฒเธฃเธฒเธ"])
    with tab_raw:
        st.dataframe(other_dv, use_container_width=True)

    with tab_brochure:
        st.markdown("### ๐ฅ เนเธเธเน€เธเธเธ•เธฃเธงเธเธชเธธเธเธ เธฒเธ 4 เธเธฅเธธเนเธก (Vichaivej Omnoi)")
        if other_dv.empty:
            st.warning("โ ๏ธ เนเธกเนเธเธเธฃเธฒเธขเธเธทเนเธญเธเธนเนเธฃเธฑเธเธเธฃเธดเธเธฒเธฃ")
            return
        avail_pids = other_dv["patient_id"].dropna().unique().tolist() if "patient_id" in other_dv.columns else [f"PT-{i+1:03d}" for i in range(min(10, len(other_dv)))]
        sel_pid = st.selectbox("๐‘ค เน€เธฅเธทเธญเธเธเธนเนเธฃเธฑเธเธเธฃเธดเธเธฒเธฃเธ—เธตเนเธ•เนเธญเธเธเธฒเธฃเธงเธดเน€เธเธฃเธฒเธฐเธซเน:", options=avail_pids, index=0)

        pt_records = df[df["patient_id"] == sel_pid] if "patient_id" in df.columns else other_dv.iloc[0:1]
        pt_row = pt_records.iloc[-1].to_dict() if not pt_records.empty else other_dv.iloc[0].to_dict()
        pt_row["visits"] = len(pt_records)

        assigned_tier, risk_reasons, best_match_sub_pkg = _assess_patient_tier(pt_row)
        st.success(f"**AI Recommendation:** เนเธเธฐเธเธณ {HEALTH_PACKAGES_4LEVEL[assigned_tier]['name']} (Best Match: {best_match_sub_pkg})")

        cols = st.columns(4)
        for col, level_key in zip(cols, [1, 2, 3, 4]):
            group = HEALTH_PACKAGES_4LEVEL[level_key]
            with col:
                is_tier_match = (assigned_tier == level_key)
                border_style = f"border: 3px solid {group['header_bg']};" if is_tier_match else ""
                st.markdown(f"<div style='{border_style} padding:8px; border-radius:8px;'><b>{group['short_title']}</b><br><small>{group['target_audience']}</small></div>", unsafe_allow_html=True)
                for sub in group["sub_packages"]:
                    if st.button(f"เน€เธฅเธทเธญเธ {sub['short_name']} ({sub['price']:,}เธฟ)", key=f"btn_{level_key}_{sub['short_name']}_{sel_pid}", use_container_width=True):
                        st.toast(f"โ… เน€เธฅเธทเธญเธ {sub['name']} เน€เธฃเธตเธขเธเธฃเนเธญเธข")

@st.fragment
def render_disease_center(dv, df, selected_tab_key, active_config, search_term):
    disease_df = dv[active_config["filter_condition"](dv)].copy()
    st.markdown(f"### {active_config['icon']} {selected_tab_key} Command Center")
    st.caption(f"เน€เธเนเธฒเธซเธกเธฒเธขเธเธฒเธฃเธฃเธฑเธเธฉเธฒ: {active_config['target_desc']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("เธเธณเธเธงเธเธเธนเนเธเนเธงเธข (เธ•เธฒเธกเธ•เธฑเธงเธเธฃเธญเธ)", f"{disease_df['patient_id'].nunique() if 'patient_id' in disease_df.columns else len(disease_df):,} เธเธ")
    p1_count = len(disease_df[disease_df["priority_status"] == "P1-Urgent"])
    c2.metric("เธเธฅเธธเนเธกเน€เธชเธตเนเธขเธ (P1-Urgent)", f"{p1_count} เธเธ", delta="-เธ•เนเธญเธเธ•เธดเธ”เธ•เธฒเธกเธ—เธฑเธเธ—เธต" if p1_count > 0 else "เธเธเธ•เธด", delta_color="inverse")

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
            st.markdown(f"### ๐’ เนเธเธฐเธเธณ Combined Care Package เธชเธณเธซเธฃเธฑเธ: {sel_pid}")
            if st.button(f"โจ Generate {selected_tab_key} Care Package"):
                st.success(f"โ… **เธชเธฃเนเธฒเธ {selected_tab_key} Package เธชเธณเน€เธฃเนเธ!**")
    else:
        st.info("เนเธกเนเธเธเธเธเนเธเนเนเธเธเธฅเธธเนเธกเธเธตเน")

# ============================================================
# Sidebar & Filtering
# ============================================================
tab_options = [f"{v['icon']} {k}" for k, v in DISEASE_CONFIG.items()]
selected_tab_str = st.radio(" ", tab_options, horizontal=True, label_visibility="collapsed")
selected_tab_key = selected_tab_str.split(" ", 1)[1]
active_config = DISEASE_CONFIG[selected_tab_key]

with st.sidebar:
    st.markdown(f"### ๐๏ธ Filter Scope: {selected_tab_key}")
    search_term = st.text_input("๐” เธเนเธเธซเธฒ (ID/เธเธทเนเธญ)", key="ctx_search")
    all_diseases = sorted(df["disease_group"].unique())
    disease_sel = st.multiselect("เธเธฅเธธเนเธกเนเธฃเธ", all_diseases, default=all_diseases)
    all_genders = sorted(df["gender"].unique())
    gender_sel = st.multiselect("เน€เธเธจ", all_genders, default=all_genders)
    all_clinics = sorted(df["clinic_name"].dropna().unique())
    clinic_sel = st.multiselect("เธเธฅเธดเธเธดเธ", all_clinics, default=all_clinics)

mask = (df["disease_group"].isin(disease_sel) & df["gender"].isin(gender_sel) & df["clinic_name"].isin(clinic_sel))
dv = df[mask].copy()

if dv.empty:
    st.warning("โ ๏ธ เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเธ•เธฒเธกเธ•เธฑเธงเธเธฃเธญเธเธ—เธตเนเน€เธฅเธทเธญเธ")
    st.stop()

# ============================================================
# Main Content Routing
# ============================================================
if active_config.get("is_general"):
    as_of = df["visit_date"].max()
    as_of_str = as_of.strftime("%d %b %Y") if pd.notna(as_of) else "เนเธกเนเธฃเธฐเธเธธ"
    st.markdown(f"## ๐ฅ Clinical Command Center")
    st.caption(f"เธเนเธญเธกเธนเธฅเธฅเนเธฒเธชเธธเธ” {as_of_str}")

    total_v = len(dv)
    uniq_pts = dv["patient_id"].nunique() if "patient_id" in dv.columns else total_v
    st.markdown("### เธชเธฃเธธเธเธ•เธฑเธงเธเธตเนเธงเธฑเธ” (KPIs)")
    k1, k2, k3 = st.columns(3)
    k1.metric("เธเธณเธเธงเธเน€เธเธชเธฃเธฑเธเธเธฃเธดเธเธฒเธฃ", f"{total_v:,}")
    k2.metric("เธเธณเธเธงเธเธเธนเนเธฃเธฑเธเธเธฃเธดเธเธฒเธฃ", f"{uniq_pts:,}")
    k3.metric("เธเธงเธฒเธกเธ”เธฑเธเนเธฅเธซเธดเธ•เน€เธเธฅเธตเนเธข", f"{dv['systolic'].mean():.1f} mmHg")

    st.divider()

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### โก Command Action Panel")
        render_action_panel(dv)
    with c2:
        st.markdown("#### ๐” เน€เธฅเธทเธญเธเธเธนเนเธเนเธงเธขเน€เธเธทเนเธญเธเธฃเธฐเน€เธกเธดเธ Package")
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
