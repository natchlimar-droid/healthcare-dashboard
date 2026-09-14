"""
Data Ingestion, Cleaning, Imputation, Caching, and Preprocessing Pipeline
"""

import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st


def read_csv_or_zip(file_path: str, preferred_csv_names=None) -> pd.DataFrame:
    """
    Reads a CSV file directly or extracts a CSV from inside a ZIP archive.
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


def find_data_file(file_candidates: list):
    """
    Returns the first file that exists from a list of candidate filenames.
    """
    for file_name in file_candidates:
        if os.path.exists(file_name):
            return file_name
    return None


@st.cache_data(show_spinner="กำลังโหลดและประมวลผลข้อมูลสุขภาพ...")
def load_data():
    """
    Primary data loader with auto-detection of cleaned datasets (CSV/ZIP).
    Returns (df, monthly_df, has_date, has_age).
    """
    main_file = find_data_file([
        "visits_cleaned.csv",
        "visits_cleaned.zip",
        "visits_with_monthly_count.csv",
        "visits_with_monthly_count.zip",
        "visits_with_patient_id.csv",
        "visits_with_patient_id.zip"
    ])

    if main_file is None:
        return None, None, False, False

    try:
        df = read_csv_or_zip(
            main_file,
            preferred_csv_names=[
                "visits_cleaned.csv",
                "visits_with_monthly_count.csv",
                "visits_with_patient_id.csv"
            ]
        )
    except Exception as e:
        st.error(f"⚠️ อ่านไฟล์ข้อมูลหลักไม่สำเร็จ ({main_file}): {e}")
        return None, None, False, False

    # Clean whitespace in column names
    df.columns = df.columns.astype(str).str.strip()

    # 1. Parse Dates
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

    # 2. Convert Numeric Columns
    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. BP Processing (Fallback from bp_raw if systolic/diastolic missing)
    if "systolic_bp" not in df.columns:
        df["systolic_bp"] = np.nan
    if "diastolic_bp" not in df.columns:
        df["diastolic_bp"] = np.nan

    if "bp_raw" in df.columns and df["systolic_bp"].isna().all():
        bp_split = df["bp_raw"].astype(str).str.replace(",", "", regex=False).str.extract(
            r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$"
        )
        sys_bp = pd.to_numeric(bp_split[0], errors="coerce")
        dia_bp = pd.to_numeric(bp_split[1], errors="coerce")
        sys_bp[~sys_bp.between(60, 250)] = np.nan
        dia_bp[~dia_bp.between(30, 150)] = np.nan
        df["systolic_bp"] = sys_bp
        df["diastolic_bp"] = dia_bp

    df.rename(columns={"systolic_bp": "systolic", "diastolic_bp": "diastolic"}, inplace=True)

    # 4. Gender Normalization
    df["gender"] = df["gender"].fillna("ไม่ระบุ") if "gender" in df.columns else "ไม่ระบุ"
    df["gender_code"] = df["gender"].map({"ช": 0, "ญ": 1}).fillna(0.5)

    # 5. BMI Imputation & Flag
    if "bmi" in df.columns:
        df["bmi_imputed"] = df["bmi"].isna()
        med_bmi = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(med_bmi if pd.notna(med_bmi) else 22.0)
    else:
        df["bmi_imputed"] = True
        df["bmi"] = 22.0

    # 6. Age Categorization
    has_age = "age_at_visit" in df.columns and df["age_at_visit"].notna().any()
    if has_age:
        med_age = df["age_at_visit"].median()
        df["age_at_visit"] = df["age_at_visit"].fillna(med_age if pd.notna(med_age) else 35.0)
        df["is_adult"] = df["age_at_visit"] >= 18
        df["age_group"] = pd.cut(
            df["age_at_visit"],
            bins=[0, 29, 39, 49, 59, 120],
            labels=["<30 ปี", "30-40 ปี", "40-50 ปี", "50-60 ปี", ">60 ปี"]
        ).astype(str).replace("nan", "ไม่ระบุ")

        df["pyramid_group"] = pd.cut(
            df["age_at_visit"],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
            right=False,
            labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        ).astype(str)
    else:
        df["age_at_visit"] = 35.0
        df["is_adult"] = True
        df["age_group"] = "ไม่ระบุ"
        df["pyramid_group"] = "ไม่ระบุ"

    # 7. Blood Pressure & Critical Risk Categories
    bp_cat = pd.cut(
        df["systolic"],
        bins=[-1, 120, 139, 300],
        labels=["ปกติ (<120)", "เฝ้าระวัง (120-139)", "สูง (≥140)"]
    ).astype(object)
    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat

    df["critical_risk"] = (df["is_adult"] & (df["bmi"] >= 25) & (df["systolic"] >= 140)).astype(int)

    # 8. Diagnosis & Clinic Defaults
    if "diagnosis_text" in df.columns:
        df["diagnosis_clean"] = (
            df["diagnosis_text"].astype(str).str.strip()
            .replace({"": "ไม่ระบุ", "-": "ไม่ระบุ", ":": "ไม่ระบุ"})
            .where(df["diagnosis_text"].notna(), "ไม่ระบุ")
        )
    else:
        df["diagnosis_clean"] = "ไม่ระบุ"

    if "disease_group" not in df.columns:
        df["disease_group"] = "ทั่วไป"
    if "clinic_name" not in df.columns:
        df["clinic_name"] = "ไม่ระบุ"

    return df, None, has_date, has_age


@st.cache_data(show_spinner=False)
def process_patient_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Computes days since last visit, identifies dialysis indications,
    and assigns clinical triage priority status (P1-Urgent, P2-Warning, P3-Normal).
    """
    df_proc = dataframe.copy()
    today = df_proc["visit_date"].max() if pd.notna(df_proc["visit_date"].max()) else pd.Timestamp.today().normalize()

    # Dialysis detection
    if "diagnosis_text" in df_proc.columns:
        df_proc["is_dialysis"] = (
            df_proc["diagnosis_text"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains("ไต|kidney|dialysis|ckd", regex=True)
        )
    else:
        df_proc["is_dialysis"] = False

    # Days since last visit
    if "visit_date" in df_proc.columns:
        days = (today - df_proc["visit_date"]).dt.days.fillna(0).clip(lower=0)
    else:
        days = pd.Series(0, index=df_proc.index)
    df_proc["days_since_last_visit"] = days

    is_dia = df_proc["is_dialysis"]
    is_diabetes = df_proc["disease_group"].astype(str).str.contains("เบาหวาน")

    conditions = [
        is_dia & (days > 7),
        is_dia & (days > 3),
        is_dia,
        (~is_dia) & is_diabetes & (days > 90),
        (~is_dia) & is_diabetes & (days > 30),
        (~is_dia) & is_diabetes,
        (~is_dia) & (~is_diabetes) & (days > 180)
    ]
    choices = [
        "P1-Urgent",
        "P2-Warning",
        "P3-Normal",
        "P1-Urgent",
        "P2-Warning",
        "P3-Normal",
        "P1-Urgent"
    ]
    df_proc["priority_status"] = np.select(conditions, choices, default="P3-Normal")
    return df_proc


@st.cache_data(show_spinner=False)
def build_summary_pts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates visit records into patient-level summary metrics for profiling.
    """
    agg_kwargs = {
        "bmi": ("bmi", "mean"),
        "systolic": ("systolic", "mean"),
        "gender_code": ("gender_code", "first"),
        "age_at_visit": ("age_at_visit", "max")
    }
    agg_kwargs["visits"] = (
        ("visit_id", "count") if "visit_id" in dataframe.columns else ("patient_id", "count")
    )
    sp = dataframe.groupby("patient_id").agg(**agg_kwargs).round(1)
    sp["systolic"] = sp["systolic"].fillna(dataframe["systolic"].median())
    sp["target"] = ((sp["visits"] >= 3) | (sp["systolic"] >= 135)).astype(int)
    return sp
