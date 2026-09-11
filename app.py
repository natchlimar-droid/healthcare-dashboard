# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    """
    รองรับข้อมูลหลักทั้ง:
    - visits_cleaned.csv
    - visits_cleaned.zip  (ภายใน ZIP ต้องมีไฟล์ CSV เพียง 1 ไฟล์)

    รองรับ monthly summary ทั้ง:
    - monthly_visit_summary.csv
    - monthly_visit_summary.zip
    """

    # --------------------------------------------------------
    # 1) Main visit data (required)
    # --------------------------------------------------------
    if os.path.exists("visits_cleaned.csv"):
        main_file = "visits_cleaned.csv"
    elif os.path.exists("visits_cleaned.zip"):
        main_file = "visits_cleaned.zip"
    else:
        return None, None, False, False

    try:
        df = pd.read_csv(
            main_file,
            encoding="utf-8-sig",
            low_memory=False,
            compression="infer"
        )
    except Exception as e:
        st.error(f"⚠️ อ่านไฟล์ข้อมูลหลักไม่สำเร็จ: {e}")
        return None, None, False, False

    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # 2) Date preparation
    # --------------------------------------------------------
    has_date = False

    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(
            df["visit_date"],
            errors="coerce"
        )

        valid_date = (
            df["visit_date"].notna()
            & (df["visit_date"].dt.year >= 2000)
        )

        has_date = valid_date.any()
        df.loc[~valid_date, "visit_date"] = pd.NaT
    else:
        df["visit_date"] = pd.NaT

    df["year_month"] = (
        df["visit_date"].dt.to_period("M").astype(str)
        if has_date else "ไม่ระบุ"
    )

    df["visit_day"] = (
        df["visit_date"].dt.normalize()
        if has_date else pd.NaT
    )

    # --------------------------------------------------------
    # 3) Numeric columns
    # --------------------------------------------------------
    for col in ["age_at_visit", "height_cm", "weight_kg", "bmi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # 4) Blood pressure preparation
    # --------------------------------------------------------
    for col in ["systolic_bp", "diastolic_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # หากยังไม่มี BP column ให้สร้างไว้ก่อน
    if "systolic_bp" not in df.columns:
        df["systolic_bp"] = np.nan

    if "diastolic_bp" not in df.columns:
        df["diastolic_bp"] = np.nan

    # ใช้ bp_raw เป็น fallback กรณี systolic_bp ไม่มีข้อมูลทั้งหมด
    if "bp_raw" in df.columns and df["systolic_bp"].isna().all():
        bp_clean = (
            df["bp_raw"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        bp_split = bp_clean.str.extract(
            r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$"
        )

        sys_bp = pd.to_numeric(bp_split[0], errors="coerce")
        dia_bp = pd.to_numeric(bp_split[1], errors="coerce")

        # ตัดค่าที่ไม่สมเหตุสมผลออก
        sys_bp[~sys_bp.between(60, 250)] = np.nan
        dia_bp[~dia_bp.between(30, 150)] = np.nan

        df["systolic_bp"] = sys_bp
        df["diastolic_bp"] = dia_bp

    df.rename(
        columns={
            "systolic_bp": "systolic",
            "diastolic_bp": "diastolic"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # 5) Gender
    # --------------------------------------------------------
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("ไม่ระบุ")
    else:
        df["gender"] = "ไม่ระบุ"

    df["gender_code"] = (
        df["gender"]
        .map({"ช": 0, "ญ": 1})
        .fillna(0.5)
    )

    # --------------------------------------------------------
    # 6) BMI
    # --------------------------------------------------------
    if "bmi" in df.columns:
        df["bmi_imputed"] = df["bmi"].isna()

        med_bmi = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(
            med_bmi if pd.notna(med_bmi) else 22.0
        )
    else:
        df["bmi_imputed"] = True
        df["bmi"] = 22.0

    # --------------------------------------------------------
    # 7) Age
    # --------------------------------------------------------
    has_age = (
        "age_at_visit" in df.columns
        and df["age_at_visit"].notna().any()
    )

    if has_age:
        med_age = df["age_at_visit"].median()

        df["age_at_visit"] = df["age_at_visit"].fillna(
            med_age if pd.notna(med_age) else 35.0
        )

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
            labels=[
                "0-9", "10-19", "20-29", "30-39", "40-49",
                "50-59", "60-69", "70-79", "80+"
            ]
        ).astype(str)

    else:
        df["age_at_visit"] = 35.0
        df["is_adult"] = True
        df["age_group"] = "ไม่ระบุ"
        df["pyramid_group"] = "ไม่ระบุ"

    # --------------------------------------------------------
    # 8) BP level / critical risk
    # --------------------------------------------------------
    bp_cat = pd.cut(
        df["systolic"],
        bins=[-1, 120, 139, 300],
        labels=[
            "ปกติ (<120)",
            "เฝ้าระวัง (120-139)",
            "สูง (≥140)"
        ]
    ).astype(object)

    bp_cat[df["systolic"].isna()] = "ไม่มีข้อมูล"
    df["bp_level"] = bp_cat

    df["critical_risk"] = (
        df["is_adult"]
        & (df["bmi"] >= 25)
        & (df["systolic"] >= 140)
    ).astype(int)

    # --------------------------------------------------------
    # 9) Diagnosis / disease / clinic
    # --------------------------------------------------------
    def clean_diagnosis(value):
        if pd.isna(value):
            return "ไม่ระบุ"

        value = str(value).strip()
        return "ไม่ระบุ" if value in (":", "", "-") else value

    if "diagnosis_text" in df.columns:
        df["diagnosis_clean"] = df["diagnosis_text"].apply(clean_diagnosis)
    else:
        df["diagnosis_text"] = ""
        df["diagnosis_clean"] = "ไม่ระบุ"

    if "disease_group" not in df.columns:
        df["disease_group"] = "ทั่วไป"

    if "clinic_name" not in df.columns:
        df["clinic_name"] = "ไม่ระบุ"

    # --------------------------------------------------------
    # 10) Monthly summary (optional)
    # --------------------------------------------------------
    monthly = None

    try:
        if os.path.exists("monthly_visit_summary.csv"):
            monthly_file = "monthly_visit_summary.csv"
        elif os.path.exists("monthly_visit_summary.zip"):
            monthly_file = "monthly_visit_summary.zip"
        else:
            monthly_file = None

        if monthly_file:
            monthly = pd.read_csv(
                monthly_file,
                encoding="utf-8-sig",
                compression="infer"
            )

            if "visit_count" in monthly.columns:
                monthly["visit_count"] = pd.to_numeric(
                    monthly["visit_count"],
                    errors="coerce"
                )

    except Exception:
        monthly = None

    return df, monthly, has_date, has_age


# โหลดข้อมูล
df, monthly_df, has_date, has_age = load_data()

if df is None:
    st.error(
        "⚠️ ไม่พบหรืออ่านไฟล์ไม่ได้ กรุณาตรวจสอบว่ามี "
        "`visits_cleaned.csv` หรือ `visits_cleaned.zip` "
        "อยู่ในโฟลเดอร์เดียวกับ `app.py`"
    )
    st.stop()
