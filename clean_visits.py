import sys
import pandas as pd
import numpy as np

# บังคับ stdout ใช้ UTF-8 เสมอ ป้องกัน UnicodeEncodeError บน Windows (cp874)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -----------------------------
# 1) อ่านไฟล์ที่จับคู่ patient_id แล้ว
# -----------------------------
try:
    df = pd.read_excel("visits_with_patient_id.xlsx")
except FileNotFoundError:
    raise SystemExit("[ERROR] ไม่พบไฟล์ visits_with_patient_id.xlsx -- กรุณารัน match_patient_id.py ก่อน")

print("คอลัมน์ที่พบ:")
print(df.columns.tolist())

# -----------------------------
# 2) เลือกเฉพาะคอลัมน์ที่ใช้ในโปรเจกต์
#    (ตัดขั้นตอน rename ออก เพราะชื่อคอลัมน์ตรงกับที่ต้องการอยู่แล้ว)
# -----------------------------
required_columns = [
    "record_no",
    "visit_date",
    "visit_time",
    "visit_id",
    "patient_id",
    "gender",
    "age_at_visit",
    "clinic_name",
    "diagnosis_text",
    "bp_raw",
    "height_cm",
    "weight_kg",
    "bmi"
]

existing_columns = [col for col in required_columns if col in df.columns]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("[WARNING] คอลัมน์ที่ขาดหายไปจากไฟล์ต้นฉบับ:", missing_columns)

df = df[existing_columns].copy()

# -----------------------------
# 3) ทำความสะอาดข้อความและค่าว่าง
# -----------------------------
text_columns = ["visit_id", "patient_id", "gender", "clinic_name", "diagnosis_text", "bp_raw"]

EMPTY_PLACEHOLDERS = ["", "-", "N/A", "NA", "nan", "None", "none", "null", "NULL"]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(EMPTY_PLACEHOLDERS, pd.NA)

# -----------------------------
# 4) แปลงวันที่เป็น YYYY-MM-DD (รองรับหลาย format เพื่อเก็บค่าครบทุกแถว)
# -----------------------------
DATE_FORMATS = [
    "%Y-%m-%d",            # 2025-06-15  (หลัก)
    "%Y/%m/%d",            # 2025/06/15
    "%Y-%m-%d %H:%M:%S",   # 2025-06-15 08:30:00
    "%Y/%m/%d %H:%M:%S",   # 2025/06/15 08:30:00
    "%d/%m/%Y",            # 15/06/2025  (dayfirst)
    "%d-%m-%Y",            # 15-06-2025
    "%d/%m/%Y %H:%M:%S",   # 15/06/2025 08:30:00
]

def parse_date_robust(val) -> pd.Timestamp:
    """ลองแปลงค่าวันที่ทีละ format จนสำเร็จ"""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    # ตัด timezone suffix ที่อาจติดมา เช่น "+07:00"
    if "+" in s:
        s = s.split("+")[0].strip()
    for fmt in DATE_FORMATS:
        try:
            from datetime import datetime
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    # fallback สุดท้าย: ให้ pandas ลองเอง
    try:
        return pd.to_datetime(s, dayfirst=False)
    except Exception:
        return pd.NaT

# เก็บ raw values ไว้ก่อน (ก่อนที่ astype("string") จะแปลง)
raw_date_col = df["visit_date"].copy()

# รอบแรก: แปลง format หลัก YYYY-MM-DD อย่างรวดเร็ว
df["visit_date"] = pd.to_datetime(df["visit_date"], format="%Y-%m-%d", errors="coerce")

# รอบสอง: แถวที่ยังเป็น NaT ให้ลอง format อื่น
nat_mask = df["visit_date"].isna()
if nat_mask.any():
    df.loc[nat_mask, "visit_date"] = raw_date_col[nat_mask].apply(parse_date_robust)

failed_dates = df["visit_date"].isna().sum()
if failed_dates > 0:
    print(f"[WARNING] วันที่แปลงไม่ได้ {failed_dates} แถว (เก็บเป็น NaT)")
else:
    print("[OK] แปลงวันที่สำเร็จครบทุกแถว")


# -----------------------------
# 5) แปลงคอลัมน์ตัวเลข
# -----------------------------
number_columns = ["age_at_visit", "height_cm", "weight_kg", "bmi"]
for col in number_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 6) แยก BP เช่น 120/80 หรือ 155.00 / 96.00 เป็น systolic_bp และ diastolic_bp
# -----------------------------
if "bp_raw" in df.columns:
    # 1) ลบ comma thousand separator ก่อน เช่น 1,124.00 → 1124.00
    bp_clean = df["bp_raw"].str.replace(",", "", regex=False)
    # 2) รองรับ 1-4 หลัก + ทศนิยม (เช่น 0.00 / 0.00, 120/80, 155.00 / 96.00)
    bp_split = bp_clean.str.extract(
        r"^\s*(\d{1,4}(?:\.\d+)?)\s*/\s*(\d{1,4}(?:\.\d+)?)\s*$"
    )
    df["systolic_bp"] = pd.to_numeric(bp_split[0], errors="coerce").round().astype("Int64")
    df["diastolic_bp"] = pd.to_numeric(bp_split[1], errors="coerce").round().astype("Int64")
    # หมายเหตุ: ค่าที่ผิดปกติ (เช่น 0, 7684) จะถูก mask_out_of_range() ในขั้น 7 เปลี่ยนเป็น NaN
else:
    df["systolic_bp"] = np.nan
    df["diastolic_bp"] = np.nan

# -----------------------------
# 7) ตรวจสอบช่วงค่าผิดปกติ แล้วเปลี่ยนเป็นค่าว่าง
#    ใช้ notna() guard ก่อน เพื่อไม่ให้ NaN ถูก overwrite ซ้ำซ้อน
# -----------------------------
def mask_out_of_range(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """เปลี่ยนค่าที่อยู่นอกช่วง [lo, hi] เป็น NaN โดยไม่แตะค่าที่ว่างอยู่แล้ว"""
    valid = series.notna()
    out_of_range = valid & ~series.between(lo, hi)
    series = series.copy()
    series.loc[out_of_range] = np.nan
    return series

if "age_at_visit" in df.columns:
    df["age_at_visit"] = mask_out_of_range(df["age_at_visit"], 0, 120)
if "height_cm" in df.columns:
    df["height_cm"] = mask_out_of_range(df["height_cm"], 80, 250)
if "weight_kg" in df.columns:
    df["weight_kg"] = mask_out_of_range(df["weight_kg"], 1, 300)
if "bmi" in df.columns:
    df["bmi"] = mask_out_of_range(df["bmi"], 10, 80)

df["systolic_bp"] = mask_out_of_range(df["systolic_bp"], 60, 250)
df["diastolic_bp"] = mask_out_of_range(df["diastolic_bp"], 30, 150)

# -----------------------------
# 8) จัดกลุ่มโรคจาก diagnosis_text
# -----------------------------
def classify_disease(diagnosis) -> str:
    if pd.isna(diagnosis):
        return "ไม่ระบุ"
    text = str(diagnosis).lower().strip()
    # เบาหวาน: ตรวจทั้งแบบขึ้นต้น, มีคำต่อท้าย, หรืออยู่กลางประโยค
    if any(k in text for k in ["diabetes", "เบาหวาน"]) or \
       text == "dm" or text.startswith("dm ") or text.startswith("dm,") or \
       " dm" in text:
        return "เบาหวาน"
    # ความดัน
    if any(k in text for k in ["hypertension", "ความดัน"]) or \
       text == "ht" or text.startswith("ht ") or text.startswith("ht,") or \
       " ht" in text:
        return "ความดันโลหิตสูง"
    # ไขมัน
    if any(k in text for k in ["dyslipidemia", "ไขมัน"]) or \
       text == "dlp" or text.startswith("dlp ") or text.startswith("dlp,") or \
       " dlp" in text:
        return "ไขมันในเลือดสูง"
    return "อื่น ๆ"

df["disease_group"] = df["diagnosis_text"].apply(classify_disease)

# -----------------------------
# 9) ลบแถวที่ไม่มี visit_id หรือ patient_id
# -----------------------------
before_remove_missing = len(df)
df = df.dropna(subset=["visit_id", "patient_id"])
print(f"ลบแถวที่ไม่มี visit_id หรือ patient_id: {before_remove_missing - len(df)} แถว")

# -----------------------------
# 10) ตรวจสอบ Visit ซ้ำ
# -----------------------------
duplicate_visits = df[df.duplicated(subset=["patient_id", "visit_id"], keep=False)].copy()
if not duplicate_visits.empty:
    out_path = "duplicate_visit_id.xlsx"
    try:
        duplicate_visits.to_excel(out_path, index=False)
        print(f"พบ visit_id ซ้ำ: {len(duplicate_visits)} แถว → บันทึกไว้ใน {out_path}")
    except PermissionError:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"duplicate_visit_id_{ts}.xlsx"
        duplicate_visits.to_excel(out_path, index=False)
        print(f"[WARNING] ไฟล์ duplicate_visit_id.xlsx ถูกเปิดอยู่ -- บันทึกแทนที่ {out_path}")
else:
    print("[OK] ไม่พบ visit_id ซ้ำ")


df = df.drop_duplicates(subset=["patient_id", "visit_id"], keep="first")

# -----------------------------
# 11) จัดรูปแบบวันที่และเวลาเพื่อ Export
#     ใช้ where() เพื่อรักษา NaT เป็น None แทนการ format ทับ
# -----------------------------
df["visit_date"] = df["visit_date"].where(
    df["visit_date"].isna(),
    df["visit_date"].dt.strftime("%Y-%m-%d")
)
if "visit_time" in df.columns:
    df["visit_time"] = df["visit_time"].astype("string").str.strip()

# -----------------------------
# 12) บันทึกผลลัพธ์
# -----------------------------
df.to_excel("visits_cleaned.xlsx", index=False)
df.to_csv("visits_cleaned.csv", index=False, encoding="utf-8-sig")

# -----------------------------
# 13) รายงานสรุป
# -----------------------------
print("\n--- สรุปผลการทำความสะอาด ---")
print("จำนวนแถวสุดท้าย:", len(df))
print("จำนวนผู้รับบริการไม่ซ้ำ:", df["patient_id"].nunique())
print("จำนวน Visit ไม่ซ้ำ:", df["visit_id"].nunique())
print("\nจำนวนข้อมูลว่าง:")
print(df.isna().sum())
print("\nจำนวนกลุ่มโรค:")
print(df["disease_group"].value_counts(dropna=False))
print("\nสร้างไฟล์สำเร็จ: visits_cleaned.xlsx, visits_cleaned.csv")

# -----------------------------
# 14) นับจำนวนครั้งที่ patient มาในแต่ละเดือน
# -----------------------------

# แปลง visit_date กลับเป็น datetime ก่อน (เพราะขั้น 11 แปลงเป็น string แล้ว)
df["visit_date_dt"] = pd.to_datetime(df["visit_date"], errors="coerce")

# สร้างคอลัมน์ year_month เช่น "2026-05"
df["year_month"] = df["visit_date_dt"].dt.to_period("M").astype(str)

# นับ visit ต่อ patient ต่อเดือน
monthly_count = (
    df.groupby(["patient_id", "year_month"], sort=True)
    .agg(
        visit_count=("visit_id", "count"),
        first_visit=("visit_date", "min"),
        last_visit=("visit_date", "max"),
    )
    .reset_index()
)

# Merge visit_count กลับเข้า df ทุกแถว
df = df.merge(
    monthly_count[["patient_id", "year_month", "visit_count"]],
    on=["patient_id", "year_month"],
    how="left"
)

# ลบคอลัมน์ชั่วคราว
df = df.drop(columns=["visit_date_dt"])

# บันทึก: ตาราง Visit-level พร้อม visit_count_in_month
df.to_excel("visits_with_monthly_count.xlsx", index=False)
df.to_csv("visits_with_monthly_count.csv", index=False, encoding="utf-8-sig")

# บันทึก: ตาราง Summary รายผู้รับบริการต่อเดือน
monthly_count.to_excel("monthly_visit_summary.xlsx", index=False)
monthly_count.to_csv("monthly_visit_summary.csv", index=False, encoding="utf-8-sig")

# รายงาน
print("\n--- สรุปจำนวน Visit ต่อเดือน ---")
print(f"จำนวน patient ไม่ซ้ำ  : {monthly_count['patient_id'].nunique()}")
print(f"จำนวนเดือนที่มีข้อมูล : {monthly_count['year_month'].nunique()}")
print("\nกระจายจำนวนครั้ง/เดือน:")
print(monthly_count["visit_count"].value_counts().sort_index().to_string())
print("\nTop 10 patient มาบ่อยที่สุดในเดือนเดียว:")
top10 = (
    monthly_count.sort_values("visit_count", ascending=False)
    .head(10)[["patient_id", "year_month", "visit_count", "first_visit", "last_visit"]]
)
print(top10.to_string(index=False))
print("\nสร้างไฟล์สำเร็จ:")
print("  - visits_with_monthly_count.xlsx / .csv")
print("  - monthly_visit_summary.xlsx / .csv")
