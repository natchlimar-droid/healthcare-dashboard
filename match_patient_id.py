import pandas as pd

# -----------------------------
# 1) อ่านไฟล์ Excel
# -----------------------------
raw_df = pd.read_excel("Datavisit_05-062026.xlsx")
mapping_df = pd.read_excel("patient_mapping.xlsx")

print("คอลัมน์ใน raw_df:", raw_df.columns.tolist())
print("คอลัมน์ใน mapping_df:", mapping_df.columns.tolist())

# -----------------------------
# 2) ค้นหาคอลัมน์ HN แบบอัตโนมัติ (กันปัญหาชื่อไม่ตรงเป๊ะ)
# -----------------------------
def find_hn_column(df):
    for col in df.columns:
        if str(col).strip().upper() == "HN":
            return col
    raise KeyError(f"ไม่พบคอลัมน์ HN ในไฟล์ ชื่อคอลัมน์ที่มี: {df.columns.tolist()}")

hn_col_raw = find_hn_column(raw_df)
hn_col_map = find_hn_column(mapping_df)

# เปลี่ยนชื่อให้เป็น "HN" มาตรฐานทั้งสองไฟล์
raw_df = raw_df.rename(columns={hn_col_raw: "HN"})
mapping_df = mapping_df.rename(columns={hn_col_map: "HN"})

# -----------------------------
# 3) ทำความสะอาด HN
# -----------------------------
raw_df["HN"] = raw_df["HN"].astype(str).str.strip()
mapping_df["HN"] = mapping_df["HN"].astype(str).str.strip()

# กรณี Excel อ่าน HN เป็น 123456.0 ให้ตัด .0 ออก
raw_df["HN"] = raw_df["HN"].str.replace(r"\.0$", "", regex=True)
mapping_df["HN"] = mapping_df["HN"].str.replace(r"\.0$", "", regex=True)

# -----------------------------
# 4) ตรวจสอบ HN ซ้ำใน Mapping
# -----------------------------
duplicate_hn = mapping_df[mapping_df.duplicated(subset=["HN"], keep=False)]

if not duplicate_hn.empty:
    print("พบ HN ซ้ำใน patient_mapping.xlsx")
    print(duplicate_hn.sort_values("HN"))
    raise SystemExit("กรุณาแก้ HN ให้เหลือ 1 แถวต่อ 1 patient_id ก่อนรันใหม่")

# -----------------------------
# 5) Match HN กับ patient_id
# -----------------------------
result_df = raw_df.merge(
    mapping_df[["HN", "patient_id"]],
    on="HN",
    how="left"
)

# -----------------------------
# 6) ตรวจสอบ HN ที่หา patient_id ไม่พบ
# -----------------------------
not_found_df = result_df[result_df["patient_id"].isna()].copy()

print("จำนวนแถว Raw Data:", len(result_df))
print("จำนวน HN ไม่ซ้ำใน Raw Data:", result_df["HN"].nunique())
print("จำนวนแถวที่หา patient_id ไม่พบ:", len(not_found_df))

if not_found_df.empty:
    print("จับคู่ patient_id สำเร็จครบทุกแถว")
else:
    not_found_df.to_excel("hn_not_found.xlsx", index=False)
    print("บันทึก HN ที่ไม่พบไว้ในไฟล์: hn_not_found.xlsx")

# -----------------------------
# 7) เปลี่ยนชื่อคอลัมน์ให้ตรงกับมาตรฐานโปรเจกต์
#    (แก้ฝั่งซ้ายให้ตรงกับชื่อจริงในไฟล์ raw ถ้าไม่ตรง)
# -----------------------------
rename_map = {
    "ลำดับ": "record_no",
    "No": "record_no",
    "วันที่": "visit_date",
    "Date": "visit_date",
    "เวลา": "visit_time",
    "Time": "visit_time",
    "VN": "visit_id",
    "Sex": "gender",
    "เพศ": "gender",
    "Age": "age_at_visit",
    "อายุ": "age_at_visit",
    "Clinic": "clinic_name",
    "คลินิก": "clinic_name",
    "Diag": "diagnosis_text",
    "วินิจฉัย": "diagnosis_text",
    "BP": "bp_raw",
    "ความดัน": "bp_raw",
    "Height": "height_cm",
    "ส่วนสูง": "height_cm",
    "Weight": "weight_kg",
    "น้ำหนัก": "weight_kg",
    "BMI": "bmi"
}
result_df = result_df.rename(columns=rename_map)

# -----------------------------
# 8) จัดลำดับคอลัมน์ให้ตรงตามมาตรฐาน (เฉพาะคอลัมน์ที่มีจริง)
# -----------------------------
target_order = [
    "record_no", "visit_date", "visit_time", "visit_id", "patient_id",
    "gender", "age_at_visit", "clinic_name", "diagnosis_text", "bp_raw",
    "height_cm", "weight_kg", "bmi"
]
existing_cols = [c for c in target_order if c in result_df.columns]
other_cols = [c for c in result_df.columns if c not in existing_cols and c != "HN"]
final_df = result_df[existing_cols + other_cols]

# -----------------------------
# 9) บันทึกไฟล์ผลลัพธ์
# -----------------------------
final_df.to_excel("visits_with_patient_id.xlsx", index=False)
final_df.to_csv("visits_with_patient_id.csv", index=False, encoding="utf-8-sig")

print("\nคอลัมน์ในไฟล์ผลลัพธ์:", final_df.columns.tolist())
print("สร้างไฟล์สำเร็จ:")
print("- visits_with_patient_id.xlsx")
print("- visits_with_patient_id.csv")