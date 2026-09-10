# Healthcare Analysis Dashboard

แดชบอร์ดวิเคราะห์ข้อมูลการรับบริการของผู้ป่วยสร้างด้วย Python + Streamlit

## ▶️ รันบน Streamlit Cloud
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 🗂️ โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | Streamlit Dashboard |
| `clean_visits.py` | ทำความสะอาดข้อมูลและผลิตไฟล์ CSV |
| `match_patient_id.py` | จับคู่ HN กับ patient_id |
| `visits_cleaned.csv` | ข้อมูล Visit ที่ clean แล้ว |
| `monthly_visit_summary.csv` | สรุปจำนวน Visit รายเดือนต่อ Patient |
| `visits_with_monthly_count.csv` | ข้อมูล Visit พร้อม visit_count ต่อเดือน |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit theme/config |

## 🚀 ขั้นตอนการ Deploy บน Streamlit Cloud

1. Push โปรเจกต์นี้ขึ้น GitHub repository
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เข้าสู่ระบบด้วย GitHub account
4. กด **New app** → เลือก repository → Branch: `main` → Main file: `app.py`
5. กด **Deploy!**

## ⚙️ รันในเครื่อง

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **หมายเหตุ**: ต้องรัน `clean_visits.py` ก่อนเพื่อสร้างไฟล์ CSV ที่จำเป็น
