# 🏥 Clinical Command Center & Smart Health Dashboard

แดชบอร์ดวิเคราะห์ข้อมูลสุขภาพผู้ป่วย พร้อมระบบ AI Analysis ประเมินความเสี่ยงและแนะนำแพ็กเกจสุขภาพแบบ Personalized ตามมาตรฐานโรงพยาบาลวิชัยเวช อินเตอร์เนชั่นแนล อ้อมน้อย

## ▶️ Live Application
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://healthcare-dashboard-4ral5ztwl9x6t8oajcjz4n.streamlit.app/)

คลิกปุ่มด้านบนหรือเข้าสู่ระบบผ่านลิงก์: [https://healthcare-dashboard-4ral5ztwl9x6t8oajcjz4n.streamlit.app/](https://healthcare-dashboard-4ral5ztwl9x6t8oajcjz4n.streamlit.app/)

---

## 🏗️ System Architecture & Modular Structure

โปรเจกต์ได้รับการออกแบบตามหลักการ **Clean Modular Architecture** เพื่อประสิทธิภาพสูงสุดในการประมวลผล การบำรุงรักษา และการต่อขยายในอนาคต:

```text
Project Dashboard/
├── app.py                      # Main Entrypoint: Navigation, Global State, Routing (~110 บรรทัด สะอาด ชัดเจน)
├── modules/
│   ├── __init__.py
│   ├── config.py               # Design Tokens, Color Palette, Package Catalogs, Disease Config
│   ├── styles.py               # Complete CSS Design System (Executive Glassmorphism, Typography, Cards)
│   ├── data_loader.py          # Data Pipeline: Caching, CSV/ZIP Fallback, Data Preprocessing, Aggregation
│   ├── clinical_engine.py       # Clinical Logic: Risk Scoring, Tier 1-4 Assessment, Package Recommendations
│   └── views/
│       ├── __init__.py
│       ├── general_view.py     # หน้าแรก: General Command Center (Pulse KPIs, Sunburst, Risk Table, Profile)
│       ├── forecast_view.py    # หน้าพยากรณ์: AI Forecast Dashboard 24 เดือน (พ.ศ. 2568-2569)
│       ├── disease_view.py     # หน้าโรคเรื้อรัง: Dialysis, Diabetes, HT, Dyslipidemia Command Centers
│       └── package_view.py     # หน้าระบบคัดกรอง: AI Health Data Architect & 4-Level Brochure Grid
│
├── archive/                    # โฟลเดอร์จัดเก็บสคริปต์สำรองและ Patch เพื่อความเป็นระเบียบ
│   ├── patch_app*.py
│   ├── raw_df.py
│   ├── app_old_lead_scoring.py
│   └── app_clinical_command_center.py
│
├── clean_visits.py             # Data preparation script (ทำความสะอาดข้อมูลดิบ)
├── match_patient_id.py         # Data preparation script (จับคู่ HN กับ patient_id)
├── load_data.py                # Wrapper script สำหรับ backward compatibility
├── requirements.txt            # Python dependencies
└── README.md                   # คู่มือระบบและสถาปัตยกรรม
```

---

## 🧠 Core Features & Functional Centers

### 1. 🏥 General Dashboard (หน้าแรก)
- **Pulse + KPIs:** ภาพรวม Health Score (0-100), จำนวนเคส, ผู้รับบริการ, ค่าความดันเฉลี่ย และสัดส่วนกลุ่มเสี่ยงวิกฤต
- **Data Quality Indicators:** มาตรวัดความสมบูรณ์ของข้อมูลความดัน, อายุผู้ใหญ่, การจัดกลุ่มโรค และ BMI จริง
- **Sunburst Chart (Drill-down):** แผนภูมิดวงอาทิตย์คลิกเจาะลึกจากกลุ่มโรคลงไปยังรหัสโรครายตัว
- **Command Action Panel:** ดาวน์โหลด Alert List (CSV) และสร้างคิวติดตามออกตรวจเชิงรุก (Outreach Queue)
- **Clinical Biomarkers & Risk Association:** กราฟ Scatter ความสัมพันธ์อายุกับความดันโลหิต พร้อมเส้น OLS Trendline และพีระมิดประชากร
- **Smart Package Recommender & Patient Profile (3 Tabs):**
  1. *ข้อมูลสุขภาพ:* คำนวณ Health Score พร้อมแสดง AI Analysis Insights
  2. *ประวัติการรักษา:* ไฮไลท์ค่าผิดปกติ (BP ≥ 140, BMI ≥ 25)
  3. *แผนการตรวจที่แนะนำ:* แนะนำ Base Package (Essential, Advanced, Longevity) + Add-ons (Mammogram, PSA) พร้อมปุ่มดาวน์โหลด Proposal

### 2. 🔮 AI Forecast (พยากรณ์ 2568 - 2569)
- โมเดลพยากรณ์ปริมาณผู้รับบริการและกลุ่มเสี่ยง NCDs ล่วงหน้า 24 เดือน ด้วย Linear Trend & Thai Seasonal Index
- ปรับแผนสถานการณ์ได้ 3 ระดับ: มาตรฐาน (Baseline), เชิงรุก (+10%), อนุรักษ์นิยม (-5%)
- แสดงช่วงความเชื่อมั่น 95% Confidence Band
- บทวิเคราะห์เชิงกลยุทธ์ 3 ด้าน: แผนโปรโมชั่น, การจัดอัตรากำลังแพทย์/พยาบาล/ห้อง Lab, และการดูแลเชิงป้องกัน
- ตารางผลพยากรณ์พร้อมปุ่มดาวน์โหลด CSV

### 3. 🩺 Chronic Disease Command Centers
- **ศูนย์ล้างไต (Dialysis):** ติดตามกลุ่ม P1-Urgent ที่ขาดนัดเกิน 7 วัน
- **ศูนย์เบาหวาน (Diabetes):** ควบคุมระดับ HbA1c < 7.0
- **ศูนย์ความดันโลหิตสูง (Hypertension):** ควบคุมความดัน < 140/90 mmHg
- **ศูนย์ไขมันในเลือดสูง (Dyslipidemia):** ควบคุมระดับ LDL < 100 mg/dL
- **Combined Care Package Generator:** ตรวจจับพหุโรค (Multi-morbidity) และผสานรายการตรวจพร้อมคำนวณราคาอัตโนมัติ

### 4. 🧬 กลุ่มอื่น ๆ (AI Health Data Architect)
- สถาปัตยกรรมคัดกรองสุขภาพ 4 ระดับ (Level 1 ตรวจทั่วไป, Level 2 มีความเสี่ยง, Level 3 โรคระยะแรก, Level 4 ตรวจเชิงลึก) ของ รพ.วิชัยเวชฯ อ้อมน้อย
- คำนวณความเสี่ยงเฉพาะบุคคล (ประเมินรอบเอว, BMI, BP, ประวัติโรค)
- โบรชัวร์ดิจิทัล 12 แพ็กเกจย่อย พร้อมแท็ก ⭐ AI Best Match
- Interactive Popovers: เครื่องคำนวณ Metabolic, กราฟแนวโน้มค่าไต/ตับย้อนหลัง 3 ปี, และตัวเลือก Add-ons ส่องกล้อง/CT/Tumor Markers
- ส่งออกรายงานทางการแพทย์ และจำลองการส่งข้อมูลเข้าสู่ LINE Official Account

---

## 🚀 การติดตั้งและรันโปรเจกต์ (Local Development)

1. ติดตั้ง Dependencies:
```bash
pip install -r requirements.txt
```

2. (ทางเลือก) สร้างข้อมูลที่ Clean แล้ว:
```bash
python clean_visits.py
```

3. รัน Streamlit Dashboard:
```bash
streamlit run app.py
```
