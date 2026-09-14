import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import uuid

st.set_page_config(page_title="Vichaivej Omnoi Command Center", page_icon="🏥", layout="wide")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration
USE_DEMO_DATA = True

# Colors
COLORS = {
    "Navy": "#16325C",
    "Teal": "#009A9A",
    "Green": "#16A34A",
    "Amber": "#F59E0B",
    "Red": "#DC2626",
    "Purple": "#7C3AED",
    "Background": "#F8FAFC"
}

def inject_css():
    st.markdown(f"""
    <style>
    .kpi-card {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid {{COLORS['Teal']}};
        margin-bottom: 15px;
    }}
    .kpi-title {{ color: #64748b; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }}
    .kpi-value {{ color: {{COLORS['Navy']}}; font-size: 1.8rem; font-weight: 700; }}
    .kpi-delta.positive {{ color: {{COLORS['Green']}}; font-size: 0.9rem; font-weight: 600; }}
    .kpi-delta.negative {{ color: {{COLORS['Red']}}; font-size: 0.9rem; font-weight: 600; }}
    
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
        border: 2px solid {{COLORS['Purple']}}; background-color: #f5f3ff;
    }}
    </style>
    """.replace("{COLORS['Teal']}", COLORS['Teal'])
       .replace("{COLORS['Navy']}", COLORS['Navy'])
       .replace("{COLORS['Green']}", COLORS['Green'])
       .replace("{COLORS['Red']}", COLORS['Red'])
       .replace("{COLORS['Purple']}", COLORS['Purple']), unsafe_allow_html=True)

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
# Demo Data Generation
# ============================================================
@st.cache_data
def generate_demo_data():
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
        # Create seasonal effect and trend
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
    
    # 4. Clinical Visits
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
def load_data():
    if USE_DEMO_DATA:
        return generate_demo_data()
    # Logic to load from CSV/ZIP would go here
    return None

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
# Recommendation Engine
# ============================================================
def _recommend_packages(customer_row, clinical_row_or_none, role, packages_df):
    results = []
    
    # 1. Consent Check
    consent = customer_row.get("consent_status", "")
    if consent == "ไม่อนุญาตใช้เพื่อการตลาด" and role == "Sales":
        return [{"pkg": None, "score": 0, "reason": "ลูกค้าไม่อนุญาตให้ใช้ข้อมูลเพื่อการตลาด (Consent Denied)"}]
        
    age = customer_row.get("age", 35)
    interests = str(customer_row.get("health_interests", ""))
    budget = str(customer_row.get("budget_range", ""))
    
    for _, pkg in packages_df.iterrows():
        score = 50 # Base score
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
        
        # Determine match level
        match_level = "สูง" if score >= 80 else ("ปานกลาง" if score >= 60 else "พื้นฐาน")
        
        # Select reasons based on role
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
        
    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3] # Return top 3 (1 Best match, 2 alternatives)


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
    
    # Header
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
        recs = _recommend_packages(cust_row, clin_row, role, packages)
        if recs and recs[0]["pkg"] is not None:
            rcols = st.columns(3)
            for i, rec in enumerate(recs):
                with rcols[i]:
                    pkg = rec["pkg"]
                    css_class = "recommendation-card best-match" if i == 0 else "recommendation-card"
                    match_badge = get_badge_html(f"Match: {rec['match_level']}", "success" if i==0 else "info")
                    
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
    
    # Pre-calculate common metrics
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
        
        # Monthly Revenue Chart
        sales['month_year'] = sales['sale_date'].dt.to_period('M').astype(str)
        monthly = sales.groupby('month_year')['net_amount'].sum().reset_index()
        fig_rev = px.line(monthly, x='month_year', y='net_amount', title="รายได้รายเดือน (Actual)", markers=True, color_discrete_sequence=[COLORS["Navy"]])
        
        # Top Packages
        top_pkg = sales.groupby('package_name')['net_amount'].sum().nlargest(10).reset_index()
        fig_top = px.bar(top_pkg, x='net_amount', y='package_name', orientation='h', title="Top 10 Packages", color_discrete_sequence=[COLORS["Teal"]])
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        
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
                
                # Apply multiplier
                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat'] *= mult
                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat_lower'] *= mult
                forecast.loc[forecast['ds'] > df_p['ds'].max(), 'yhat_upper'] *= mult
                
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color=COLORS['Navy'])))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color=COLORS['Amber'], dash="dash")))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(245, 158, 11, 0.2)', name="Confidence Interval"))
                
                st.plotly_chart(fig_fc, use_container_width=True)
                
                # Calculate simple MAPE on train
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
        # Define funnel order
        status_order = ["New", "Contacted", "Interested", "Appointed", "Purchased", "Declined"]
        funnel["Status"] = pd.Categorical(funnel["Status"], categories=status_order, ordered=True)
        funnel = funnel.sort_values("Status")
        
        fig_f = go.Figure(go.Funnel(y=funnel["Status"], x=funnel["Count"], marker={"color": [COLORS["Navy"], COLORS["Teal"], COLORS["Amber"], COLORS["Purple"], COLORS["Green"], COLORS["Red"]]}))
        st.plotly_chart(fig_f, use_container_width=True)


# ============================================================
# Clinical Command Center Module
# ============================================================
def render_clinical_dashboard(clinical, appointments):
    st.markdown("## 🏥 Clinical Command Center")
    
    if clinical.empty:
        st.warning("ไม่มีข้อมูลการเข้ารับบริการในช่วงเวลาที่เลือก")
        return
        
    tab1, tab2, tab3, tab4 = st.tabs([
        "Clinical Overview", "Appointment Center", "Risk Registry", "Screening & Outcomes"
    ])
    
    with tab1:
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: kpi_card("ผู้ป่วยรับบริการ", f"{len(clinical):,}")
        with cc2: 
            needs_review = len(clinical[clinical['screening_result_status'] == 'Needs Review'])
            kpi_card("Needs Review", f"{needs_review:,}", invert_color=True)
        with cc3:
            referred = len(clinical[clinical['screening_result_status'] == 'Referred'])
            kpi_card("Referred", f"{referred:,}", invert_color=True)
        with cc4:
            completed = len(clinical[clinical['appointment_status'] == 'Completed'])
            kpi_card("Completed", f"{completed:,}")
            
        st.markdown("### แนวโน้มการเข้ารับบริการ")
        clinical['month_year'] = clinical['visit_date'].dt.to_period('M').astype(str)
        monthly_v = clinical.groupby('month_year').size().reset_index(name='visits')
        fig_v = px.line(monthly_v, x='month_year', y='visits', markers=True, color_discrete_sequence=[COLORS["Teal"]])
        st.plotly_chart(fig_v, use_container_width=True)
        
    with tab2:
        st.markdown("### 📅 Appointment Center")
        if not appointments.empty:
            st.dataframe(appointments, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลการนัดหมาย (ข้อมูลจำลองยังไม่สร้างในส่วนนี้แบบเต็ม)")
            
    with tab3:
        st.markdown("### ⚠️ Risk Registry (Tracking Only)")
        st.caption("หมายเหตุ: ข้อมูลในตารางนี้ใช้เพื่อการติดตาม (Tracking) ตามเกณฑ์ที่กำหนดไว้ล่วงหน้า ไม่ใช่การวินิจฉัยทางการแพทย์ (Not a Diagnosis).")
        
        # Apply Rules
        risk_df = clinical.copy()
        risk_df["priority"] = "P3 Monitor"
        risk_df["risk_tags"] = ""
        
        for idx, row in risk_df.iterrows():
            tags = []
            if row["systolic_bp"] >= 140 or row["diastolic_bp"] >= 90:
                tags.append("BP สูง")
            if row["bmi"] >= 25:
                tags.append("BMI สูง")
            if row["screening_result_status"] == "Needs Review":
                tags.append("Needs Review")
                risk_df.at[idx, "priority"] = "P2 Warning"
            if row["screening_result_status"] == "Referred":
                tags.append("Referral")
                risk_df.at[idx, "priority"] = "P1 Urgent"
                
            risk_df.at[idx, "risk_tags"] = ", ".join(tags)
            
        show_risk = risk_df[risk_df["risk_tags"] != ""]
        show_risk = show_risk[["patient_id", "visit_date", "priority", "risk_tags", "screening_result_status", "follow_up_date"]]
        st.dataframe(show_risk.sort_values("priority"), use_container_width=True, hide_index=True)
        
    with tab4:
        st.markdown("### 📊 Screening & Outcomes")
        scr_sum = clinical["screening_result_status"].value_counts().reset_index()
        scr_sum.columns = ["Status", "Count"]
        fig_scr = px.pie(scr_sum, values="Count", names="Status", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_scr, use_container_width=True)


# ============================================================
# Main Application
# ============================================================
def main():
    inject_css()
    
    # Load Data
    data = load_data()
    if data is None:
        st.error("Failed to load data.")
        return
        
    packages, customers, sales_transactions, clinical_visits, appointments, package_recommendations, sales_targets = data
    
    # Sidebar Role Selection
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
            
        branches = ["All"] + list(customers["preferred_branch"].unique())
        sel_branch = st.selectbox("สาขา", branches)
        
        channels = ["All"] + list(sales_transactions["channel"].unique())
        sel_channel = st.selectbox("ช่องทาง", channels)
        
        st.markdown("---")
        module = st.radio("📂 เลือกโมดูล", ["Sales Command Center", "Clinical Command Center", "Patient Insight Card"])
        
    # Apply global filters
    f_sales = apply_global_filters(sales_transactions, "sale_date", start_d, end_d, sel_branch, sel_channel)
    f_clin = apply_global_filters(clinical_visits, "visit_date", start_d, end_d, sel_branch, "All")
    
    # Routing
    if module == "Sales Command Center":
        render_sales_dashboard(f_sales, sales_targets, customers, packages)
        
    elif module == "Clinical Command Center":
        if user_role == "Sales":
            st.error("🚫 คุณไม่มีสิทธิ์เข้าถึง Clinical Command Center")
        else:
            render_clinical_dashboard(f_clin, appointments)
            
    elif module == "Patient Insight Card":
        st.markdown("## 🔍 ค้นหาข้อมูลผู้รับบริการ")
        search_q = st.text_input("กรอกชื่อ หรือ Customer ID (เช่น CUS-1000)")
        if search_q:
            # Simple search
            res = customers[customers["customer_id"].str.contains(search_q, case=False) | customers["full_name"].str.contains(search_q, case=False)]
            if not res.empty:
                sel_cust = st.selectbox("เลือกลูกค้า", res["customer_id"].tolist(), format_func=lambda x: f"{x} - {res[res['customer_id']==x]['full_name'].values[0]}")
                if sel_cust:
                    render_insight_card(sel_cust, user_role, customers, f_sales, clinical_visits, appointments, packages)
            else:
                st.warning("ไม่พบข้อมูล")

if __name__ == "__main__":
    main()


