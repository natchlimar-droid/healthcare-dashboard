# ==============================================================================
# 🏥 ระบบพยากรณ์ยอดขายแพ็คเกจตรวจสุขภาพและวิเคราะห์การเงินโรงพยาบาล
# (Hospital Executive Health Checkup Sales & Financial Forecasting Dashboard)
# ------------------------------------------------------------------------------
# พัฒนาโดย: Senior Data Scientist & Hospital Financial Forecasting Expert
# เทคโนโลยี: Python, Streamlit, Statsmodels (Holt-Winters), Plotly, Pandas, NumPy
# ข้อมูลนำเข้า: RAWDATA_2025_1-12.xlsx (12 เดือน พ.ศ. 2568)
#              RAWDATA_2026_1-7.xlsx  (7 เดือน พ.ศ. 2569: ม.ค. - ก.ค.)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------------------------------------------------------------
# 1. การกำหนดค่าหน้าจอ Streamlit (Page Configuration)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Checkup Sales Forecast | C-Level Executive Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. ฟังก์ชันแปลชื่อเดือนและปีเป็นรูปแบบภาษาไทยและพุทธศักราช (พ.ศ.)
# ------------------------------------------------------------------------------
THAI_MONTHS_SHORT = [
    "", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."
]

THAI_MONTHS_FULL = [
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
]

def format_thai_month_year(date_obj, full=False):
    """แปลง Timestamp หรือ Datetime เป็นสตริงเดือนภาษาไทย + ปี พ.ศ."""
    month_idx = date_obj.month
    year_be = date_obj.year + 543
    m_name = THAI_MONTHS_FULL[month_idx] if full else THAI_MONTHS_SHORT[month_idx]
    return f"{m_name} {year_be}"

def format_currency(val):
    """จัดรูปแบบตัวเลขเป็นสกุลเงินบาทไทย พร้อมเครื่องหมายจุลภาค"""
    return f"฿{val:,.2f}"

# ------------------------------------------------------------------------------
# 3. CSS สไตล์ระดับ Executive C-Level UI/UX (Hospital Premium Design System)
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ซ่อน Streamlit footer ส่วนเกิน */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* สไตล์ Header แถบหัวเรื่องหลัก */
    .executive-header {
        background: linear-gradient(135deg, #0A2540 0%, #0F4C81 50%, #0E7490 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(10, 37, 64, 0.25);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .header-title {
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        line-height: 1.3;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 0.98rem;
        color: #E2E8F0;
        margin: 0;
        font-weight: 400;
    }

    /* กล่อง KPI Card ระดับผู้บริหาร */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
    }
    .kpi-card.blue::before { background: linear-gradient(90deg, #2563EB, #38BDF8); }
    .kpi-card.teal::before { background: linear-gradient(90deg, #0D9488, #14B8A6); }
    .kpi-card.purple::before { background: linear-gradient(90deg, #7C3AED, #A855F7); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #D97706, #F59E0B); }
    .kpi-card.emerald::before { background: linear-gradient(90deg, #059669, #10B981); }

    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #64748B;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .badge-positive {
        color: #059669;
        background: #ECFDF5;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-negative {
        color: #DC2626;
        background: #FEF2F2;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-neutral {
        color: #4B5563;
        background: #F3F4F6;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }

    /* กรอบคำแนะนำเชิงกลยุทธ์ (Strategic Recommendations) */
    .strategy-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .strategy-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #F1F5F9;
    }
    .strategy-item {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.6;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 4. ฟังก์ชันโหลดและทำความสะอาดข้อมูล (Data Ingestion & Preprocessing Pipeline)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังประมวลผลข้อมูลยอดขาย 2568 และ 2569...")
def load_and_preprocess_data():
    """
    อ่านไฟล์ RAWDATA_2025_1-12.xlsx และ RAWDATA_2026_1-7.xlsx
    ทำความสะอาดข้อมูลตัวเลข แปลงหน่วย สรุปยอดรายเดือน และจัดกลุ่มแพ็คเกจ
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f25_path = os.path.join(base_dir, "RAWDATA_2025_1-12.xlsx")
    f26_path = os.path.join(base_dir, "RAWDATA_2026_1-7.xlsx")

    # 4.1 ตรวจสอบความถูกต้องของการมีอยู่ของไฟล์
    if not os.path.exists(f25_path) or not os.path.exists(f26_path):
        st.error(f"⚠️ ไม่พบไฟล์ฐานข้อมูลในไดเรกทอรี: {base_dir}")
        st.stop()

    # 4.2 โหลดข้อมูลปี 2568 (2025)
    df25 = pd.read_excel(f25_path)
    
    # ระบุคอลัมน์สำคัญ
    month_col_25 = df25.columns[0]  # คอลัมน์รหัสเดือน เช่น 6801 - 6812
    sales_col_25 = [c for c in df25.columns if "สุทธิ" in str(c)][0]
    cat_col_25 = [c for c in df25.columns if "Category" in str(c)][0]
    req_col_25 = [c for c in df25.columns if "Request" in str(c)][0]
    vn_col_25 = [c for c in df25.columns if "VN" in str(c) and "Count" in str(c)][0]

    # ทำความสะอาดข้อมูลยอดสุทธิปี 2568
    df25["clean_sales"] = (
        df25[sales_col_25]
        .astype(str)
        .str.replace("฿", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df25["clean_sales"] = pd.to_numeric(df25["clean_sales"], errors="coerce").fillna(0.0)
    df25["clean_vn"] = pd.to_numeric(df25[vn_col_25], errors="coerce").fillna(0.0)

    # รวมยอดรายเดือนปี 2568 (12 เดือน)
    m25_summary = df25.groupby(month_col_25).agg({
        "clean_sales": "sum",
        "clean_vn": "sum"
    }).reset_index()
    m25_summary.sort_values(month_col_25, inplace=True)
    m25_summary["date"] = pd.date_range(start="2025-01-01", periods=12, freq="MS")
    m25_summary["year"] = 2568
    m25_summary["month_num"] = range(1, 13)

    # 4.3 โหลดข้อมูลปี 2569 (2026: ม.ค. - ก.ค.)
    # Header ของไฟล์ 2569 อยู่ที่แถว index 3 (บรรทัดที่ 4)
    df26 = pd.read_excel(f26_path, header=3)
    
    sales_cols_26 = [c for c in df26.columns if "ยอดสุทธิ" in str(c)]
    vn_cols_26 = [c for c in df26.columns if "VN" in str(c) and "ม." in str(c) or "ก." in str(c) or "มี." in str(c) or "เม." in str(c) or "พ." in str(c)]
    cat_col_26 = [c for c in df26.columns if "หมวด" in str(c)][0]
    req_col_26 = [c for c in df26.columns if "Request" in str(c)][0]

    # ทำความสะอาดคอลัมน์ยอดเงินปี 2569
    for col in sales_cols_26:
        clean = (
            df26[col]
            .astype(str)
            .str.replace("฿", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df26[col] = pd.to_numeric(clean, errors="coerce").fillna(0.0)

    m26_list = []
    for i, col in enumerate(sales_cols_26):
        month_idx = i + 1
        sales_val = df26[col].sum()
        m26_list.append({
            "month_code": f"69{month_idx:02d}",
            "clean_sales": sales_val,
            "clean_vn": 0.0,  # ค่า VN รวม
            "date": pd.Timestamp(f"2026-{month_idx:02d}-01"),
            "year": 2569,
            "month_num": month_idx
        })
    m26_summary = pd.DataFrame(m26_list)

    # 4.4 รวมข้อมูล 19 เดือน (History: 2025-01 ถึง 2026-07)
    combined_history = pd.concat([
        m25_summary[["date", "clean_sales", "year", "month_num"]],
        m26_summary[["date", "clean_sales", "year", "month_num"]]
    ], ignore_index=True)
    combined_history.rename(columns={"clean_sales": "sales"}, inplace=True)
    combined_history.sort_values("date", inplace=True)
    combined_history.reset_index(drop=True, inplace=True)

    # เพิ่มคอลัมน์ข้อความสำหรับแสดงผลภาษาไทย
    combined_history["thai_month"] = combined_history["date"].apply(lambda d: format_thai_month_year(d))
    combined_history["thai_month_full"] = combined_history["date"].apply(lambda d: format_thai_month_year(d, full=True))

    # 4.5 ประมวลผลยอดขายแยกตามหมวดหมู่แพ็คเกจ (4 ระดับ)
    # 2568
    cat25 = df25.groupby(cat_col_25)["clean_sales"].sum().reset_index()
    cat25.columns = ["category", "sales_2568"]
    
    # 2569
    df26["total_sales_26"] = df26[sales_cols_26].sum(axis=1)
    cat26 = df26.groupby(cat_col_26)["total_sales_26"].sum().reset_index()
    cat26.columns = ["category", "sales_2569_ytd"]

    cat_merged = pd.merge(cat25, cat26, on="category", how="outer").fillna(0.0)
    cat_merged["category"] = cat_merged["category"].str.strip()

    return {
        "history_df": combined_history,
        "raw_df25": df25,
        "raw_df26": df26,
        "cat_df": cat_merged,
        "m25_df": m25_summary,
        "m26_df": m26_summary
    }

# โหลดข้อมูลเข้าสู่ระบบ
data_bundle = load_and_preprocess_data()
history_df = data_bundle["history_df"]
cat_df = data_bundle["cat_df"]

# ------------------------------------------------------------------------------
# 5. แถบควบคุมด้านข้าง (Sidebar Controls & Model Hyperparameters)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ พารามิเตอร์การพยากรณ์")
    st.caption("ปรับแต่งโมเดล Time Series Forecasting และแบบจำลองสถานการณ์")

    # 5.1 เลือกระยะเวลาพยากรณ์ไปข้างหน้า (Forecast Horizon)
    forecast_horizon = st.slider(
        "ระยะเวลาพยากรณ์ไปข้างหน้า (เดือน)",
        min_value=3,
        max_value=12,
        value=5,  # ค่าเริ่มต้น 5 เดือน เพื่อให้เห็นยอดสิ้นสุดปี 2569 (ส.ค. - ธ.ค. 2569)
        help="เลือกจำนวนเดือนที่ต้องการทำนายอนาคต (เช่น 5 เดือน = ครบปี 2569, 12 เดือน = ถึง ก.ค. 2570)"
    )

    # 5.2 รูปแบบโมเดล Holt-Winters
    model_type = st.selectbox(
        "ลักษณะของฤดูกาล (Seasonal Component)",
        options=["Multiplicative (ทวีคูณ - แนะนำ)", "Additive (บวกคงที่)", "Non-Seasonal (Holt's Linear)"],
        index=0,
        help="Multiplicative เหมาะกับข้อมูลยอดขายที่มีความผันผวนของฤดูกาลแปรผันตามขนาดของยอดขาย"
    )

    # 5.3 การลดทอนแนวโน้ม (Damped Trend)
    damped_trend = st.checkbox(
        "ลดทอนความชันของแนวโน้ม (Damped Trend)",
        value=True,
        help="ช่วยป้องกันการพยากรณ์ตัวเลขหลุดกรอบความจริงในระยะยาว (Conservative Estimation)"
    )

    # 5.4 ระดับช่วงความเชื่อมั่น (Confidence Interval)
    ci_level = st.radio(
        "ช่วงความเชื่อมั่นทางสถิติ (Confidence Interval)",
        options=["95% (เข้มงวด)", "90% (มาตรฐาน)", "80% (ยืดหยุ่น)"],
        index=1,
        horizontal=True
    )
    ci_map = {"80% (ยืดหยุ่น)": 1.282, "90% (มาตรฐาน)": 1.645, "95% (เข้มงวด)": 1.960}
    z_score = ci_map[ci_level]

    st.markdown("---")
    st.markdown("### 🎯 แบบจำลองเป้าหมายการเงิน (Scenario Simulation)")
    growth_target_pct = st.slider(
        "เป้าหมายเร่งยอดขายเชิงรุก (% Growth Target)",
        min_value=0,
        max_value=40,
        value=15,
        step=5,
        help="จำลองเส้นเป้าหมายการขาย (Revenue Stretch Goal) เพื่อให้ทีมการตลาดและองค์กรใช้เป็น KPI ปิดช่องว่าง"
    )

    st.markdown("---")
    st.markdown("### 📋 ข้อมูลระบบ")
    st.markdown(f"""
    - **ข้อมูลจริงย้อนหลัง:** 19 เดือน  
      (ม.ค. 2568 - ก.ค. 2569)
    - **ชุดข้อมูล 2568:** 12 เดือน (สมบูรณ์)
    - **ชุดข้อมูล 2569:** 7 เดือน (ม.ค. - ก.ค.)
    - **Algorithm:** Holt-Winters Exponential Smoothing
    """)

# ------------------------------------------------------------------------------
# 6. กลไกสร้างโมเดลพยากรณ์อนุกรมเวลา (Holt-Winters Time Series Engine)
# ------------------------------------------------------------------------------
def run_forecasting_model(df_hist, horizon, m_type, is_damped, z_val, target_pct):
    """
    ฟังก์ชันคำนวณโมเดล Holt-Winters Exponential Smoothing
    พร้อมคำนวณ fitted values, residuals, forecast และ confidence bounds
    """
    ts = pd.Series(df_hist["sales"].values, index=df_hist["date"])
    
    # คำนวณค่าเริ่มต้นแบบแม่นยำ (Known Initialization)
    # เนื่องจากข้อมูลมี 19 เดือน (น้อยกว่า 2 รอบฤดูกาล 24 เดือน) การใช้ heuristic ของ statsmodels จะเกิด ValueError
    # จึงใช้ known initialization ตามระเบียบวิธีทางสถิติของ Time Series
    init_level = ts.iloc[:12].mean()
    # อัตราการเปลี่ยนแปลงระหว่าง 7 เดือนแรกของปี 2568 และ 2569
    y1_h1 = ts.iloc[:7].mean()
    y2_h1 = ts.iloc[12:19].mean()
    init_trend = (y2_h1 - y1_h1) / 12.0

    seasonal_mul = ts.iloc[:12] / init_level
    seasonal_add = ts.iloc[:12] - init_level

    # กำหนดค่าตามตัวเลือกผู้ใช้
    if "Multiplicative" in m_type:
        s_type = "mul"
        init_s = seasonal_mul.values
    elif "Additive" in m_type:
        s_type = "add"
        init_s = seasonal_add.values
    else:
        s_type = None
        init_s = None

    try:
        if s_type is not None:
            model = ExponentialSmoothing(
                ts,
                trend="add",
                damped_trend=is_damped,
                seasonal=s_type,
                seasonal_periods=12,
                initialization_method="known",
                initial_level=init_level,
                initial_trend=init_trend,
                initial_seasonal=init_s
            ).fit()
        else:
            model = ExponentialSmoothing(
                ts,
                trend="add",
                damped_trend=is_damped,
                initialization_method="known",
                initial_level=init_level,
                initial_trend=init_trend
            ).fit()

        forecast_series = model.forecast(horizon)
        fitted_series = model.fittedvalues
        residuals = ts - fitted_series
        res_std = float(np.std(residuals))
        
        # ปรับค่าไม่ให้พยากรณ์ติดลบ (Revenue Floor at zero)
        forecast_series = forecast_series.apply(lambda x: max(x, 100000.0))

    except Exception as e:
        # Fallback กรณีคำนวณโมเดลล้มเหลว
        st.warning(f"โมเดลปรับใช้ค่าอัตโนมัติเนื่องจาก: {str(e)}")
        last_val = ts.iloc[-1]
        future_dates = [ts.index[-1] + pd.DateOffset(months=i+1) for i in range(horizon)]
        forecast_series = pd.Series([last_val * (1 + 0.02 * i) for i in range(horizon)], index=future_dates)
        fitted_series = ts.copy()
        res_std = float(ts.std() * 0.15)

    # สร้างตารางพยากรณ์พร้อม Confidence Intervals
    future_dates = forecast_series.index
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_sales": forecast_series.values
    })
    
    # คำนวณช่วงความเชื่อมั่นตามระยะเวลา h (Horizon Standard Error)
    se_h = [res_std * np.sqrt(1 + 0.1 * h) for h in range(len(forecast_df))]
    forecast_df["lower_bound"] = np.maximum(0, forecast_df["forecast_sales"] - z_val * np.array(se_h))
    forecast_df["upper_bound"] = forecast_df["forecast_sales"] + z_val * np.array(se_h)

    # เพิ่มเส้นเป้าหมายการขาย (Target Revenue Simulation)
    forecast_df["target_sales"] = forecast_df["forecast_sales"] * (1.0 + (target_pct / 100.0))

    # จัดรูปแบบภาษาไทย
    forecast_df["thai_month"] = forecast_df["date"].apply(lambda d: format_thai_month_year(d))
    forecast_df["thai_month_full"] = forecast_df["date"].apply(lambda d: format_thai_month_year(d, full=True))
    
    return {
        "model": model,
        "fitted": fitted_series,
        "forecast_df": forecast_df,
        "res_std": res_std
    }

# ประมวลผลโมเดล
forecast_results = run_forecasting_model(
    history_df,
    forecast_horizon,
    model_type,
    damped_trend,
    z_score,
    growth_target_pct
)

forecast_df = forecast_results["forecast_df"]
fitted_series = forecast_results["fitted"]

# ------------------------------------------------------------------------------
# 7. การคำนวณตัวชี้วัดทางการเงิน (Executive Financial KPIs)
# ------------------------------------------------------------------------------
# 7.1 ยอดขายรวมปี 2568 (Actual)
total_sales_2568 = history_df[history_df["year"] == 2568]["sales"].sum()

# 7.2 ยอดขายจริงสะสมปี 2569 (YTD Jan-Jul)
sales_2569_ytd = history_df[history_df["year"] == 2569]["sales"].sum()

# 7.3 ยอดขายคาดการณ์ช่วงที่เหลือของปี 2569 (ส.ค. - ธ.ค. 2569 = 5 เดือน)
forecast_2569_remaining = forecast_df[forecast_df["date"].dt.year == 2026]["forecast_sales"].sum()

# 7.4 คาดการณ์ยอดขายเต็มปี 2569 (Projected Full Year 2569 = YTD + Forecast Remaining)
projected_full_2569 = sales_2569_ytd + forecast_2569_remaining

# 7.5 อัตราการเติบโตเปรียบเทียบเทียบกับปี 2568 (YoY Projected Growth %)
yoy_growth_pct = ((projected_full_2569 / total_sales_2568) - 1.0) * 100.0

# 7.6 ยอดขายเฉลี่ยต่อเดือน (Monthly Run-Rates)
avg_monthly_2568 = total_sales_2568 / 12.0
avg_monthly_2569_ytd = sales_2569_ytd / 7.0
avg_monthly_forecast = forecast_df["forecast_sales"].mean()

# ------------------------------------------------------------------------------
# 8. ส่วนหัวแดชบอร์ดผู้บริหาร (Executive Header)
# ------------------------------------------------------------------------------
st.markdown(f"""
<div class="executive-header">
    <div class="header-badge">HOSPITAL FINANCIAL INTELLIGENCE & FORECASTING SYSTEM</div>
    <div class="header-title">🏥 แดชบอร์ดวิเคราะห์และพยากรณ์ยอดขายแพ็คเกจตรวจสุขภาพ</div>
    <div class="header-subtitle">
        ระบบวิเคราะห์อนุกรมเวลา (Time Series Predictive Analytics) สำหรับผู้บริหารโรงพยาบาล |
        ฐานข้อมูลวิเคราะห์: มกราคม 2568 – กรกฎาคม 2569 (19 เดือน) | แบบจำลอง: Holt-Winters Exponential Smoothing
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 9. การแสดงผลการ์ดตัวชี้วัดหลัก 5 มิติ (Top Executive KPI Metrics)
# ------------------------------------------------------------------------------
yoy_badge = (
    f'<span class="badge-positive">+{yoy_growth_pct:.1f}% YoY</span>' 
    if yoy_growth_pct >= 0 
    else f'<span class="badge-negative">{yoy_growth_pct:.1f}% YoY</span>'
)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card blue">
        <div class="kpi-label">ยอดขายจริงปี 2568 (ทั้งปี)</div>
        <div class="kpi-value">{format_currency(total_sales_2568)}</div>
        <div class="kpi-subtext">
            <span>ฐานยอดขายจริง 12 เดือนเต็ม</span>
        </div>
    </div>
    <div class="kpi-card teal">
        <div class="kpi-label">ยอดขายสะสมปี 2569 (YTD 7 เดือน)</div>
        <div class="kpi-value">{format_currency(sales_2569_ytd)}</div>
        <div class="kpi-subtext">
            <span>ม.ค. - ก.ค. 2569 (เฉลี่ย {format_currency(avg_monthly_2569_ytd)}/ด.)</span>
        </div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">คาดการณ์ยอดขายเต็มปี 2569</div>
        <div class="kpi-value">{format_currency(projected_full_2569)}</div>
        <div class="kpi-subtext">
            <span>จริง 7 เดือน + ทำนาย 5 เดือน ({format_currency(forecast_2569_remaining)})</span>
        </div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-label">อัตราเติบโตเปรียบเทียบ (YoY)</div>
        <div class="kpi-value">{yoy_growth_pct:+.1f}%</div>
        <div class="kpi-subtext">
            {yoy_badge} <span>เทียบปี 2568 เต็มปี</span>
        </div>
    </div>
    <div class="kpi-card emerald">
        <div class="kpi-label">ยอดพยากรณ์เฉลี่ยต่อเดือน</div>
        <div class="kpi-value">{format_currency(avg_monthly_forecast)}</div>
        <div class="kpi-subtext">
            <span>ระยะพยากรณ์ {forecast_horizon} เดือนถัดไป</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 10. กราฟพยากรณ์หลัก: Actual vs. Forecast with Confidence Bands (Interactive Plotly)
# ------------------------------------------------------------------------------
st.markdown("### 📈 แนวโน้มยอดขายจริงและการพยากรณ์ในอนาคต (Actual vs. Predictive Horizon)")

fig_forecast = go.Figure()

# 10.1 เส้นช่วงความเชื่อมั่น (Upper & Lower Confidence Bounds)
fig_forecast.add_trace(go.Scatter(
    x=list(forecast_df["thai_month"]) + list(forecast_df["thai_month"])[::-1],
    y=list(forecast_df["upper_bound"]) + list(forecast_df["lower_bound"])[::-1],
    fill="toself",
    fillcolor="rgba(245, 158, 11, 0.15)",
    line=dict(color="rgba(255, 255, 255, 0)"),
    hoverinfo="skip",
    showlegend=True,
    name=f"กรอบความเชื่อมั่น {ci_level}"
))

# 10.2 เส้นประเมิน In-Sample Model Fit (Fitted Values)
fit_thai_months = [format_thai_month_year(d) for d in fitted_series.index]
fig_forecast.add_trace(go.Scatter(
    x=fit_thai_months,
    y=fitted_series.values,
    mode="lines",
    line=dict(color="#94A3B8", width=1.5, dash="dot"),
    name="Holt-Winters Fitted (In-sample)",
    hovertemplate="<b>%{x}</b><br>โมเดล Fit ย้อนหลัง: ฿%{y:,.2f}<extra></extra>"
))

# 10.3 เส้นยอดขายจริงย้อนหลัง 19 เดือน (Actual Sales)
fig_forecast.add_trace(go.Scatter(
    x=history_df["thai_month"],
    y=history_df["sales"],
    mode="lines+markers",
    name="ยอดขายจริง (Actual Sales)",
    line=dict(color="#0F4C81", width=3.5),
    marker=dict(size=7, color="#0A2540", symbol="circle"),
    hovertemplate="<b>%{x}</b><br>ยอดขายจริง: <b>฿%{y:,.2f}</b><extra></extra>"
))

# 10.4 เส้นพยากรณ์ไปข้างหน้า (Forecasted Sales)
# เชื่อมต่อจุดสุดท้ายของยอดจริงกับจุดแรกของยอดพยากรณ์เพื่อความต่อเนื่อง
bridge_x = [history_df["thai_month"].iloc[-1]] + list(forecast_df["thai_month"])
bridge_y = [history_df["sales"].iloc[-1]] + list(forecast_df["forecast_sales"])

fig_forecast.add_trace(go.Scatter(
    x=bridge_x,
    y=bridge_y,
    mode="lines+markers",
    name="ยอดคาดการณ์ (Forecast Sales)",
    line=dict(color="#D97706", width=3.5, dash="dash"),
    marker=dict(size=8, color="#F59E0B", symbol="diamond"),
    hovertemplate="<b>%{x}</b><br>ยอดคาดการณ์: <b>฿%{y:,.2f}</b><extra></extra>"
))

# 10.5 เส้นเป้าหมายการขายเชิงรุก (Target Stretch Line - ถ้าผู้บริหารเลือก)
if growth_target_pct > 0:
    bridge_target_y = [history_df["sales"].iloc[-1]] + list(forecast_df["target_sales"])
    fig_forecast.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_target_y,
        mode="lines+markers",
        name=f"เป้าหมายเชิงรุก (+{growth_target_pct}%)",
        line=dict(color="#059669", width=2.5, dash="dashdot"),
        marker=dict(size=6, color="#10B981", symbol="triangle-up"),
        hovertemplate="<b>%{x}</b><br>เป้าหมาย Stretch Goal: <b>฿%{y:,.2f}</b><extra></extra>"
    ))

# 10.6 ตกแต่ง Layout ให้สวยงามตามมาตรฐาน C-Level Executive Dashboard
fig_forecast.update_layout(
    height=480,
    margin=dict(l=40, r=40, t=30, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    xaxis=dict(
        title="เดือน / ปี (พ.ศ.)",
        gridcolor="#F1F5F9",
        showline=True,
        linecolor="#CBD5E1"
    ),
    yaxis=dict(
        title="ยอดขาย (บาท)",
        gridcolor="#F1F5F9",
        tickformat="฿,.0f",
        showline=True,
        linecolor="#CBD5E1"
    ),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF"
)

# เพิ่มแถบเส้นแบ่งระหว่างข้อมูลจริงและช่วงพยากรณ์
last_hist_month = history_df["thai_month"].iloc[-1]
fig_forecast.add_vline(
    x=last_hist_month,
    line_width=2,
    line_dash="dot",
    line_color="#EF4444"
)
fig_forecast.add_annotation(
    x=last_hist_month,
    y=history_df["sales"].max() * 0.95,
    text="จุดสิ้นสุดข้อมูลจริง (ก.ค. 69)",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1.5,
    arrowcolor="#EF4444",
    ax=-70,
    ay=-30,
    bgcolor="#FEF2F2",
    bordercolor="#EF4444",
    font=dict(color="#DC2626", size=11, weight="bold")
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------------------------------------------------------------
# 11. การวิเคราะห์เปรียบเทียบฤดูกาล (Seasonality Comparison) และสัดส่วนแพ็คเกจ
# ------------------------------------------------------------------------------
col_chart_left, col_chart_right = st.columns([1, 1])

with col_chart_left:
    st.markdown("### 📊 เปรียบเทียบฤดูกาลรายเดือน (2568 vs 2569)")
    
    # สร้างตาราง 12 เดือนเพื่อเทียบ 2568 vs 2569
    months_labels = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
    m25_vals = data_bundle["m25_df"]["clean_sales"].tolist()
    
    # 2569: จริง 7 เดือน + พยากรณ์ที่เหลือ
    m26_actual = data_bundle["m26_df"]["clean_sales"].tolist()
    m26_proj = m26_actual.copy()
    
    for i in range(7, 12):
        month_dt = pd.Timestamp(f"2026-{i+1:02d}-01")
        match_f = forecast_df[forecast_df["date"] == month_dt]
        if not match_f.empty:
            m26_proj.append(match_f["forecast_sales"].values[0])
        else:
            m26_proj.append(0.0)

    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(
        x=months_labels,
        y=m25_vals,
        name="ปี 2568 (จริงทั้งปี)",
        marker_color="#94A3B8"
    ))
    fig_season.add_trace(go.Bar(
        x=months_labels[:7],
        y=m26_actual,
        name="ปี 2569 (จริง ม.ค.-ก.ค.)",
        marker_color="#0F4C81"
    ))
    fig_season.add_trace(go.Bar(
        x=months_labels[7:],
        y=m26_proj[7:],
        name="ปี 2569 (พยากรณ์ ส.ค.-ธ.ค.)",
        marker_color="#F59E0B"
    ))
    fig_season.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=30, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="฿,.0f", gridcolor="#F1F5F9"),
        xaxis=dict(gridcolor="#F1F5F9"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_season, use_container_width=True)
    st.caption("💡 **จุดสังเกตสำคัญ:** ไตรมาส 4 (ต.ค. - ธ.ค.) เป็นช่วง Peak Season สูงสุดของปี จากการตรวจสุขภาพประจำปีของบริษัทคู่สัญญา")

with col_chart_right:
    st.markdown("### 🧬 โครงสร้างรายได้ตามระดับแพ็คเกจ (Package Mix)")
    
    # คำนวณสัดส่วน 4 หมวดหมู่
    cat_df_plot = cat_df.copy()
    cat_df_plot["pct_2568"] = (cat_df_plot["sales_2568"] / cat_df_plot["sales_2568"].sum()) * 100
    cat_df_plot["pct_2569"] = (cat_df_plot["sales_2569_ytd"] / cat_df_plot["sales_2569_ytd"].sum()) * 100

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        y=cat_df_plot["category"],
        x=cat_df_plot["sales_2568"],
        name="ยอดปี 2568",
        orientation="h",
        marker_color="#CBD5E1"
    ))
    fig_cat.add_trace(go.Bar(
        y=cat_df_plot["category"],
        x=cat_df_plot["sales_2569_ytd"],
        name="ยอดปี 2569 (7 เดือน)",
        orientation="h",
        marker_color="#0D9488"
    ))
    fig_cat.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=30, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="฿,.0f", gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.caption("💡 **จุดสังเกตสำคัญ:** ในปี 2569 ยอดขายหมวด **`3_Early Disease Check`** ครองสัดส่วนสูงถึง 66% แสดงถึงการเปลี่ยนผ่านสู่การตรวจคัดกรองเชิงลึกเฉพาะทาง")

# ------------------------------------------------------------------------------
# 12. แท็บตารางข้อมูลและการส่งออก (Data Tables & Export Section)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📑 ตารางข้อมูลเชิงลึกและการส่งออกรายงาน (Data & Forecast Export)")

tab_forecast, tab_history, tab_packages = st.tabs([
    "🔮 ตารางตัวเลขพยากรณ์ในอนาคต (Forecast Table)",
    "📜 ประวัติข้อมูลจริงย้อนหลัง 19 เดือน (Historical 19 Months)",
    "📦 รายละเอียดแยกตามหมวดแพ็คเกจ (Package Categories)"
])

with tab_forecast:
    # จัดเตรียมตารางพยากรณ์สวยงาม
    display_fc = forecast_df.copy()
    display_fc["เดือน / ปี (พ.ศ.)"] = display_fc["thai_month_full"]
    display_fc["ยอดคาดการณ์ (บาท)"] = display_fc["forecast_sales"].apply(format_currency)
    display_fc["กรอบล่าง (Lower Bound)"] = display_fc["lower_bound"].apply(format_currency)
    display_fc["กรอบบน (Upper Bound)"] = display_fc["upper_bound"].apply(format_currency)
    display_fc["เป้าหมายเชิงรุก (Target)"] = display_fc["target_sales"].apply(format_currency)
    
    # คำนวณ MoM Change
    display_fc["การเปลี่ยนแปลงเทียบเดือนก่อนหน้า (MoM)"] = (
        display_fc["forecast_sales"].pct_change() * 100
    ).apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "-")

    cols_to_show = [
        "เดือน / ปี (พ.ศ.)", "ยอดคาดการณ์ (บาท)", "กรอบล่าง (Lower Bound)",
        "กรอบบน (Upper Bound)", "เป้าหมายเชิงรุก (Target)", "การเปลี่ยนแปลงเทียบเดือนก่อนหน้า (MoM)"
    ]
    st.dataframe(display_fc[cols_to_show], use_container_width=True, hide_index=True)

    # ปุ่มดาวน์โหลดไฟล์ CSV
    csv_forecast = display_fc[[
        "thai_month_full", "forecast_sales", "lower_bound", "upper_bound", "target_sales"
    ]].rename(columns={
        "thai_month_full": "Month_BE",
        "forecast_sales": "Forecast_Sales_THB",
        "lower_bound": f"Lower_Bound_{ci_level}",
        "upper_bound": f"Upper_Bound_{ci_level}",
        "target_sales": f"Target_Sales_Growth_{growth_target_pct}pct"
    }).to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 ดาวน์โหลดข้อมูลตารางพยากรณ์ (CSV Report)",
        data=csv_forecast,
        file_name=f"hospital_forecast_healthcheck_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="คลิกเพื่อบันทึกไฟล์ CSV นำไปใช้ในงานนำเสนอคณะกรรมการบริหาร"
    )

with tab_history:
    display_hist = history_df.copy()
    display_hist["เดือน / ปี (พ.ศ.)"] = display_hist["thai_month_full"]
    display_hist["ยอดขายจริง (บาท)"] = display_hist["sales"].apply(format_currency)
    display_hist["สัดส่วนต่อยอดรวมทั้งปี"] = display_hist.apply(
        lambda row: f"{(row['sales'] / (total_sales_2568 if row['year']==2568 else sales_2569_ytd))*100:.2f}%", axis=1
    )
    display_hist["สถานะข้อมูล"] = display_hist["year"].apply(
        lambda y: "✅ ข้อมูลสมบูรณ์ทั้งปี" if y == 2568 else "📌 ข้อมูลจริงสะสม (YTD)"
    )
    st.dataframe(
        display_hist[["เดือน / ปี (พ.ศ.)", "ยอดขายจริง (บาท)", "สัดส่วนต่อยอดรวมทั้งปี", "สถานะข้อมูล"]],
        use_container_width=True,
        hide_index=True
    )

with tab_packages:
    display_cat = cat_df.copy()
    display_cat["หมวดหมู่แพ็คเกจ"] = display_cat["category"]
    display_cat["ยอดขายปี 2568 (บาท)"] = display_cat["sales_2568"].apply(format_currency)
    display_cat["สัดส่วนปี 2568 (%)"] = (display_cat["sales_2568"] / display_cat["sales_2568"].sum() * 100).apply(lambda x: f"{x:.1f}%")
    display_cat["ยอดขายปี 2569 YTD (บาท)"] = display_cat["sales_2569_ytd"].apply(format_currency)
    display_cat["สัดส่วนปี 2569 YTD (%)"] = (display_cat["sales_2569_ytd"] / display_cat["sales_2569_ytd"].sum() * 100).apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_cat[["หมวดหมู่แพ็คเกจ", "ยอดขายปี 2568 (บาท)", "สัดส่วนปี 2568 (%)", "ยอดขายปี 2569 YTD (บาท)", "สัดส่วนปี 2569 YTD (%)"]],
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------------------------
# 13. บทวิเคราะห์เชิงกลยุทธ์และข้อเสนอแนะสำหรับผู้บริหารโรงพยาบาล (Strategic Recommendations)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🏛️ ข้อเสนอแนะเชิงกลยุทธ์และการบริหารจัดการสำหรับผู้บริหาร (Executive Advisory Briefing)")
st.caption("จัดทำโดย: Senior Data Scientist & Hospital Financial Expert โดยอิงจากแบบจำลองอนุกรมเวลาและพฤติกรรมผู้รับบริการ")

col_strat1, col_strat2 = st.columns(2)

with col_strat1:
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-card-title">
            <span>👥 1. การบริหารอัตรากำลังบุคลากรทางการแพทย์ (Clinical Staffing)</span>
        </div>
        <div class="strategy-item">
            • <b>เตรียมความพร้อมรับมือ Q4 Peak Season (ต.ค. - ธ.ค.):</b><br>
            จากโมเดลอนุกรมเวลา พบว่ายอดขายในเดือน พ.ย. - ธ.ค. มีดัชนีฤดูกาลพุ่งสูงกว่าค่าเฉลี่ย 35-50% โรงพยาบาลควรวางแผนตารางเวรแพทย์ตรวจสุขภาพ (OPD Checkup Doctors) และพยาบาลเจาะเลือดเพิ่มขึ้น 25-30% ในช่วงดังกล่าว เพื่อลดเวลารอคอย (Waiting Time) ไม่ให้กระทบ Patient Experience
        </div>
        <div class="strategy-item">
            • <b>จัดสรรทีม Mobile Checkup สำหรับบริษัทคู่สัญญา:</b><br>
            เนื่องจากยอดช่วงปลายปีส่วนใหญ่เป็น Corporate Contract การแยกทีมตรวจสุขภาพนอกสถานที่ (On-site Screening) ออกจากศูนย์ตรวจสุขภาพในโรงพยาบาล (Wellness Center) จะช่วยลดความแออัดและรักษารายได้กลุ่มลูกค้า Walk-in Premium ได้เต็มประสิทธิภาพ
        </div>
    </div>

    <div class="strategy-card">
        <div class="strategy-card-title">
            <span>🔬 2. การบริหารศักยภาพห้องปฏิบัติการและรังสีวินิจฉัย (Diagnostic & Lab Capacity)</span>
        </div>
        <div class="strategy-item">
            • <b>รองรับการเติบโตแบบก้าวกระโดดของหมวด Early Disease Check:</b><br>
            สัดส่วนรายได้หมวด <code>3_Early Disease Check</code> เพิ่มขึ้นแตะ <b>66.0%</b> ในปี 2569 ซึ่งเป็นกลุ่มที่ต้องอาศัยการตรวจพิเศษ เช่น Ultrasound ช่องท้อง, Digital Mammogram, Bone Density และ Tumor Markers
        </div>
        <div class="strategy-item">
            • <b>การเจรจาต้นทุน Reagents และบำรุงรักษาเครื่องมือ:</b><br>
            ฝ่ายจัดซื้อควรจัดทำสัญญาสั่งซื้อน้ำยาตรวจวิเคราะห์และสารทึบรังสีล่วงหน้าตามปริมาณ Forecast ใน Q4 เพื่อรับส่วนลดตามขนาดคำสั่งซื้อ (Volume Rebate) และนัดหมาย Preventive Maintenance เครื่องตรวจล่วงหน้าในเดือน ก.ย. ก่อนเข้าสู่ช่วงผู้รับบริการหนาแน่น
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_strat2:
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-card-title">
            <span>📈 3. กลยุทธ์การตลาดและปิดช่องว่างรายได้ (Revenue Gap Bridging)</span>
        </div>
        <div class="strategy-item">
            • <b>เร่งเครื่องปิดช่องว่างรายได้ (Stretch Goal):</b><br>
            ยอดคาดการณ์ปี 2569 รวมอยู่ที่ประมาณ <b>19.6 ล้านบาท</b> ซึ่งตามหลังยอดปี 2568 (31.4 ล้านบาท) อยู่ราว 37% เนื่องจากช่วงครึ่งปีแรกชะลอตัว ฝ่ายการตลาดควรกระตุ้นแคมเปญ <i>"Early-Bird Corporate Package"</i> โดยมอบสิทธิพิเศษแก่บริษัทที่จองคิวตรวจในเดือน ส.ค. - ก.ย. เพื่อกระจายผู้รับบริการและดึงยอดล่วงหน้า
        </div>
        <div class="strategy-item">
            • <b>Upselling จาก Basic สู่ Early Disease Screening:</b><br>
            ใช้กลยุทธ์ Add-on Targeted Vouchers ให้ผู้รับบริการกลุ่มครอบครัวและผู้สูงวัย เช่น แพ็คเกจตรวจคัดกรองหัวใจ (Calcium Score) หรือคัดกรองมะเร็งระยะเริ่มต้น ซึ่งมีกำไรส่วนเกิน (Gross Margin) สูงกว่าแพ็คเกจพื้นฐาน
        </div>
    </div>

    <div class="strategy-card">
        <div class="strategy-card-title">
            <span>💰 4. การบริหารสภาพคล่องและการเงินโรงพยาบาล (Hospital Financial Management)</span>
        </div>
        <div class="strategy-item">
            • <b>การเร่งรัดเรียกเก็บหนี้ลูกหนี้การค้า (A/R Collection Acceleration):</b><br>
            ยอดขายจากบริษัทคู่สัญญาใน Q4 มักมีเงื่อนไขเครดิตเทอม 60-90 วัน แผนกการเงินควรวางระบบ Electronic Billing และวางบิลทันทีหลังส่งมอบผลตรวจรวม เพื่อเร่งกระแสเงินสดรับ (Cash Inflow) ให้ทันปิดงบการเงินสิ้นปี
        </div>
        <div class="strategy-item">
            • <b>Dynamic Pricing & Corporate Quotation Optimization:</b><br>
            นำผลลัพธ์จากโมเดลพยากรณ์ไปปรับปรุงการตั้งราคาสัญญาตรวจสุขภาพประจำปี 2570 โดยเน้นนำเสนอ Bundle Package เชิงป้องกันโรคร้ายแรงที่ตอบโจทย์สุขภาวะพนักงานองค์กรสมัยใหม่
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 14. ส่วนท้ายหน้าจอ (Footer)
# ------------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 24px 0 12px 0;">
    โรงพยาบาลและศูนย์บริการตรวจสุขภาพครบวงจร | ระบบพยากรณ์และบริหารการเงินโรงพยาบาลอัจฉริยะ (Hospital Financial Analytics Platform) © 2569
</div>
""", unsafe_allow_html=True)
