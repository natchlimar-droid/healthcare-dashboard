"""
Design System, Custom CSS Styles, and Safe HTML Rendering Utilities
"""

import streamlit as st
from modules.config import INK, BG, SURFACE, TEAL, SAGE, AMBER, RED, MUTED


def render_custom_html(html_str: str) -> None:
    """
    Renders HTML safely without CommonMark/Markdown turning indented lines
    into <pre><code> blocks.
    """
    clean_str = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_str, unsafe_allow_html=True)


def apply_custom_styles() -> None:
    """
    Injects custom typography, responsive card containers, glassmorphism,
    and micro-interaction CSS into the Streamlit application.
    """
    css_content = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {INK};
    }}

    .stApp {{
        background-color: {BG};
    }}

    .block-container {{
        padding-top: 1.1rem !important;
        padding-bottom: 2.2rem !important;
        max-width: 98% !important;
    }}

    /* Metrics Styling */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(11, 27, 43, 0.05);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(11, 27, 43, 0.08);
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.55rem !important;
        font-weight: 700;
        color: {INK};
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.78rem !important;
        font-weight: 600;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}

    /* Card Panels */
    .panel-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {INK};
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .panel-sub {{
        font-size: 0.76rem;
        color: {MUTED};
        margin-bottom: 12px;
    }}

    /* Pulse Health Score Card */
    .pulse-card {{
        background: {SURFACE};
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 10px rgba(11, 27, 43, 0.05);
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .pulse-dot {{
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 1.15rem;
        color: white;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }}
    .pulse-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    .pulse-status {{
        font-size: 1.05rem;
        font-weight: 700;
    }}

    /* Meter Progress Bar */
    .meter-wrap {{
        background: {SURFACE};
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
    }}
    .meter-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .meter-track {{
        width: 100%;
        height: 8px;
        background: #E2E8F0;
        border-radius: 4px;
        overflow: hidden;
    }}
    .meter-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }}

    /* Drilldown Banner */
    .drill-banner {{
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 8px;
        margin-top: 8px;
        display: inline-block;
    }}

    /* Headers */
    .header-title {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {INK};
        letter-spacing: -0.4px;
        margin-bottom: 2px;
    }}
    .header-sub {{
        font-size: 0.85rem;
        color: {MUTED};
    }}

    /* Health Package Cards */
    .pkg-card-box {{
        background: {SURFACE};
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        padding: 20px;
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }}
    .pkg-card-box:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }}
    .pkg-pill-match {{
        background: linear-gradient(135deg, #0E5C56, #16A34A);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(14,92,86,0.3);
    }}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)
