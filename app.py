import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time

# optional dependencies
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


# ============================================================
# Design tokens
# ============================================================
INK = "#0B1B2B"

# ... constants อื่น ๆ ...


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    # โค้ด load_data ที่แก้ให้รองรับ ZIP
    ...
