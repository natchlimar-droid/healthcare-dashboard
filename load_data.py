"""
Data loader helper module for backward compatibility.
Delegates to modules.data_loader.
"""

from modules.data_loader import load_data, process_patient_data, build_summary_pts

__all__ = ["load_data", "process_patient_data", "build_summary_pts"]