"""
Streamlit Cloud Entrypoint Wrapper.
Ensures seamless deployment whether the file path is set to 'streamlit_app.py' or 'app/streamlit_app.py'.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main application module
import runpy

app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
