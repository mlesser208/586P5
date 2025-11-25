"""Entrypoint for Streamlit Cloud deployments.

Streamlit Cloud looks for `streamlit_app.py` by default, so this
module simply imports the main app defined in `streamlit_map.py`.
"""

# Importing the module executes the Streamlit app defined there.
from streamlit_map import *  # noqa: F401,F403
