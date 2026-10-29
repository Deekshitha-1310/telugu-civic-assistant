"""
Streamlit app for the Dialect & Noise-Robust Telugu Civic Assistant.

Run from the repository root:
    streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from pipeline import CivicAssistant
from ui.components import header
from ui.i18n import t
from ui.pages import about, ask, dashboard, evaluation, history, schemes
from ui.state import init
from ui.theme import inject

PAGES = {
    "dashboard": dashboard.render,
    "ask": ask.render,
    "schemes": schemes.render,
    "history": history.render,
    "evaluation": evaluation.render,
    "about": about.render,
}


@st.cache_resource(show_spinner=False)
def get_assistant():
    return CivicAssistant()


def main():
    st.set_page_config(
        page_title="Telugu Civic Assistant",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject(st)
    init(st)
    header(st)

    with st.spinner(t("app.loading", st.session_state.get("ui_lang", "te"))):
        assistant = get_assistant()
    # Advanced settings (Evaluation page) genuinely change retrieval.
    assistant.retriever.bm25_weight = st.session_state.bm25_weight
    assistant.retriever.dense_weight = st.session_state.dense_weight
    assistant.retriever.suffix_stripping = st.session_state.suffix_stripping
    PAGES[st.session_state.nav](st, assistant)


if __name__ == "__main__":
    main()
