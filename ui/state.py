"""Session-state defaults shared by all pages."""
import json
import os
from datetime import datetime
from pathlib import Path

from config import RETRIEVAL_DEFAULTS
from ui import voice_flow
from ui.components import PAGE_IDS
from ui.i18n import DEFAULT_LANG

# Persistent history file. Saved to data/history.json so conversations survive
# browser refreshes. The file is NOT sensitive (no auth tokens, no utterances
# beyond the question text), and the app runs locally on each user's machine,
# so this is a demo-level persistence mechanism.
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "history.json"


def _load_history():
    """Load historical queries from disk. Returns [] if the file does not exist."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(entries):
    """Write the history list back to disk (atomic write).

    Leaves existing entries untouched: only this process appends to the list.
    This is not a production-grade persistence layer — it is sufficient for a
    single-user local app where the file is not sensitive. In production,
    a database or a write-through endpoint would be used.
    """
    try:
        tmp = HISTORY_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except OSError:
        # Never fail the app on a storage error.
        pass

# Old page labels (before stable page ids) -> ids, so stale sessions keep working.
_LEGACY_NAV = {
    "Dashboard": "dashboard", "Ask the Assistant": "ask", "Government Schemes": "schemes",
    "Query History": "history", "Evaluation": "evaluation", "About": "about",
}


def init(st):
    ss = st.session_state
    # Load history from disk once. Streamlit runs the script on every rerun,
    # so we must guard the load with ``setdefault``: the list is in session
    # state after the first run and never read from disk again.
    ss.setdefault("history", _load_history())
    ss.setdefault("last_result", None)
    ss.setdefault("ask_text", "")
    ss.setdefault("nav", "dashboard")
    if ss.nav not in PAGE_IDS:
        ss.nav = _LEGACY_NAV.get(ss.nav, "dashboard")
    ss.setdefault("ui_lang", DEFAULT_LANG)
    ss.setdefault("answer_language", "te")
    ss.setdefault("top_k", RETRIEVAL_DEFAULTS["top_k"])
    ss.setdefault("bm25_weight", RETRIEVAL_DEFAULTS["bm25_weight"])
    ss.setdefault("dense_weight", RETRIEVAL_DEFAULTS["dense_weight"])
    ss.setdefault("suffix_stripping", RETRIEVAL_DEFAULTS["suffix_stripping"])
    voice_flow.init(ss)


def record(st, result):
    if not result or not result.get("ok"):
        return
    answer = result.get("answer") or {}
    entry = {
        "id": len(st.session_state.history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": result.get("original_query"),
        "input_mode": result.get("input_mode"),
        "intent": result.get("intent"),
        "category": result.get("question_category"),
        "confidence": result.get("confidence"),
        "answer_status": answer.get("status"),
        "sources": [s.get("scheme") for s in answer.get("sources") or []],
        "result": result,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.last_result = result
    # Persist in the background (non-blocking). The file write is atomic and
    # failures are swallowed so the app never breaks.
    _save_history(st.session_state.history)
