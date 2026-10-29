from config import CATEGORY_EN, CATEGORY_TE
from ui.components import esc, page_title
from ui.i18n import current_lang, tr
from ui import voice_flow as vf
from asr import MAX_AUDIO_BYTES, accepted_audio_types
from ui.voice_flow import is_busy, request_asr, request_retry
from ui.state import record as state_record
import html as _html


def _reopen(st):
    chosen = st.session_state.get("history_choice")
    match = next((e for e in st.session_state.history if e["id"] == chosen), None)
    if match:
        st.session_state.last_result = match["result"]
        st.session_state.nav = "ask"


def _voice_followup(st, assistant, busy):
    ss = st.session_state
    asr_ok = assistant.asr.available()
    types = accepted_audio_types()
    recorded = st.audio_input(tr(st, "history.followup.record"), key="vf_hist_rec", disabled=busy) if hasattr(st, "audio_input") else None
    uploaded = st.file_uploader(tr(st, "ask.upload"), type=types, key="vf_hist_upload", disabled=busy)
    st.caption(tr(st, "ask.upload_help", types=", ".join(types), mb=MAX_AUDIO_BYTES // (1024 * 1024)))
    incoming = recorded or uploaded
    if incoming is not None and asr_ok:
        data = incoming.getvalue()
        name = getattr(incoming, "name", None) or "recording.wav"
        if vf.receive_audio(ss, data, name):
            vf.request_asr(ss)
    audio = ss.get(vf.K["audio"])
    if audio:
        st.markdown(f'<div class="tca-muted">{esc(tr(st, "ask.audio_received", name=audio["name"]))}</div>',
                     unsafe_allow_html=True)
        st.audio(audio["bytes"])


def render(st, assistant):
    """Render the history page: list of past conversations and voice follow-up."""
    page_title(st, "history.title", "history.intro")
    history = st.session_state.get("history", [])
    if not history:
        st.caption(tr(st, "history.empty"))
        st.caption(tr(st, "history.empty_help"))
        if st.button(tr(st, "history.clear"), type="secondary", use_container_width=True, on_click=_clear_history, args=(st,)):
            pass
        return

    for entry in history:
        result = entry.get("result") or {}
        answer = result.get("answer") or {}
        assistant_answer = answer.get("answer") or ""
        status = answer.get("status")
        sources = answer.get("sources") or []
        query = entry.get("query") or ""

        with st.expander(f'{esc(tr(st, "history.query", query=query))}  ·  {entry.get("timestamp", "")}', expanded=False):
            st.caption(esc(assistant_answer if assistant_answer else "(no answer)"))
            if status:
                st.caption(tr(st, f"history.status.{status}", status=status))
            if sources:
                for s in sources:
                    st.caption(esc(s))

            ids = [e["id"] for e in history]
            st.selectbox(tr(st, "history.reopen"), ids, key="history_choice", format_func=lambda i: f"#{i}", on_change=_reopen, args=(st,))
            busy = is_busy(st.session_state)
            _voice_followup(st, assistant, busy)
            if st.button(tr(st, "history.clear"), type="secondary", use_container_width=True, on_click=_clear_history, args=(st,)):
                pass


def _clear_history(st):
    st.session_state.history = []
    st.session_state.last_result = None
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    st.session_state.pop("history_choice", None)
