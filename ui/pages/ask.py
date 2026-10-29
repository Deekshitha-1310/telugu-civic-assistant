"""Ask page: voice / upload / text input, honest progress, answer, sources, follow-up.

Order of a run (important for Streamlit):
  1. input widgets (disabled while an operation is pending/running)
  2. ``voice_flow.run_pending`` executes at most one requested operation
  3. transcript editor (its value is set by step 2 before it is rendered)
  4. error + retry, then the answer, sources, follow-up, technical details
"""
from config import EXAMPLE_QUERIES, INTENT_EN
from asr import MAX_AUDIO_BYTES, accepted_audio_types
from ui import voice_flow as vf
from ui.components import esc, notice, page_title, section, source_card, stage_tracker
from ui.i18n import current_lang, tr
from ui.state import record


# ---------------------------------------------------------------------------
# Operations (called by voice_flow.run_pending)
# ---------------------------------------------------------------------------

def _answer_fn(st, assistant):
    def run(text, mode, transcript, progress):
        last = st.session_state.get("last_result")
        prev = (last.get("schemes_mentioned") or []) if last else None
        result = assistant.run(
            text,
            input_mode=mode,
            transcript=transcript,
            previous_schemes=prev,
            top_k=st.session_state.get("top_k", 5),
            conversation_context=[
                {"query": h.get("query"),
                 "answer": (h.get("result") or {}).get("answer", {}).get("text", "")}
                for h in st.session_state.get("history", [])[:3]
            ],
            answer_language=st.session_state.get("answer_language", "te"),
            progress=progress,
        )
        if result.get("ok"):
            record(st, result)
        return result
    return run


# ---------------------------------------------------------------------------
# Callbacks (run before the script body on the next rerun)
# ---------------------------------------------------------------------------

def _submit_text(st):
    text = (st.session_state.get("ask_text") or "").strip()
    if not text:
        st.session_state["ask_empty"] = True
        return
    st.session_state["ask_empty"] = False
    vf.request_answer(st.session_state, text, mode="text")


def _submit_example(st, question):
    st.session_state["ask_text"] = question
    vf.request_answer(st.session_state, question, mode="text")


def _submit_transcript(st):
    text = (st.session_state.get(vf.K["edit"]) or "").strip()
    if not text:
        st.session_state["transcript_empty"] = True
        return
    st.session_state["transcript_empty"] = False
    vf.request_answer(st.session_state, text, mode="audio", transcript=text)


def _submit_followup(st):
    text = (st.session_state.get("followup_box") or "").strip()
    if text and vf.request_answer(st.session_state, text, mode="text"):
        st.session_state["followup_box"] = ""


def _voice_followup(st, assistant, busy):
    ss = st.session_state
    asr_ok = assistant.asr.available()
    if not asr_ok:
        return
    types = accepted_audio_types()
    recorded = st.audio_input(tr(st, "followup.record"), key="vf_followup_rec", disabled=busy) if hasattr(st, "audio_input") else None
    uploaded = st.file_uploader(tr(st, "ask.upload"), type=types, key="vf_followup_upload", disabled=busy)
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
        c1, c2 = st.columns(2)
        c1.button(tr(st, "ask.transcribe"), key="vf_followup_transcribe", disabled=busy or not asr_ok,
                  on_click=lambda: vf.request_asr(ss, force=True), use_container_width=True)
        c2.button(tr(st, "ask.clear_audio"), key="vf_followup_clear", disabled=busy,
                  on_click=lambda: vf.clear_audio(ss), use_container_width=True)


def _clear_conversation(st):
    st.session_state.history = []
    st.session_state.last_result = None
    vf.clear_audio(st.session_state)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _voice_panel(st, assistant, busy):
    ss = st.session_state
    asr_ok = assistant.asr.available()
    section(st, "ask.voice_title")
    if not asr_ok:
        notice(st, tr(st, "ask.asr_unavailable", reason=assistant.asr.error or "—"), "warn")

    types = accepted_audio_types()
    # Voice input: always offer the recorder button (older Streamlit may hide
    # st.audio_input, but the ASR backend is backend-independent). Keeping the
    # button visible avoids showing text-only input where voice is possible.
    recorded = st.audio_input(tr(st, "ask.record"), key="vf_recorder", disabled=busy or not asr_ok) if hasattr(st, "audio_input") else None
    uploaded = st.file_uploader(
        tr(st, "ask.upload"), type=types, key="vf_uploader", disabled=busy or not asr_ok,
        help=tr(st, "ask.upload_help", types=", ".join(types), mb=MAX_AUDIO_BYTES // (1024 * 1024)),
    )
    st.caption(tr(st, "error.upload_lost"))

    incoming = recorded or uploaded
    if incoming is not None and asr_ok:
        data = incoming.getvalue()
        name = getattr(incoming, "name", None) or "recording.wav"
        if vf.receive_audio(ss, data, name):
            vf.request_asr(ss)  # continue straight to transcription once the file has arrived

    audio = ss.get(vf.K["audio"])
    if audio:
        st.markdown(f'<div class="tca-muted">{esc(tr(st, "ask.audio_received", name=audio["name"]))}</div>',
                    unsafe_allow_html=True)
        st.audio(audio["bytes"])
        c1, c2 = st.columns(2)
        c1.button(tr(st, "ask.transcribe"), key="vf_transcribe_btn", disabled=busy or not asr_ok,
                  on_click=lambda: vf.request_asr(ss, force=True), use_container_width=True)
        c2.button(tr(st, "ask.clear_audio"), key="vf_clear_btn", disabled=busy,
                  on_click=lambda: vf.clear_audio(ss), use_container_width=True)


def _text_panel(st, busy):
    section(st, "ask.or_type")
    st.text_area(tr(st, "ask.text_label"), key="ask_text", height=110,
                 placeholder=tr(st, "ask.text_placeholder"), disabled=busy)
    st.button(tr(st, "ask.submit"), type="primary", key="ask_submit", disabled=busy,
              on_click=_submit_text, args=(st,), use_container_width=True)
    if st.session_state.get("ask_empty"):
        st.error(tr(st, "ask.question_empty"))
    st.markdown(f'<div class="tca-muted" style="margin-top:.8rem">{esc(tr(st, "ask.examples"))}</div>',
                unsafe_allow_html=True)
    for i, q in enumerate(EXAMPLE_QUERIES[:4]):
        st.button(q, key=f"example_{i}", disabled=busy, on_click=_submit_example, args=(st, q),
                  use_container_width=True)


def _run_pending(st, assistant):
    ss = st.session_state
    pending = ss.get(vf.K["pending"])
    if not pending:
        return
    stages = ["audio", "asr"] if pending["op"] == "asr" else (
        (["audio", "asr"] if pending.get("mode") == "audio" else []) + ["analysis", "retrieval", "answer"]
    )
    placeholder = st.empty()

    def on_stage(stage):
        with placeholder.container():
            stage_tracker(st, stages, ss.get(vf.K["completed"]) or [], active=stage)
            st.markdown(f'<div class="tca-muted">{esc(tr(st, "progress." + stage))}</div>',
                        unsafe_allow_html=True)

    with st.spinner(tr(st, "progress.working")):
        outcome = vf.run_pending(ss, assistant.asr.transcribe, _answer_fn(st, assistant),
                                 on_stage=on_stage)
    if outcome and outcome[0] == "asr" and outcome[1] and outcome[1].get("ok"):
        ss["_asr_model"] = outcome[1].get("backend")
    placeholder.empty()


def _flash(st):
    key = st.session_state.get(vf.K["flash"])
    if key:
        st.session_state[vf.K["flash"]] = None
        if hasattr(st, "toast"):
            st.toast(tr(st, key))
        else:
            st.success(tr(st, key))


def _transcript_editor(st, busy):
    ss = st.session_state
    if ss.get(vf.K["transcript"]) is None:
        return
    with st.container(border=True):
        stage_tracker(st, ["audio", "asr"], ["audio", "asr"])
        model = ss.get("_asr_model")
        st.text_area(tr(st, "ask.transcript"), key=vf.K["edit"], height=90, disabled=busy)
        if model:
            st.caption(tr(st, "asr.model_in_use", model=model))
        st.button(tr(st, "ask.submit_transcript"), type="primary", key="vf_submit_transcript",
                  disabled=busy, on_click=_submit_transcript, args=(st,))
        if ss.get("transcript_empty"):
            st.error(tr(st, "ask.transcript_empty"))


def _error_panel(st, busy):
    ss = st.session_state
    if ss.get(vf.K["stage"]) != "error" or not ss.get(vf.K["error"]):
        return
    err = ss[vf.K["error"]]
    failed = ss.get(vf.K["failed"])
    code = err.get("code") or "answer_error"
    try:
        message = tr(st, f"error.{code}", detail=err.get("detail") or "—",
                     mb=MAX_AUDIO_BYTES // (1024 * 1024))
    except KeyError:
        message = tr(st, "error.answer_error", detail=err.get("detail") or code)
    with st.container(border=True):
        stages = ["audio", "asr"] if failed == "asr" else ["analysis", "retrieval", "answer"]
        stage_tracker(st, stages, ss.get(vf.K["completed"]) or [],
                      failed=("asr" if failed == "asr" else None))
        st.error(message)
        can_retry = (failed == "asr" and code not in ("invalid_audio", "audio_too_large", "asr_unavailable")) \
            or (failed == "answer" and ss.get(vf.K["question"]))
        if failed == "asr" and ss.get(vf.K["audio"]):
            st.caption(tr(st, "ask.retained_audio", name=ss[vf.K["audio"]]["name"]))
        if can_retry:
            st.button(tr(st, f"retry.{failed}"), key="vf_retry", disabled=busy,
                      on_click=lambda: vf.request_retry(ss))


def show_result(st, result):
    """Answer first, then notices, sources, follow-up and technical details."""
    answer = result.get("answer") or {}
    section(st, "answer.title")
    st.markdown(f'<div class="tca-question">{esc(tr(st, "answer.you_asked"))}: '
                f'{esc(result.get("original_query"))}</div>', unsafe_allow_html=True)
    if result.get("context_note") and result.get("scheme"):
        notice(st, tr(st, "context.followup", scheme=result["scheme"]), "info")
    if not answer.get("grounded"):
        notice(st, tr(st, "answer.no_evidence"), "info")
    elif answer.get("generation") == "llm":
        notice(st, tr(st, "answer.llm"), "info")
    if answer.get("llm_error"):
        notice(st, tr(st, "answer.llm_error"), "warn")
    st.markdown(f'<div class="tca-answer">{esc(answer.get("text", ""))}</div>', unsafe_allow_html=True)
    for code in answer.get("notices") or []:
        notice(st, tr(st, f"notice.{code}"), "warn" if code in ("historical", "legacy_unverified",
                                                                   "currentness_unverified") else "info")

    evidence = answer.get("evidence") or []
    if evidence:
        section(st, "sources.title")
        for item in evidence:
            source_card(st, item)
    hints = answer.get("source_hints") or []
    if not evidence and hints:
        section(st, "sources.hints")
        for h in hints:
            source_card(st, {
                "title": h.get("source_title") or h.get("scheme"), "url": h.get("source_url"),
                "authority": h.get("authority"), "date": h.get("published_date") or h.get("updated_date"),
                "verification_status": h.get("verification_status"),
                "source_classification": h.get("source_classification"),
            })


def _followup(st, assistant, busy):
    ss = st.session_state
    asr_ok = assistant.asr.available()
    with st.container(border=True):
        section(st, "followup.title")
        st.text_area(tr(st, "followup.text_label"), key="followup_box", height=70, disabled=busy)
        col1, col2 = st.columns([1, 1])
        with col1:
            if not asr_ok:
                st.caption(tr(st, "followup.record_off"))
            else:
                st.caption(tr(st, "followup.record_unsupported"))
        with col2:
            st.button(tr(st, "followup.submit"), key="followup_btn", disabled=busy,
                      on_click=_submit_followup, args=(st,))
        _voice_followup(st, assistant, busy)


def _technical(st, result):
    answer = result.get("answer") or {}
    with st.expander(tr(st, "tech.title"), expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{tr(st, 'tech.original')}**")
            st.write(result.get("original_query"))
            st.markdown(f"**{tr(st, 'tech.transcript')}**")
            st.write(result.get("transcript") or tr(st, "tech.no_transcript"))
        with c2:
            st.markdown(f"**{tr(st, 'tech.normalized')}**")
            st.write(result.get("normalized_query") or "—")
            edits = result.get("normalization_edits") or []
            if edits:
                st.markdown(f"**{tr(st, 'tech.edits')}**")
                st.dataframe(edits, hide_index=True, use_container_width=True)
            else:
                st.caption(tr(st, "tech.no_edits"))
        src = result.get("category_source") or "none"
        conf = result.get("confidence")
        rows = [
            (tr(st, "tech.category"), result.get("question_category") or "—"),
            (tr(st, "tech.category_source"), tr(st, f"catsrc.{src}")),
            (tr(st, "tech.intent"), INTENT_EN.get(result.get("intent"), result.get("intent") or "—")),
            (tr(st, "tech.confidence"), f"{conf:.2f}" if conf is not None else "—"),
            (tr(st, "tech.scheme"), result.get("scheme") or "—"),
            (tr(st, "tech.tier"), answer.get("evidence_tier") or "—"),
        ]
        st.table({"": [r[0] for r in rows], " ": [str(r[1]) for r in rows]})
        st.markdown(f"**{tr(st, 'tech.retrieved')}**")
        used = {p.get("doc_id") for p in answer.get("used_passages") or []}
        passages = result.get("passages") or []
        if not passages:
            st.caption(tr(st, "tech.none"))
        for p in passages:
            tag = tr(st, "tech.used") if p["doc_id"] in used else tr(st, "tech.not_used")
            dense = f" · dense {p['score_dense']:.3f}" if p.get("score_dense") is not None else ""
            st.markdown(
                f"`#{p['rank']}` **{esc(p.get('source_title') or p['scheme'])}** — {p['category']} "
                f"({tag}) · hybrid {p['score_hybrid']:.3f} · BM25 {p['score_bm25']:.2f} "
                f"· TF-IDF {p['score_tfidf']:.3f}{dense}"
            )
            st.markdown(f'<div class="tca-scroll">{esc(p.get("content", ""))}</div>',
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(st, assistant):
    ss = st.session_state
    vf.init(ss)
    page_title(st, "ask.title", "ask.intro")
    st.radio(tr(st, "ask.answer_lang"), ["te", "en"], key="answer_language", horizontal=True,
             format_func=lambda v: "తెలుగు" if v == "te" else "English")

    busy = vf.is_busy(ss)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.container(border=True):
            _voice_panel(st, assistant, busy)
    with right:
        with st.container(border=True):
            _text_panel(st, busy)

    _run_pending(st, assistant)
    _flash(st)
    busy = vf.is_busy(ss)
    _transcript_editor(st, busy)
    _error_panel(st, busy)

    result = ss.get("last_result")
    if result and result.get("ok"):
        show_result(st, result)
        _followup(st, assistant, busy)
        _technical(st, result)
        st.button(tr(st, "ask.clear_chat"), key="clear_chat", on_click=_clear_conversation, args=(st,))