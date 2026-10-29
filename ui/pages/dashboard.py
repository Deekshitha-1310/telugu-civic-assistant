"""Home: what the assistant does, how to ask, what it knows, what really works.

Lighter than the previous version:
* hero text is trimmed to one line
* three primary action buttons sit side by side under the hero
* two compact columns replace the long "recent questions" list
* the system status row is on a single line
* the scheme list is a row of pills, not a paragraph
"""

from ui.components import esc, go_to, section, status_badge
from ui.i18n import tr


def _kb_counts(assistant):
    chunks = assistant.chunks
    # Count unique source URLs that are NOT legacy (team-written) records.
    # These are "official" in the sense they come from government domains,
    # but they are NOT marked as verified current — all sit at
    # "official_accessed_currentness_unverified", so we do not overclaim.
    official_urls = {
        c["source_url"]
        for c in chunks
        if c.get("source_role") != "legacy_record" and c.get("source_url")
    }
    verified = {
        c["source_url"]
        for c in chunks
        if c.get("verification_status") == "verified_current"
    }
    return (
        len(assistant.schemes),
        len(chunks),
        len(official_urls),
        len(verified),
    )


def render(st, assistant):
    # Hero: one short line, no long paragraph.
    st.markdown(
        f'<div class="tca-hero"><h1>{esc(tr(st, "dash.hero_title"))}</h1></div>',
        unsafe_allow_html=True,
    )

    # Action buttons in one row.
    asr_ok = assistant.asr.available()
    c1, c2, c3 = st.columns(3)
    c1.button(
        "\U0001F50A " + tr(st, "dash.voice_cta"),
        type="primary",
        use_container_width=True,
        disabled=not asr_ok,
        on_click=go_to,
        args=(st, "ask"),
        key="dash_voice",
    )
    c2.button(
        "\U0001F4E4 " + tr(st, "dash.upload_cta"),
        use_container_width=True,
        disabled=not asr_ok,
        on_click=go_to,
        args=(st, "ask"),
        key="dash_upload",
    )
    c3.button(
        "\U0001F4AC " + tr(st, "dash.text_cta"),
        type="secondary" if asr_ok else "primary",
        on_click=go_to,
        args=(st, "ask"),
        key="dash_text",
    )
    if not asr_ok:
        st.caption(tr(st, "dash.voice_off"))

    # Two compact columns: KB at a glance, recent questions side by side with
    # the system status so nothing dominates the page.
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section(st, "dash.kb_title")
        n_schemes, n_chunks, n_official, n_verified = _kb_counts(assistant)
        with st.container(border=True):
            cols = st.columns(4)
            # Show 4 stats: schemes, passages, source domains, verified.
            # The verified count is hidden when it is 0, because showing "0"
            # implies a failure. The note below covers the honesty about
            # currentness separately.
            display_cols = [
                (n_schemes, "dash.kb_schemes"),
                (n_chunks, "dash.kb_passages"),
                (n_official, "dash.kb_sources"),
            ]
            if n_verified > 0:
                display_cols.append((n_verified, "dash.kb_verified"))
            for col, (value, key) in zip(cols, display_cols):
                col.markdown(
                    f'<div class="tca-stat-value">{value}</div>'
                    f'<div class="tca-stat-label">{esc(tr(st, key))}</div>',
                    unsafe_allow_html=True,
                )
            if n_verified == 0:
                st.caption(tr(st, "dash.kb_note_unverified"))
            st.markdown(
                " ".join(
                    f'<span class="tca-badge no" style="margin:.15rem">{esc(s["scheme"])}</span>'
                    for s in assistant.schemes
                ),
                unsafe_allow_html=True,
            )

        section(st, "dash.recent")
        history = st.session_state.history[:5]
        if not history:
            st.caption(tr(st, "dash.no_recent"))
        for entry in history:
            status = (
                tr(st, f"result.{entry['answer_status']}")
                if entry.get("answer_status") in ("grounded", "no_evidence")
                else "—"
            )
            st.markdown(
                f"<p class=\"tca-history-item\"><b>{esc(entry['query'])}</b>"
                f"<span class=\"tca-muted\"> {esc(entry['timestamp'])} · {status}</span></p>",
                unsafe_allow_html=True,
            )

    with right:
        section(st, "dash.system_title")
        with st.container(border=True):
            rows = "".join(
                f'<div class="tca-status-row"><span class="tca-status-name">'
                f'{esc(tr(st, "component." + c["id"]))}</span>{status_badge(st, c["status"])}</div>'
                for c in assistant.component_status()
            )
            st.markdown(rows, unsafe_allow_html=True)
            st.caption(tr(st, "dash.system_note"))
