from config import CATEGORY_EN, CATEGORY_TE
from ui.components import page_title, verification_label
from ui.i18n import current_lang, tr
from ui.theme import badge


def render(st, assistant):
    page_title(st, "schemes.title", "schemes.intro")
    labels = CATEGORY_TE if current_lang(st) == "te" else CATEGORY_EN
    needle = (st.text_input(tr(st, "schemes.search")) or "").strip().casefold()
    shown = 0
    for scheme in assistant.schemes:
        hay = (scheme["scheme"] + " " + " ".join(
            (c.get("source_title") or "") + " " + c["content"] for c in scheme["chunks"])).casefold()
        if needle and needle not in hay:
            continue
        shown += 1
        with st.container(border=True):
            st.subheader(scheme["scheme"])
            st.markdown(f"**{tr(st, 'schemes.categories')}:** "
                        + ", ".join(labels.get(c, c) for c in scheme["categories"]))
            with st.expander(f"{tr(st, 'schemes.passages')} ({scheme['n_chunks']})"):
                for chunk in scheme["chunks"]:
                    status, label = verification_label(st, chunk)
                    date = chunk.get("published_date") or ""
                    st.markdown(
                        f"**{chunk.get('source_title') or chunk['doc_id']}** — "
                        f"{labels.get(chunk['category'], chunk['category'])} {badge(status, label)} {date}",
                        unsafe_allow_html=True)
                    st.markdown(f'<div class="tca-scroll">{_esc(chunk["content"])}</div>',
                                unsafe_allow_html=True)
                    if chunk.get("source_url"):
                        st.caption(chunk["source_url"])
    if not shown:
        st.info(tr(st, "schemes.no_match"))


def _esc(text):
    import html
    return html.escape(text or "")
