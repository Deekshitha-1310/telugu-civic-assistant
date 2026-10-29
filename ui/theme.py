"""Shared design tokens and CSS for every page.

Palette (from the brief): light neutral background, navy primary, subtle teal
accent, dark readable text, muted secondary text. One family for both
scripts (Noto Sans + Noto Sans Telugu share metrics, so mixed Telugu/English
lines sit on the same baseline). Widths and font sizes are Streamlit's
defaults at normal browser zoom; nothing was resized from zoomed-out
screenshots.
"""

TOKENS = {
    "bg": "#F4F6F9",
    "surface": "#FFFFFF",
    "navy": "#1B3A6B",
    "navy_hover": "#244C8A",
    "teal": "#0E7C7B",
    "teal_soft": "#E6F3F2",
    "text": "#17202E",
    "muted": "#5A6577",
    "border": "#DCE2EA",
    "warn_bg": "#FFF8E6", "warn_fg": "#7A5300", "warn_bd": "#F1D9A0",
    "err_bg": "#FDF1F1", "err_fg": "#9B1C1C", "err_bd": "#F3C9C9",
    "ok_bg": "#EEF7F1", "ok_fg": "#1D6B3A", "ok_bd": "#C5E3CF",
    "radius_lg": "14px", "radius": "10px", "radius_sm": "6px",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&display=swap');

:root {{
  --bg: {bg}; --surface: {surface}; --navy: {navy}; --navy-hover: {navy_hover};
  --teal: {teal}; --teal-soft: {teal_soft}; --text: {text}; --muted: {muted};
  --border: {border}; --r-lg: {radius_lg}; --r: {radius}; --r-sm: {radius_sm};
}}
html, body, .stApp, .stMarkdown, button, input, textarea, label, select {{
  font-family: "Noto Sans", "Noto Sans Telugu", system-ui, sans-serif;
}}
.stApp {{ background: var(--bg); color: var(--text); }}
.block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1180px; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.01em; }}
a {{ color: var(--navy); }}

/* Header */
.tca-brand {{ display: flex; align-items: center; gap: .75rem; }}
.tca-mark {{
  width: 40px; height: 40px; border-radius: 10px; background: var(--navy);
  display: grid; place-items: center; color: #fff; font-weight: 700; font-size: 1.15rem;
}}
.tca-brand-name {{ font-weight: 700; font-size: 1.12rem; color: var(--navy); line-height: 1.2; }}
.tca-brand-sub {{ font-size: .82rem; color: var(--muted); line-height: 1.35; }}
.tca-rule {{ height: 1px; background: var(--border); margin: .4rem 0 1.1rem; }}

/* Navigation: the radio that follows .tca-nav-anchor renders as tabs */
div:has(> .tca-nav-anchor) {{ display: none; }}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"],
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] {{
  gap: .25rem; flex-wrap: wrap;
}}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"] label,
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] label {{
  padding: .42rem .9rem; border-radius: 999px; margin: 0; border: 1px solid transparent;
  cursor: pointer;
}}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"] label > div:first-child,
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] label > div:first-child {{
  display: none;
}}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"] label:hover,
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] label:hover {{
  background: #E9EEF5;
}}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"] label:has(input:checked),
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] label:has(input:checked) {{
  background: var(--navy); border-color: var(--navy);
}}
[data-testid="stElementContainer"]:has(.tca-nav-anchor) + [data-testid="stElementContainer"] [role="radiogroup"] label:has(input:checked) p,
.element-container:has(.tca-nav-anchor) + .element-container [role="radiogroup"] label:has(input:checked) p {{
  color: #fff; font-weight: 600;
}}

/* Page heading */
.tca-page-title {{ font-size: 1.6rem; font-weight: 700; margin: 0 0 .2rem; color: var(--text); }}
.tca-page-intro {{ color: var(--muted); margin: 0 0 1.1rem; max-width: 68ch; line-height: 1.55; }}
.tca-section {{ font-size: 1.05rem; font-weight: 650; color: var(--text); margin: 1.4rem 0 .5rem; }}
.tca-muted {{ color: var(--muted); font-size: .88rem; line-height: 1.5; }}

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--surface); border-color: var(--border) !important; border-radius: var(--r) !important;
}}

/* Hero (dashboard) */
.tca-hero {{ padding: .4rem 0 1rem; max-width: 62ch; }}
.tca-hero h1 {{ font-size: 2rem; line-height: 1.25; margin: 0 0 .6rem; color: var(--navy); font-weight: 700; }}
.tca-hero p {{ color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin: 0; }}

/* Stat row */
.tca-stat-value {{ font-size: 1.55rem; font-weight: 700; color: var(--navy); line-height: 1.2; }}
.tca-stat-label {{ font-size: .85rem; color: var(--muted); }}

/* Badges */
/* Dashboard: lighter visual layout */
.tca-hero {{ padding: .5rem 0 1.2rem; max-width: 62ch; }}
.tca-hero h1 {{ font-size: 2.1rem; line-height: 1.22; margin: 0 0 .55rem; color: var(--navy); font-weight: 700; }}
.tca-hero p {{ color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin: 0; }}
.tca-stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--navy); line-height: 1.2; }}
.tca-stat-label {{ font-size: .82rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
.tca-badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .72rem;
  font-weight: 600; border: 1px solid; white-space: nowrap; margin: 0 .15rem .4rem 0; }}
.tca-badge.ok {{ background: {ok_bg}; color: {ok_fg}; border-color: {ok_bd}; }}
.tca-badge.warn {{ background: {warn_bg}; color: {warn_fg}; border-color: {warn_bd}; }}
.tca-badge.no {{ background: #F1F4F8; color: var(--muted); border-color: var(--border); }}
.tca-badge.err {{ background: {err_bg}; color: {err_fg}; border-color: {err_bd}; }}
.tca-badge.teal {{ background: var(--teal-soft); color: var(--teal); border-color: #BFE0DE; }}
.tca-status-row {{ display: flex; justify-content: space-between; align-items: center; gap: .75rem;
  padding: .55rem 0; border-bottom: 1px solid var(--border); }}
.tca-status-row:last-child {{ border-bottom: none; }}
.tca-status-name {{ font-weight: 550; font-size: .92rem; }}
.tca-history-item {{ font-size: .9rem; line-height: 1.55; margin: .3rem 0; color: var(--text); }}
.tca-history-item .tca-muted {{ font-size: .8rem; }}

.tca-badge.ok {{ background: {ok_bg}; color: {ok_fg}; border-color: {ok_bd}; }}
.tca-badge.warn {{ background: {warn_bg}; color: {warn_fg}; border-color: {warn_bd}; }}
.tca-badge.no {{ background: #F1F4F8; color: var(--muted); border-color: var(--border); }}
.tca-badge.err {{ background: {err_bg}; color: {err_fg}; border-color: {err_bd}; }}
.tca-badge.teal {{ background: var(--teal-soft); color: var(--teal); border-color: #BFE0DE; }}

/* Status list */
.tca-status-row {{ display: flex; justify-content: space-between; align-items: center; gap: .75rem;
  padding: .55rem 0; border-bottom: 1px solid var(--border); }}
.tca-status-row:last-child {{ border-bottom: none; }}
.tca-status-name {{ font-weight: 550; }}

/* Stage tracker */
.tca-stages {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .6rem; }}
.tca-stage {{ font-size: .8rem; padding: .22rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  color: var(--muted); background: var(--surface); }}
.tca-stage.done {{ color: {ok_fg}; border-color: {ok_bd}; background: {ok_bg}; }}
.tca-stage.active {{ color: var(--navy); border-color: var(--navy); font-weight: 600; }}
.tca-stage.failed {{ color: {err_fg}; border-color: {err_bd}; background: {err_bg}; }}

/* Answer */
.tca-question {{ color: var(--muted); font-size: .9rem; margin-bottom: .35rem; }}
.tca-answer {{
  background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--teal);
  border-radius: var(--r-sm) var(--r) var(--r) var(--r-sm); padding: 1rem 1.2rem;
  font-size: 1.04rem; line-height: 1.85; white-space: pre-wrap; color: var(--text);
  font-family: "Noto Sans Telugu", "Noto Sans", sans-serif;
}}
.tca-notice {{ border-radius: var(--r-sm); padding: .55rem .85rem; font-size: .88rem; line-height: 1.5;
  margin: .45rem 0; border: 1px solid; }}
.tca-notice.warn {{ background: {warn_bg}; color: {warn_fg}; border-color: {warn_bd}; }}
.tca-notice.info {{ background: var(--teal-soft); color: #0B5655; border-color: #BFE0DE; }}

/* Source card */
.tca-source {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r);
  padding: .8rem 1rem; margin-bottom: .55rem; }}
.tca-source-title {{ font-weight: 600; color: var(--text); margin-bottom: .25rem; }}
.tca-source-meta {{ font-size: .82rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: .3rem .9rem;
  margin-bottom: .35rem; align-items: center; }}
.tca-source-excerpt {{ font-size: .86rem; color: #344054; border-left: 2px solid var(--border);
  padding-left: .65rem; margin: .4rem 0; line-height: 1.55; }}

/* Controls */
.stButton > button, .stDownloadButton > button {{ border-radius: 8px; font-weight: 550; }}
.stButton > button[kind="primary"] {{ background: var(--navy); border-color: var(--navy); }}
.stButton > button[kind="primary"]:hover {{ background: var(--navy-hover); border-color: var(--navy-hover); }}
.stButton > button:focus-visible {{ outline: 3px solid #9CC3E8; outline-offset: 2px; }}
.stTextArea textarea, .stTextInput input {{ border-radius: 8px; line-height: 1.65; }}
[data-testid="stAlert"] {{ border-radius: var(--r-sm); }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""

CSS = _CSS.format(**TOKENS)


def inject(st):
    """Inject the global CSS once per run."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


_BADGE_CLASS = {
    "operational": "ok", "available": "ok", "verified_current": "ok",
    "not_loaded": "warn", "fallback": "teal", "official_historical": "warn",
    "official_accessed_currentness_unverified": "warn", "legacy_unverified": "no",
    "not_evaluated": "no", "unavailable": "err",
}


def badge(status, label=None):
    """HTML badge. ``label`` should be a localized string; falls back to the raw status."""
    cls = _BADGE_CLASS.get(status, "no")
    text = label if label is not None else str(status).replace("_", " ")
    return f'<span class="tca-badge {cls}">{text}</span>'

CSS += """
.tca-scroll { max-height: 240px; overflow: auto; font-size: .82rem; line-height: 1.5;
  background: #F8FAFC; border: 1px solid var(--border); border-radius: var(--r-sm);
  padding: .5rem .7rem; white-space: pre-wrap; color: #344054; }
"""
