"""Evaluation and diagnostics: real metrics only, plus advanced retrieval settings."""
import os

import pandas as pd

from config import INTENT_EN, RETRIEVAL_DEFAULTS
from eval_report import heldout_intent_metrics, load_model_card, plot_paths
from paths import OUTPUTS_DIR
from ui.components import page_title, section, status_badge
from ui.i18n import tr


def _reset_advanced(st):
    for key in ("top_k", "bm25_weight", "dense_weight", "suffix_stripping"):
        st.session_state[key] = RETRIEVAL_DEFAULTS[key]


def _advanced(st, assistant):
    with st.expander(tr(st, "adv.title")):
        st.caption(tr(st, "adv.help"))
        st.slider(tr(st, "adv.top_k"), 1, 10, key="top_k")
        st.slider(tr(st, "adv.bm25"), 0.0, 3.0, step=0.1, key="bm25_weight")
        neural = assistant.retriever.status()["neural_dense"] == "operational"
        st.slider(tr(st, "adv.dense"), 0.0, 3.0, step=0.1, key="dense_weight", disabled=not neural)
        st.checkbox(tr(st, "adv.suffix"), key="suffix_stripping")
        st.button(tr(st, "adv.reset"), on_click=_reset_advanced, args=(st,))


def _intent(st, assistant):
    section(st, "eval.intent")
    card = load_model_card()
    if not card:
        st.markdown(status_badge(st, "not_evaluated") + " " + tr(st, "eval.no_card"), unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        c1.metric(tr(st, "eval.holdout_f1"), f"{card.get('grouped_holdout_macro_f1', '—')}")
        c2.metric(tr(st, "eval.n_train"), str(card.get("n_train", "—")))
        cv = card.get("grouped_cv_macro_f1") or {}
        if cv:
            st.dataframe(pd.DataFrame([{"model": k, "CV macro-F1": v.get("mean"), "std": v.get("std")}
                                       for k, v in cv.items()]), hide_index=True, use_container_width=True)
    st.markdown(f"**{tr(st, 'eval.heldout')}**")
    if assistant.intent is None:
        st.warning(assistant.intent_error or tr(st, "status.unavailable"))
        return
    if "heldout_metrics" not in st.session_state:
        with st.spinner(tr(st, "eval.scoring")):
            st.session_state.heldout_metrics = heldout_intent_metrics(assistant.intent)
    metrics = st.session_state.heldout_metrics
    if not metrics:
        st.markdown(status_badge(st, "not_evaluated"), unsafe_allow_html=True)
        return
    st.dataframe(pd.DataFrame([{
        "condition": name, "accuracy": round(v["accuracy"], 4),
        "macro-F1": round(v["macro_f1"], 4),
        "precision (macro)": round(v["precision_macro"], 4), "recall (macro)": round(v["recall_macro"], 4),
    } for name, v in metrics["conditions"].items()]), hide_index=True, use_container_width=True)
    for name, vals in metrics["conditions"].items():
        with st.expander(f"{name} — {tr(st, 'eval.per_class')}"):
            st.dataframe(pd.DataFrame([{
                "intent": INTENT_EN.get(lab, lab), "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4), "f1": round(v["f1"], 4), "support": int(v["support"]),
            } for lab, v in vals["per_class"].items()]), hide_index=True, use_container_width=True)
            st.caption(tr(st, "eval.confusion"))
            labels = [INTENT_EN.get(x, x) for x in vals["labels"]]
            st.dataframe(pd.DataFrame(vals["confusion_matrix"], index=labels, columns=labels),
                         use_container_width=True)


def _coverage(st):
    section(st, "eval.coverage")
    st.caption(tr(st, "eval.retrieval_note"))
    if st.button(tr(st, "eval.coverage_run"), key="run_coverage"):
        from evaluate_coverage import audit
        with st.spinner(tr(st, "eval.coverage_running")):
            st.session_state.coverage_metrics = audit()["metrics"]
    metrics = st.session_state.get("coverage_metrics")
    if metrics:
        rows = []
        for key, value in metrics.items():
            if key == "metric_note":
                continue
            shown = f"{value:.1%}" if isinstance(value, float) else str(value)
            rows.append({"metric": tr(st, f"metric.{key}"), "value": shown})
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _asr(st, assistant):
    section(st, "eval.asr")
    st.markdown(f"{tr(st, 'eval.live_asr')}: " + status_badge(st, assistant.asr.status()),
                unsafe_allow_html=True)
    info = assistant.asr.model_info()
    st.caption(tr(st, "asr.model_in_use", model=info["model"]) if info["model"]
               else tr(st, "asr.model_none"))
    if info["load_errors"]:
        st.markdown(f"**{tr(st, 'asr.load_errors')}**")
        for err in info["load_errors"]:
            st.code(err, language=None)
    st.caption(tr(st, "eval.asr_note"))
    st.caption(tr(st, "eval.asr_stored"))
    figures = dict(plot_paths())
    wer = os.path.join(OUTPUTS_DIR, "figures", "experiment_1_wer_curve.png")
    if os.path.exists(wer):
        figures["Experiment 1 — WER curve"] = wer
    section(st, "eval.figures")
    if not figures:
        st.caption(tr(st, "eval.no_figures"))
    cols = st.columns(2)
    for i, (title, path) in enumerate(figures.items()):
        with cols[i % 2]:
            st.image(path, caption=title, use_container_width=True)


def render(st, assistant):
    page_title(st, "eval.title", "eval.intro")
    _advanced(st, assistant)
    _intent(st, assistant)
    _coverage(st)
    section(st, "eval.answer_quality")
    st.caption(tr(st, "eval.answer_note"))
    _asr(st, assistant)