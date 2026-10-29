"""End-to-end civic query pipeline. UI should call CivicAssistant only."""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_generator import generate_grounded_answer, provider_status
from asr import ASRService
from config import FOLLOWUP_MARKERS
from intent_api import IntentService
from kb import load_chunks, scheme_summaries
from paths import MODEL_PATH
from retriever import HybridRetriever
from query_analysis import detect_category, evidence_category, identify_scheme
from query_expansion import expand_query


def _mentions_scheme(text, chunks):
    t = text or ""
    names = {c["scheme"] for c in chunks}
    return any(name.split("/")[0].strip() in t or name in t for name in names) or (
        "రైతు" in t or "ఆరోగ్యశ్రీ" in t or "అమ్మ ఒడి" in t or "ఆసరా" in t or "బంధు" in t or "భరోసా" in t
    )


def _looks_like_followup(text):
    t = text or ""
    generic = ("పత్రాలు", "డాక్యుమెంట్", "దరఖాస్తు", "అర్హత", "ప్రయోజనం", "హెల్ప్‌లైన్",
               "documents", "apply", "eligible", "benefit", "helpline")
    return any(m in t for m in FOLLOWUP_MARKERS) or any(m in t.lower() for m in generic)


class CivicAssistant:
    def __init__(self):
        self.chunks = load_chunks()
        self.schemes = scheme_summaries(self.chunks)
        self.asr = ASRService()
        self.retriever = HybridRetriever(self.chunks)
        self.intent_error = None
        self.intent = None
        if not os.path.exists(MODEL_PATH):
            self.intent_error = (
                f"Intent model not found at {MODEL_PATH}. "
                "Run: python src/train_intent_model.py"
            )
        else:
            try:
                self.intent = IntentService()
                self.intent.predict("పథకం", normalize=True)
            except Exception as exc:
                self.intent = None
                self.intent_error = (
                    f"Could not load the intent model ({exc}). "
                    "If sklearn versions differ, retrain with python src/train_intent_model.py"
                )

    def component_status(self):
        """Real status of each pipeline component (no invented availability).

        Status values: operational | not_loaded | fallback | unavailable.
        ``id`` values are stable keys the UI localizes.
        """
        asr_st = self.asr.status()
        ret = self.retriever.status()
        llm = provider_status()
        return [
            {"id": "asr", "status": asr_st, "detail": self.asr.error or ""},
            {"id": "normalize", "status": "operational", "detail": "TeluguDialectNormalizer"},
            {"id": "intent", "status": "operational" if self.intent is not None else "unavailable",
             "detail": self.intent_error or "char n-gram TF-IDF + LogisticRegression"},
            {"id": "retrieve", "status": "operational",
             "detail": f"BM25 + char TF-IDF RRF, {ret['n_chunks']} chunks; neural dense: {ret['neural_dense']}"},
            {"id": "answer", "status": "operational" if llm["available"] else "fallback",
             "detail": (f"LLM provider {llm['provider']} configured; extractive fallback on failure"
                        if llm["configured"] else "Extractive grounded answers (no LLM configured)")},
        ]

    def run(
        self,
        text: str,
        input_mode="text",
        transcript=None,
        previous_schemes=None,
        top_k=5,
        conversation_context=None,
        answer_language="te",
        progress=None,
    ):
        """Answer one question.

        ``progress`` is an optional callable receiving a stage name
        ("analysis", "retrieval", "answer") just before that stage really
        starts, so the UI can show honest stage-by-stage status.
        """
        def _stage(name):
            if progress is not None:
                progress(name)

        original = (text or "").strip()
        answer_language = "en" if answer_language == "en" else "te"
        errors = []
        if not original:
            return {
                "ok": False,
                "error": (
                    "Please enter a question."
                    if answer_language == "en"
                    else "దయచేసి ఒక ప్రశ్నను నమోదు చేయండి."
                ),
                "input_mode": input_mode,
            }

        _stage("analysis")
        retrieval_query = original
        if answer_language != "en":
            # Telugu questions get query expansion before lexical retrieval so
            # dialect variants ("పత్రాలు" -> "డాక్యుమెంట్లు", "ప్రయోజనం" -> "సహాయం")
            # do not narrow the match. Typed English queries pass through unchanged.
            expansion = expand_query(original)
            if expansion != original:
                retrieval_query = expansion + " " + retrieval_query
        context_note = None
        explicit_scheme = identify_scheme(original, self.schemes and [s["scheme"] for s in self.schemes])
        active_scheme = explicit_scheme
        if previous_schemes and _looks_like_followup(original) and not explicit_scheme:
            active_scheme = previous_schemes[0]
            retrieval_query = active_scheme + " " + original
            context_note = (
                "Follow-up used the previous scheme name(s) for retrieval: "
                + active_scheme
            )
        # The display name returned by identify_scheme (e.g. "రైతు బంధు / రైతు భరోసా")
        # differs from the canonical scheme name stored on the KB chunks
        # (e.g. "రైతు భరోసా"). The display name is kept for answer rendering
        # (answer_language.py renders the English label), while the canonical
        # name is used for the retriever's exact-match scheme filter, because
        # chunks store the short canonical name.
        retrieval_scheme = active_scheme
        if active_scheme:
            from kb import scheme_summaries

            summaries = scheme_summaries(self.chunks)
            matches = [s["scheme"] for s in summaries if s["scheme"] in active_scheme]
            if matches:
                retrieval_scheme = matches[0]

        intent_result = None
        if self.intent is None:
            errors.append(self.intent_error)
            normalized = original
            edits = []
            intent_label = None
            confidence = None
            in_scope = True
        else:
            intent_result = self.intent.predict(original, normalize=True)
            normalized = intent_result["normalized"]
            edits = intent_result.get("edits") or []
            intent_label = intent_result["intent"]
            confidence = intent_result["confidence"]
            in_scope = bool(intent_result["in_scope"])
            retrieval_query = (
                (active_scheme + " " + normalized)
                if context_note
                else normalized
            )
        # Question category: explicit wording rules first (reliable), then the
        # classifier only when it is confident. A low-confidence classifier
        # label is used as a *tentative* category: retrieval is not filtered by
        # it, but the answer layer still refuses to answer from other topics.
        question_category = detect_category(original) or detect_category(normalized)
        category_source = "rules" if question_category else None
        category_uncertain = False
        retrieval_category = evidence_category(question_category) if question_category else None
        if not question_category and intent_label:
            question_category = intent_label
            if in_scope:
                category_source = "intent_model"
                retrieval_category = intent_label
            else:
                category_source = "intent_model_low_confidence"
                category_uncertain = True
        retrieval_intent = intent_label if category_source != "rules" else None
        # Use the canonical scheme name for the retriever (exact-match filter).
        # The display name (active_scheme) is kept for the answer rendering.
        scheme_for_retrieval = retrieval_scheme
        _stage("retrieval")
        passages = self.retriever.search(
            retrieval_query,
            intent=retrieval_intent,
            top_k=top_k,
            in_scope=in_scope if intent_label else True,
            scheme=retrieval_scheme,
            category=retrieval_category,
            category_uncertain=category_uncertain,
        )
        source_hints = []
        if not passages and active_scheme:
            source_hints = self.retriever.search(
                retrieval_query,
                top_k=min(top_k, 3),
                scheme=retrieval_scheme,
            )
        _stage("answer")
        answer = generate_grounded_answer(
            original,
            passages,
            in_scope=in_scope if intent_label else True,
            intent=intent_label,
            context=conversation_context,
            scheme=retrieval_scheme,
            category=question_category,
            source_hints=source_hints,
            language=answer_language,
            category_uncertain=category_uncertain,
        )

        return {
            "ok": True,
            "error": None,
            "warnings": errors,
            "input_mode": input_mode,
            "original_query": original,
            "transcript": transcript,
            "normalized_query": normalized,
            "normalization_edits": [
                {"original": a, "replacement": b, "rule": r} for a, b, r in edits
            ],
            "intent": intent_label,
            "question_category": question_category,
            "category_source": category_source,
            "scheme": retrieval_scheme,
            "confidence": confidence,
            "in_scope": in_scope if intent_label is not None else None,
            "intent_threshold": getattr(self.intent, "threshold", None) if self.intent else None,
            "context_note": context_note,
            "passages": passages,
            "answer": answer,
            "answer_language": answer_language,
            "retrieval_query": retrieval_query,
            "schemes_mentioned": (
                [active_scheme] if active_scheme else
                list({p["scheme"] for p in answer.get("used_passages") or []})
            ),
        }
