"""Optional grounded generation through Ollama or an OpenAI-compatible API.

The extractive generator remains the safe fallback. A provider is considered
available only after a successful live request; configuration alone is not
reported as availability.
"""
import json
import os
from urllib import request

from answer_generator import generate_answer


def provider_status():
    provider = os.getenv("TELUGU_CIVIC_LLM_PROVIDER", "fallback").lower()
    configured = provider == "ollama" or (
        provider in {"openai", "openai-compatible"} and bool(os.getenv("OPENAI_API_KEY"))
    )
    return {"provider": provider, "configured": configured, "available": False}


def _prompt(question, passages, context, language, category=None):
    def meta(p):
        bits = [p.get("source_title") or p.get("scheme") or ""]
        if p.get("published_date"):
            bits.append(f"dated {p['published_date']}")
        if p.get("temporal_scope"):
            bits.append(f"temporal scope {p['temporal_scope']}")
        bits.append(f"verification {p.get('verification_status') or 'unknown'}")
        return "; ".join(b for b in bits if b)

    evidence = "\n\n".join(
        f"[{p.get('doc_id')}] ({meta(p)})\n{(p.get('content') or '')[:1800]}\nURL: {p.get('source_url')}"
        for p in passages
    )
    return (
        "Answer the user's question using only the supplied official evidence. "
        "Do not invent facts, amounts, deadlines, eligibility, or URLs. "
        f"The question is about: {category or 'unspecified topic'}; answer only that topic "
        "and do not substitute information about a different topic (for example, do not "
        "answer an eligibility question with benefit amounts). "
        "Evidence marked historical or not verified as current must be described as such; "
        "never present it as today's policy. "
        f"If evidence is insufficient, say so. Answer in {language} in at most 120 words. "
        "Include only relevant source URLs.\n\n"
        f"Question: {question}\nConversation context: {context or '(none)'}\n"
        f"Official evidence:\n{evidence}"
    )


def _call_ollama(prompt):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False}
    req = request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=float(os.getenv("TELUGU_CIVIC_LLM_TIMEOUT", "20"))) as response:
        return json.loads(response.read().decode("utf-8")).get("response", "").strip()


def _call_openai(prompt):
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
               "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    req = request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    })
    with request.urlopen(req, timeout=float(os.getenv("TELUGU_CIVIC_LLM_TIMEOUT", "20"))) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def generate_grounded_answer(question, passages, in_scope=True, intent=None, context=None,
                             scheme=None, category=None, source_hints=None, language="te",
                             category_uncertain=False):
    fallback = generate_answer(
        passages, in_scope=in_scope, intent=intent, question=question,
        scheme=scheme, category=category, source_hints=source_hints, language=language,
        category_uncertain=category_uncertain,
    )
    if not fallback.get("grounded"):
        return fallback
    provider = os.getenv("TELUGU_CIVIC_LLM_PROVIDER", "fallback").lower()
    if provider not in {"ollama", "openai", "openai-compatible"}:
        fallback["generation"] = "extractive_fallback"
        return fallback
    try:
        evidence = fallback.get("used_passages") or passages
        prompt = _prompt(question, evidence, context, language, fallback.get("category"))
        text = _call_ollama(prompt) if provider == "ollama" else _call_openai(prompt)
        urls = {p.get("source_url") for p in evidence if p.get("source_url")}
        if not text or urls and not any(url in text for url in urls):
            raise ValueError("model response did not cite a retrieved official URL")
        return {**fallback, "text": text, "generation": "llm", "provider": provider}
    except Exception as exc:
        return {**fallback, "generation": "extractive_fallback",
                "llm_error": f"{type(exc).__name__}: {exc}"}
