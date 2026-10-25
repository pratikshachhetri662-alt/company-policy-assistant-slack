import os, json, re, time
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "policies.json"), encoding="utf-8") as f:
    POLICIES = json.load(f)

app = Flask(__name__, static_folder="static", static_url_path="/static")
def corpus_text(p):
    return f"{p['title']} {p['department']} {p['category']} {p['policy_text']}"
DOCS = [corpus_text(p) for p in POLICIES]
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
MATRIX = VECTORIZER.fit_transform(DOCS)

def lexical_scores(question):
    q = VECTORIZER.transform([question])
    return cosine_similarity(q, MATRIX).ravel()

def retrieve(question, k=4):
    scores = lexical_scores(question)
    ids = scores.argsort()[::-1][:k]
    return [(POLICIES[i], float(scores[i])) for i in ids if scores[i] > 0.015]

def approx_tokens(s):
    # Reproducible approximation when tiktoken is not installed.
    return max(1, int(len(s.split()) * 1.3))

def policy_answer(question, policies, mode):
    if not policies:
        return "I couldn't find a matching policy in the supplied policy database. Please contact the relevant policy owner.", []
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            context = "\n\n".join(
                f"Policy: {p['title']}\nDepartment: {p['department']}\nCategory: {p['category']}\nText: {p['policy_text']}"
                for p in policies
            )
            prompt = (
                "Answer the employee's question using only the policy records below. "
                "Do not infer missing rules. If the records do not answer the question, say so. "
                "At the end, identify the relevant policy title(s). Keep the answer concise.\n\n"
                f"Policy records:\n{context}\n\nQuestion: {question}"
            )
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt
            )
            return response.output_text, [context, prompt, response.output_text]
        except Exception as e:
            return f"LLM request failed ({type(e).__name__}). Configure a valid OPENAI_API_KEY or use the local fallback.", []
    # No API key: a grounded extractive fallback, clearly disclosed in UI.
    best = policies[:3]
    answer = "Based on the policy database:\n" + "\n".join(
        f"• {p['policy_text']} (Relevant policy: {p['title']}; department: {p['department']}; category: {p['category']})"
        for p in best
    )
    return answer, [" ".join(DOCS), question, answer]

def result_for(name, question, candidates, mode):
    started = time.perf_counter()
    if mode == "rules":
        if candidates:
            answer = "Based on the policy database:\n" + "\n".join(
                f"• {p['policy_text']} (Relevant policy: {p['title']}; department: {p['department']}; category: {p['category']})"
                for p in candidates[:3]
            )
        else:
            answer = "I couldn't find a matching policy in the supplied policy database."
        text_parts = [question, answer]
    else:
        answer, text_parts = policy_answer(question, candidates, mode)
    elapsed = (time.perf_counter() - started) * 1000
    policy_titles = [p["title"] for p in candidates]
    used_text = " ".join(text_parts) if text_parts else answer
    return {
        "approach": name,
        "answer": answer,
        "policies": [{"title": p["title"], "department": p["department"], "category": p["category"], "policy_text": p["policy_text"]} for p in candidates],
        "response_ms": round(elapsed, 2),
        "token_use": approx_tokens(used_text),
        "token_note": "Estimated from whitespace word count; not provider billing tokens." if not os.getenv("OPENAI_API_KEY") else "Estimated locally; exact provider usage may differ.",
        "unsupported_risk": {
            "rules": "Low: answer is directly extracted from matching records; may miss paraphrases.",
            "llm_no_index": "High: all policies are supplied, but no retrieval boundary; generation may add unsupported details.",
            "llm_vector": "Lower than no-index: retrieved policy context narrows grounding, but the model can still infer unsupported details."
        }[mode],
        "mode": "OpenAI API" if os.getenv("OPENAI_API_KEY") else "Local demo fallback (no LLM API key)"
    }

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/policies")
def list_policies():
    return jsonify(POLICIES)

@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a policy question."}), 400
    ranked = retrieve(question, 8)
    # Rules-based search: ranked records and extractive answer only.
    rules = [p for p, score in ranked[:3]]
    # LLM without vector index: all records are context, no retrieval step.
    all_policies = POLICIES
    # LLM with vector index: only top semantically similar records are context.
    vector = [p for p, score in ranked[:4]]
    # In fallback mode, avoid flooding the display with all 98 policies; the token metric still reflects full-context cost.
    out_rules = result_for("Rules-based search", question, rules, "rules")
    out_no = result_for("LLM without vector index", question, all_policies if os.getenv("OPENAI_API_KEY") else rules, "llm_no_index")
    if not os.getenv("OPENAI_API_KEY"):
        out_no["token_use"] = approx_tokens("\n".join(DOCS) + question + out_no["answer"])
    out_vec = result_for("LLM with vector index", question, vector, "llm_vector")
    return jsonify({
        "question": question,
        "results": [out_rules, out_no, out_vec],
        "note": "Unsupported-answer tendency is a qualitative design risk, not a measured hallucination rate. Response time is measured for this request. Token counts are local estimates."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
