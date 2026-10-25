# PolicyLens — Company Policy Assistant

PolicyLens compares rules-based search, an LLM with all policy records in context (no vector index), and an LLM with TF-IDF retrieval over the policy database. It uses the supplied `company_policies.csv` transformed to `policies.json` (98 records). The web interface reports each answer, matched policy, measured response latency, estimated token count, and a qualitative unsupported-answer risk. For real LLM comparisons, set `OPENAI_API_KEY`; without it, the app clearly labels its local extractive fallback rather than claiming a real LLM response. Token counts are estimates, not provider billing measurements.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY for real LLM calls
python app.py
```

Open http://localhost:5000.

## Deploy website

The web application is a Flask app. For deployment, use a Python-compatible host (Render, Railway, or a Vercel Python runtime). Configure the start command as `gunicorn app:app` and add `OPENAI_API_KEY` and `OPENAI_MODEL` as environment variables if using real LLM calls. Do not commit `.env` or secrets. The project includes `vercel.json` for Vercel deployment.

## Slack bot (Socket Mode; local runtime is fine)

1. Create a **separate Slack workspace** for this assignment. Use a workspace name that identifies you as the student, e.g. `Company Policy Assistant — [Your Name]`.
2. Create a public testing channel such as `#policy-bot-testing-[yourname]`. Include your student name in the workspace/channel name or channel topic.
3. At https://api.slack.com/apps, create a Slack app **in that new workspace**. Enable **Socket Mode** and create an app-level token with `connections:write`.
4. Add bot token scopes `app_mentions:read` and `chat:write`; subscribe to the bot event `app_mention`; install the app to the workspace.
5. Copy the bot token (`xoxb-...`) and app-level token (`xapp-...`) into environment variables based on `.env.example`. Never commit tokens.
6. Invite the bot user to the testing channel, then run:
   ```bash
   python slack_bot.py
   ```
7. In the testing channel ask, for example: `@PolicyLens How long are products covered under warranty?`
8. Invite `raz@sdu.dk` to the channel using Slack's channel member menu → **Add people** → enter `raz@sdu.dk` → **Send**. This requires workspace permissions and must be done in Slack; the code cannot invite a person by itself.
9. Capture a real screenshot showing the channel, the bot mention, and the answer with the policy title visible.

The bot uses TF-IDF retrieval and returns the matching policy title, department, category, and policy text. If `OPENAI_API_KEY` is configured, the response is generated using the retrieved policies as context; otherwise the answer uses an extractive local fallback.

## Methods and interpretation

- **Rules-based search:** deterministic TF-IDF ranking and policy-text extraction. It is fast and grounded but may miss unusual wording.
- **LLM without vector index:** sends all 98 policy records to the LLM, increasing context/token cost and giving the model more unrelated material.
- **LLM with vector index:** retrieves the top matching records first, then asks the LLM to answer only from those records. This reduces irrelevant context, but does not eliminate unsupported generation.
- The “unsupported-answer tendency” is a qualitative design-risk label, not a measured hallucination rate. To report an empirical rate, prepare a fixed set of policy and out-of-scope questions, manually score each answer against the CSV, and report unsupported answers / total answers.

## Files

- `app.py` — web server, three-way comparison API, TF-IDF index
- `static/index.html` — comparison website
- `slack_bot.py` — Slack Socket Mode bot
- `policies.json` — provided policy dataset
- `comparison.md` — required two-paragraph comparison
