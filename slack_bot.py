import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from app import retrieve, policy_answer

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.event("app_mention")
def handle_mention(event, say, logger):
    question = event.get("text", "")
    # Remove the bot mention before retrieval.
    import re
    question = re.sub(r"<@[^>]+>", "", question).strip()
    matches = retrieve(question, k=4)
    policies = [p for p, score in matches]
    if not policies:
        say("I couldn't find a relevant policy in the policy database. Please contact the policy owner or HR.")
        return
    answer, _ = policy_answer(question, policies, "llm_vector")
    references = "\n".join(
        f"• *{p['title']}* — {p['department']} / {p['category']}\n_{p['policy_text']}_"
        for p in policies[:3]
    )
    say(f"*Policy answer*\n{answer}\n\n*Relevant policy records*\n{references}\n\n_Answers are grounded in the supplied policy database; please verify with the policy owner for edge cases._")

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
