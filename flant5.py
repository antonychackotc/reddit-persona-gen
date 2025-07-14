import os
import praw
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---- 1. Reddit API Setup ----
reddit = praw.Reddit(
    client_id="OEy_ckwU9LwsHrJs7KXs0Q",
    client_secret="EXcJAlY0bQiWHWKnFGVk3sq8hAFl8Q",
    user_agent="RedditPersonaScript by /u/antonychacko"
)

# ---- 2. Helper Functions ----

def extract_username(url):
    if "reddit.com/user/" in url:
        return url.rstrip('/').split("/")[-1]
    return url  # Already a username

def fetch_user_content(username, max_items=30):
    user = reddit.redditor(username)
    posts = []
    comments = []

    for post in user.submissions.new(limit=max_items):
        posts.append(f"Title: {post.title}\nBody: {post.selftext}")

    for comment in user.comments.new(limit=max_items):
        comments.append(f"Comment: {comment.body}")

    return posts, comments

def build_prompt(posts, comments):
    sample = "\n\n".join(posts[:5] + comments[:5])
    return f"""
You are an AI assistant tasked with creating a Reddit user persona based on this user's posts and comments.

Please include:
- Age
- Occupation
- Interests
- Location
- Personality

Also provide a short quote from each post or comment as evidence.

Reddit Data:
{sample}
"""

def generate_persona_with_flan(prompt):
    model_name = "google/flan-t5-base"  # Or "flan-t5-small" for low RAM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---- 3. Main Function ----

def main():
    # ✅ Hardcoded for testing (you can use input() later)
    user_url = "https://www.reddit.com/user/kojied/"
    username = extract_username(user_url)

    print(f"🔍 Fetching Reddit data for u/{username}...")
    posts, comments = fetch_user_content(username)

    if not posts and not comments:
        print("⚠️ No content found.")
        return

    prompt = build_prompt(posts, comments)
    persona = generate_persona_with_flan(prompt)

    os.makedirs("flant5_outputs", exist_ok=True)
    output_file = f"flant5_outputs/{username}_persona.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(persona)

    print(f"✅ Persona saved to: {output_file}")

if __name__ == "__main__":
    main()
