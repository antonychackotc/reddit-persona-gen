import os
import praw #python reddit api wrapper
import streamlit as st
import google.generativeai as genai

# --- Configure Credentials (Manually, since .env is skipped)
GEMINI_API_KEY = "AIzaSyAQXXz2tgKC3H4jvsWu5doj-pMVva_RT3U"
REDDIT_CLIENT_ID = "OEy_ckwU9LwsHrJs7KXs0Q"
REDDIT_CLIENT_SECRET = "EXcJAlY0bQiWHWKnFGVk3sq8hAFl8Q"
REDDIT_USER_AGENT = "RedditPersonaScript by /u/antonychacko"

# --- Configure APIs
genai.configure(api_key=GEMINI_API_KEY)

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# --- Functions
def extract_username(url):
    if "reddit.com/user/" in url:
        return url.rstrip('/').split("/")[-1]
    return url

def fetch_user_content(username, max_items=30):
    user = reddit.redditor(username)
    posts = [f"Title: {p.title}\nBody: {p.selftext}" for p in user.submissions.new(limit=max_items)]
    comments = [f"Comment: {c.body}" for c in user.comments.new(limit=max_items)]
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

def generate_persona(prompt):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit UI
st.set_page_config(page_title="Reddit Persona Generator", layout="centered")
st.title("🔍 Reddit User Persona Generator (LLM-powered)")
st.markdown("Enter a Reddit profile URL below to generate a user persona using Gemini AI.")

user_url = st.text_input("Reddit Profile URL", placeholder="e.g., https://www.reddit.com/user/kojied/")

if st.button("Generate Persona"):
    if not user_url:
        st.warning("Please enter a Reddit profile URL.")
    else:
        with st.spinner("Fetching Reddit content..."):
            try:
                username = extract_username(user_url)
                posts, comments = fetch_user_content(username)

                if not posts and not comments:
                    st.error("No content found for this user.")
                else:
                    prompt = build_prompt(posts, comments)
                    with st.spinner("Generating persona using Gemini AI..."):
                        persona = generate_persona(prompt)

                    # Save to file
                    os.makedirs("persona_outputs", exist_ok=True)
                    output_path = f"persona_outputs/{username}_persona.txt"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(persona)

                    st.success("Persona generated successfully!")
                    st.download_button("📄 Download Persona", data=persona, file_name=f"{username}_persona.txt")
                    st.text_area("🧠 User Persona", persona, height=400)
            except Exception as e:
                st.error(f"Error: {e}")
