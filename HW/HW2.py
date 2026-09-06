import streamlit as st
from openai import OpenAI
import anthropic
import requests
from bs4 import BeautifulSoup


# reading the text content from a web page URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


st.title("📄 HW 2 - URL Summarizer")
st.write(
    "Enter a URL and choose how you'd like it summarized. "
    "Use the options in the sidebar to control the LLM, summary type, and language."
)

# the user enters a URL at the top of the page (not the sidebar)
url = st.text_input("Enter a URL to summarize:", placeholder="https://example.com/article")

#  sidebar controls 

# choosing which LLM to use (OpenAI or Claude)
llm_provider = st.sidebar.selectbox(
    "Choose an LLM:",
    ("OpenAI", "Claude"),
)

# 1) Summary type: three options.
summary_type = st.sidebar.selectbox(
    "Type of summary:",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ),
)

# 2) Output language.
language = st.sidebar.selectbox(
    "Summary language:",
    ("English", "French", "Spanish"),
)

# 3) Model choice: checkbox picks the advanced model.
use_advanced = st.sidebar.checkbox("Use advanced model")
# pick the model based on the provider and the advanced checkbox
if llm_provider == "OpenAI":
    model = "gpt-5-mini" if use_advanced else "gpt-5-nano"
else:  # Claude
    model = "claude-opus-4-8" if use_advanced else "claude-haiku-4-5-20251001"
st.sidebar.caption(f"Using {llm_provider}: {model}")

if url:
    # read the web page text from the URL
    document = read_url_content(url)

    # only summarize if we successfully read the page
    if document:
        # Turn the chosen summary type into a clear instruction for the model.
        if summary_type == "Summarize in 100 words":
            instruction = "Summarize the document in about 100 words."
        elif summary_type == "Summarize in 2 connecting paragraphs":
            instruction = "Summarize the document in two connecting paragraphs."
        else:
            instruction = "Summarize the document in exactly 5 bullet points."

        # Add the language requirement to the instruction.
        instruction += f" Write the summary in {language}."

        prompt = f"Here's a document:\n\n{document}\n\n---\n\n{instruction}"

        # send the prompt to whichever LLM the user selected
        if llm_provider == "OpenAI":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            st.write_stream(stream)
        else:  # Claude uses a different client and streaming method
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            with client.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                st.write_stream(stream.text_stream)