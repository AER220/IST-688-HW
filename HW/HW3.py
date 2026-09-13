import streamlit as st
from openai import OpenAI
import anthropic
import requests
from bs4 import BeautifulSoup


# reading the text content from a web page URL (re-used from HW2)
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


st.title("💬 HW 3 - Chatbot that Discusses URLs")

# describe to the user how this chatbot works (assignment item 6)
st.write(
    "This chatbot reads up to two web pages and answers your questions about them. "
    "Enter one or two URLs in the sidebar and pick which LLM to use. The page content "
    "is placed in a system prompt that is never discarded, so the bot always has it as context. "
    "For memory, I keep a buffer of the last 6 messages (3 back-and-forth exchanges) — older "
    "messages drop off, but the URL context and instructions always stay."
)

#  sidebar options 

# let the user enter up to two URLs
url1 = st.sidebar.text_input("URL 1:", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional):", placeholder="https://example.com")

# let the user pick which premium LLM to use (2 vendors, their flagship models)
llm_provider = st.sidebar.selectbox(
    "Choose an LLM:",
    ("OpenAI (GPT-5)", "Claude (Opus 4.8)"),
)
# how many messages to keep in memory (Part: buffer of 6 = 3 exchanges)
BUFFER_MESSAGES = 6

# read whichever URLs the user gave us and combine them into one context string
url_context = ""
if url1:
    content1 = read_url_content(url1)
    if content1:
        url_context += f"\n\n--- Content from {url1} ---\n{content1}"
if url2:
    content2 = read_url_content(url2)
    if content2:
        url_context += f"\n\n--- Content from {url2} ---\n{content2}"

# build the system prompt: the URL content is the context that is never discarded
system_prompt = (
    "You are a helpful assistant. Answer the user's questions using the web page "
    "content provided below as your main source. If the answer isn't in the content, "
    "say so honestly.\n"
    f"{url_context}"
)

# set up the message history in session_state so it survives reruns
if "hw3_messages" not in st.session_state:
    st.session_state.hw3_messages = [
        {"role": "assistant", "content": "Hi! Add one or two URLs in the sidebar, then ask me about them."}
    ]

# show the conversation so far
for message in st.session_state.hw3_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# get the next user message
if prompt := st.chat_input("Ask about the URLs..."):
    st.session_state.hw3_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # keep only the last 6 messages (the buffer)
    buffered = st.session_state.hw3_messages[-BUFFER_MESSAGES:]

    # always put the system prompt (with URL context) first so it is never dropped
    messages_to_send = [{"role": "system", "content": system_prompt}] + buffered

    with st.chat_message("assistant"):
        # send to whichever premium model the user picked
        if llm_provider == "OpenAI (GPT-5)":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            stream = client.chat.completions.create(
               model="gpt-5-mini",
                messages=messages_to_send,
                stream=True,
            )
            response = st.write_stream(stream)
        else:  # Claude — different client, and it takes the system prompt separately
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            # Anthropic wants the system prompt as its own argument, and only
            # user/assistant messages in the messages list — so filter those out.
            chat_only = [m for m in messages_to_send if m["role"] != "system"]
            with client.messages.stream(
                model="claude-opus-4-8",
                max_tokens=1024,
                system=system_prompt,
                messages=chat_only,
            ) as stream:
                response = st.write_stream(stream.text_stream)

    # save the reply into memory
    st.session_state.hw3_messages.append({"role": "assistant", "content": response})