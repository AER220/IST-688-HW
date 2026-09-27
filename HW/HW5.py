# HW 5 - the "smart" version of my orgs chatbot.
#
# what changed from HW4: in HW4 i searched ChromaDB automatically on every single
# message, always using the user's raw prompt as the search. here i do it INTELLIGENTLY.
# i give the LLM a TOOL called relevant_club_info and let IT decide:
#   - whether the question even needs a club lookup (a "hi" or "thanks" gets no search)
#   - and what the actual search query should be (it can rewrite a vague follow-up like
#     "what about faith-based ones?" into a good query using the conversation context)
#
# it's the same two-call tool pattern i built in Lab 5, but the tool runs a vector search
# instead of a weather api:
#   call 1: give the model the relevant_club_info tool (tool_choice="auto"). it either
#           answers directly, or asks me to search with a query it wrote.
#   my code: run the vector search and get the matching club chunks back.
#   call 2: hand those results back and ask for the final answer, WITH NO TOOL this time
#           so it can't search again (step 3b). i stream this one.

# sqlite fix (same as HW4) - chroma needs a newer sqlite than streamlit cloud ships.
# this swaps in pysqlite3 BEFORE chromadb loads. try/except so it still runs where it's not installed.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
from pathlib import Path

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

#  settings 
HTML_FOLDER = Path(__file__).parent / "html"                    # the org html files live in here
CHROMA_PATH = str(Path(__file__).parent / "ChromaDB_for_HW5")   # the vector db is SAVED TO DISK here
COLLECTION_NAME = "HW5Collection"
EMBED_MODEL = "text-embedding-3-small"          # openai embeddings model for the vectors
CHAT_MODEL = "gpt-5-mini"                        # same llm as my other labs
INTERACTIONS = 5                                 # short-term memory: keep the last 5 interactions
MAX_CHARS_PER_CHUNK = 8000                       # safety cap per chunk so i stay under the token limit

st.title("HW 5 - Smart Syracuse Orgs Chatbot")
st.write("Ask about Syracuse student organizations. I only search the database when I actually need to.")

#  api key (from secrets, same as my other labs) 
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# ------------------ chunking (same method as HW4) ------------------
# i split each org page into TWO halves, cutting at the nearest space to the midpoint so i
# never split a word. each org page is short and about a single club, so two halves keep
# related info together but give the retriever smaller, tighter pieces to match against.
def split_into_two(text):
    text = text.strip()
    if len(text) <= 1:
        return [text]
    mid = len(text) // 2
    split_at = text.find(" ", mid)   # slide to the next space so words stay whole
    if split_at == -1:
        split_at = mid
    first, second = text[:split_at].strip(), text[split_at:].strip()
    return [c for c in (first, second) if c]


# ------------------ read one html file into clean text ------------------
def html_to_text(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):   # drop code/style noise
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ------------------ build the vector db (build once) ------------------
def build_hw5_vectordb(api_key):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBED_MODEL,
    )

    # PersistentClient writes the db to disk so it survives app restarts
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
    )

    # only build it if it's EMPTY. if it already has data, reuse it so i only embed once.
    if collection.count() > 0:
        st.session_state._hw5_chroma_client = chroma_client
        return collection

    # find the html files: check html/ first, then su_orgs/, else search the whole project
    html_paths = sorted(HTML_FOLDER.glob("*.html"))
    if not html_paths:
        html_paths = sorted((Path(__file__).parent / "su_orgs").glob("*.html"))
    if not html_paths:
        seen, html_paths = set(), []
        for p in sorted(Path(__file__).parent.rglob("*.html")):
            if p.name not in seen:            # dedupe by filename (in case of a nested copy)
                seen.add(p.name)
                html_paths.append(p)
    if not html_paths:
        st.error("No HTML files found. Put the org .html files in an 'html/' folder next to HW5.py.")
        st.stop()

    documents, ids, metadatas = [], [], []
    for path in html_paths:
        text = html_to_text(path)
        if not text:
            continue
        for i, chunk in enumerate(split_into_two(text)):   # two mini-docs per file
            documents.append(chunk[:MAX_CHARS_PER_CHUNK])
            ids.append(f"{path.name}#chunk{i}")            # unique id per chunk
            metadatas.append({"filename": path.name, "chunk": i})

    # add in batches of 100 so a big pile of files doesn't blow past request-size limits
    for start in range(0, len(documents), 100):
        collection.add(
            documents=documents[start:start + 100],
            ids=ids[start:start + 100],
            metadatas=metadatas[start:start + 100],
        )

    st.session_state._hw5_chroma_client = chroma_client
    return collection


# build once (disk guard above + session_state guard here so reruns are instant)
if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Building the org knowledge base (first run only, 500+ files, give it a minute)..."):
        st.session_state.HW5_VectorDB = build_hw5_vectordb(openai_api_key)

collection = st.session_state.HW5_VectorDB


# ------------------ the tool function (step 3a) ------------------
# this is the NEW idea for HW5. instead of me searching on every turn, the LLM calls this
# function with a 'query' IT chose, and i return the matching club info from ChromaDB.
def relevant_club_info(query):
    results = collection.query(query_texts=[query], n_results=5)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "No matching student organizations were found in the database."

    # glue the matches together into one text blob, tagging each with its file name
    info = ""
    for meta, doc in zip(metas, docs):
        info += f"\n--- from {meta['filename']} ---\n{doc[:2000]}\n"
    return info


# ------------------ describe the tool to the LLM ------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the Syracuse University student organization database and return "
                "information about clubs relevant to a query. Call this whenever the user "
                "asks about student organizations, clubs, or activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search query describing the clubs or info to look for, "
                                       "e.g. 'data science clubs' or 'Christian faith-based organizations'.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ------------------ chatbot with short-term memory (steps 2 & 4) ------------------
# one system prompt guides both calls: when to search, and how to answer once it has results.
SYSTEM_PROMPT = (
    "You are a helpful assistant for Syracuse University student organizations. "
    "When the user asks about clubs, organizations, or activities, call the relevant_club_info "
    "tool with a good search query and then answer using what it returns, naming the "
    "organization(s) you used. If the message is not about student orgs (like a greeting or a "
    "general question), just answer normally without searching. If the returned info does not "
    "cover the question, say you are answering from general knowledge."
)

# start the chat history in session_state so it survives reruns
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        {"role": "assistant", "content": "Hi! Ask me about any Syracuse student org."}
    ]

# show the conversation so far (chat interface)
for m in st.session_state.hw5_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("e.g. what clubs are there for data science?"):
    # save + show the user's message
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # short-term memory: keep only the last 5 interactions (5 q&a pairs = 10 messages).
    # i build a LOCAL working list for this turn so the tool plumbing never pollutes my
    # saved history (i only save the clean user + final answer at the end).
    buffered = st.session_state.hw5_messages[-(INTERACTIONS * 2):]
    working = [{"role": "system", "content": SYSTEM_PROMPT}] + buffered

    # CALL 1: offer the tool and let the model decide (tool_choice="auto")
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=working,
        tools=tools,
        tool_choice="auto",
    )
    msg = first.choices[0].message

    with st.chat_message("assistant"):
        if msg.tool_calls:
            # the model wants to search. add its request to the local list, then run the tool.
            working.append(msg)
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args.get("query") or prompt          # fall back to the raw prompt if empty
                info = relevant_club_info(query)             # step 3a: the vector search
                # hand the results back to the model as a tool message
                working.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": info,
                })

            # CALL 2 (step 3b): answer using the results, WITH NO TOOLS so it can't search
            # again. streamed like my other labs.
            second = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=working,
                stream=True,
            )
            answer = st.write_stream(second)
        else:
            # the model decided no search was needed and just answered directly
            answer = msg.content
            st.markdown(answer)

    # save only the clean final answer to memory (not the tool plumbing)
    st.session_state.hw5_messages.append({"role": "assistant", "content": answer})