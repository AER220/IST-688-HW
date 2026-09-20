# sqlite fix (same as lab 4) - chroma needs a newer sqlite than streamlit cloud ships.
# this swaps in pysqlite3 BEFORE chromadb loads. try/except so it still runs where it's not installed.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from pathlib import Path

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

#  settings 
HTML_FOLDER = Path(__file__).parent / "html"                    # the org html files live in here
CHROMA_PATH = str(Path(__file__).parent / "ChromaDB_for_HW4")   # the vector db is SAVED TO DISK here
COLLECTION_NAME = "HW4Collection"
EMBED_MODEL = "text-embedding-3-small"          # openai embeddings model for the vectors
CHAT_MODEL = "gpt-5-mini"                        # same llm as lab 4 / HW3
INTERACTIONS = 5                                 # keep the last 5 interactions (step 3a)
MAX_CHARS_PER_CHUNK = 8000                       # safety cap per chunk so i stay under the token limit

st.title(" HW 4- Syracuse Student Orgs Chatbot (RAG)")

#  api key (from secrets, same as HW3/lab4) 
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# ------------------ chunking (step 2a) ------------------
# CHUNKING METHOD: i split each org page into TWO halves, cutting at the nearest space
# to the midpoint so i never split a word in half.
# WHY THIS METHOD: the homework asks for exactly 2 mini-docs per file, so a simple
# split-in-half fits that. each org page is short and about a single club, so two halves
# still keep related info together, but they give the retriever smaller, tighter pieces to
# match a question against - smaller chunks = more focused embeddings = sharper similarity
# matches than embedding one big blob. it's also simple, deterministic and cheap to run.
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


# ------------------ build the vector db (step 2) ------------------
def build_hw4_vectordb(api_key):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBED_MODEL,
    )

    # PersistentClient writes the db to disk so it survives app restarts (step 2b)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
    )

    # step 2b: only build it if it's EMPTY. if it already has data, just reuse it -
    # this is what lets me run the app over and over but only embed everything once.
    if collection.count() > 0:
        st.session_state._hw4_chroma_client = chroma_client
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
        st.error("No HTML files found. Put the org .html files in an 'html/' folder next to HW4.py.")
        st.stop()

    documents, ids, metadatas = [], [], []
    for path in html_paths:
        text = html_to_text(path)
        if not text:
            continue
        for i, chunk in enumerate(split_into_two(text)):   # two mini-docs per file (step 2a)
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

    st.session_state._hw4_chroma_client = chroma_client
    return collection


# build once (disk guard above + session_state guard here so reruns are instant)
if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Building the org knowledge base (first run only — 500+ files, give it a minute)…"):
        st.session_state.HW4_VectorDB = build_hw4_vectordb(openai_api_key)

collection = st.session_state.HW4_VectorDB


# ------------------ chatbot (steps 3 & 4) ------------------
st.subheader("Ask about Syracuse student organizations")

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {"role": "assistant", "content": "Hi! Ask me about any Syracuse student org."}
    ]

# show the conversation so far (step 4 - chat interface)
for msg in st.session_state.hw4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g. what clubs are there for data science?"):
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # step 3b: search the vector db for the most relevant chunks for this question
    results = collection.query(query_texts=[prompt], n_results=5)
    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    # glue the retrieved chunks together to feed into the prompt
    context = ""
    for meta, doc_text in zip(retrieved_meta, retrieved_docs):
        context += f"\n--- from {meta['filename']} ---\n{doc_text[:2000]}\n"

    # prompt engineering: hand the model the retrieved orgs and tell it to say when it's
    # using them (name the org/file) vs answering from general knowledge
    system_prompt = (
        "You are a helpful assistant for Syracuse University student organizations. "
        "Use the ORG DOCUMENTS below to answer the question. If you use them, clearly say so "
        "and name the organization(s)/file(s). If the answer isn't in the documents, say you "
        "are answering from general knowledge instead.\n\n"
        f"ORG DOCUMENTS:\n{context}"
    )

    # step 3a: keep only the last 5 interactions (5 q&a pairs = 10 messages), then put the
    # fresh system prompt first so the context is never dropped
    buffered = st.session_state.hw4_messages[-(INTERACTIONS * 2):]
    api_messages = [{"role": "system", "content": system_prompt}] + buffered

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=CHAT_MODEL, messages=api_messages, stream=True,
        )
        answer = st.write_stream(stream)

    st.session_state.hw4_messages.append({"role": "assistant", "content": answer})