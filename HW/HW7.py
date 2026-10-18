# HW 7 - News Information Bot
#
# this is a bot for a law firm that watches news about its clients. it answers ONLY from
# the articles in news.csv, never from outside knowledge. it does two main jobs:
#   - "find news about X"             -> a semantic search over the articles
#   - "find the most interesting news" -> a ranked list, where for a law firm "interesting"
#     means legally or financially significant (lawsuits, regulators, mergers, breaches,
#     leadership changes), with a one-line reason for each.
#
# how it's built, and why:
#   - RAG (ChromaDB vector search): there are 1,290 articles, far too many to fit in one
#     prompt, so i retrieve only the few that matter for each question.
#   - tool calling: the model picks between my two tools, same two-call loop as HW 5.
#   - prompt engineering: the law-firm definition of "interesting" and the grounding rules
#     live in the system prompt.
#   - the vector db is built once by build_news_db.py and just loaded here, so the user
#     never waits for embedding.

# sqlite fix for streamlit cloud - has to run BEFORE chromadb is imported (same as HW 4/5)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import csv
from pathlib import Path

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

#  settings 
HERE = Path(__file__).parent
CSV_PATH = HERE / "news.csv"
DB_PATH = str(HERE / "news_chroma")          # the folder build_news_db.py created
COLLECTION_NAME = "news_articles"
EMBED_MODEL = "text-embedding-3-small"        # openai embeddings model
BUFFER_MESSAGES = 6                           # short memory, like my other bots

# the two models i compare for step 4 (one cheap, one pricey). both worked in my labs.
# if my account has different names i just change them here.
MODELS = {
    "Lower cost (gpt-5-nano)": "gpt-5-nano",
    "Higher cost (gpt-4.1)": "gpt-4.1",
}

# this is my definition of "interesting" for a law firm. i use it as a search query to
# pull the significance-heavy articles as candidates to rank (see rank_interesting_news).
SIGNIFICANCE_QUERY = (
    "lawsuit litigation regulatory investigation antitrust fraud SEC probe fine "
    "settlement data breach cyberattack merger acquisition layoffs restructuring "
    "bankruptcy financial loss profit warning executive resignation CEO change scandal"
)

st.title("📰 HW 7 - Client News Monitor")
st.write("Ask about the news in the client set. Try \"what is the most interesting news?\" "
         "or \"find news about Nvidia.\"")

#  api key (from secrets, same as my other labs) 
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

#  sidebar: pick the model (this is how i compare two LLMs for step 4) 
model_label = st.sidebar.radio("Model", list(MODELS.keys()), index=0)
CHAT_MODEL = MODELS[model_label]
st.sidebar.caption("Switch models and re-ask the same question to compare.")


#  load the vector db once (built outside by build_news_db.py) 
# st.cache_resource runs this a single time per app process, so the user doesn't wait
# for it on every message. if the prebuilt db isn't there, i build it once from the csv.
@st.cache_resource(show_spinner="Loading the news database...")
def load_collection():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBED_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
    )
    if collection.count() == 0:      # empty means no prebuilt db, so build it now
        _build_from_csv(collection)
    return collection


def _build_from_csv(collection):
    # fallback builder, same logic as build_news_db.py, in case i didn't commit the
    # prebuilt db. runs once, then it's cached on disk.
    with st.spinner("First run: embedding the articles (one time only)..."):
        documents, ids, metadatas = [], [], []
        with open(CSV_PATH, encoding="utf-8", errors="replace") as f:
            for i, row in enumerate(csv.DictReader(f)):
                company = (row.get("company_name") or "").strip()
                doc = (row.get("Document") or "").strip()
                # split "Title Description: blurb" into title + description
                title, desc = (doc.split("Description:", 1) + [""])[:2] if "Description:" in doc else (doc, "")
                title, desc = title.strip(), desc.strip()
                if not title:
                    continue
                documents.append(f"{company} | {title}\n{desc}")   # company in the text so it matches
                ids.append(str(i))
                metadatas.append({"company": company, "date": (row.get("Date") or "")[:10],
                                  "url": (row.get("URL") or "").strip(), "title": title})
        for s in range(0, len(documents), 100):      # add in batches of 100
            collection.add(documents=documents[s:s + 100], ids=ids[s:s + 100],
                           metadatas=metadatas[s:s + 100])


collection = load_collection()


@st.cache_data
def known_companies():
    # the distinct company names, so i can match a user's typed company to the stored one
    got = collection.get(include=["metadatas"])
    return sorted({(m.get("company") or "") for m in got["metadatas"] if m.get("company")})


def match_company(name):
    # map a loosely typed company ("apple") to the exact stored name ("Apple"), else None
    if not name:
        return None
    name = name.strip().lower()
    for c in known_companies():          # try an exact match first
        if c.lower() == name:
            return c
    for c in known_companies():          # then a loose contains match
        if name in c.lower() or c.lower() in name:
            return c
    return None


def _format(results):
    # turn a chroma query result into a small list the model can read easily
    out = []
    metas = results["metadatas"][0]
    docs = results["documents"][0]
    for meta, doc in zip(metas, docs):
        snippet = doc.split("\n", 1)[-1][:300]   # the description part, trimmed
        out.append({
            "company": meta.get("company", ""),
            "date": meta.get("date", ""),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "snippet": snippet,
        })
    return out


#  the two tools the model can call 
def search_news(query, company=None, n=8):
    # semantic search for "find news about X". filters to one company if i can match it.
    matched = match_company(company) if company else None
    where = {"company": matched} if matched else None
    results = collection.query(query_texts=[query or SIGNIFICANCE_QUERY],
                               n_results=n, where=where)
    return _format(results)


def rank_interesting_news(focus=None, n=10):
    # candidates for ranking. if the user named a company, rank within it. otherwise pull
    # the most significance-heavy articles from the whole set using my significance query.
    # the model does the actual ranking in its answer (that's the "context for interesting").
    matched = match_company(focus) if focus else None
    where = {"company": matched} if matched else None
    query = focus if (focus and not matched) else SIGNIFICANCE_QUERY
    results = collection.query(query_texts=[query], n_results=max(n * 2, 20), where=where)
    return _format(results)


# tell the model what my two tools do
TOOLS = [
    {"type": "function", "function": {
        "name": "search_news",
        "description": "Search the client news for a topic or company. Use for 'find news about X'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for, e.g. 'AI chips' or 'Zelle lawsuit'."},
            "company": {"type": "string", "description": "Optional company name to filter to, e.g. 'Apple'."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "rank_interesting_news",
        "description": "Get candidate articles to rank as most interesting/important. Use for "
                       "'most interesting news' or 'biggest stories', optionally about one company.",
        "parameters": {"type": "object", "properties": {
            "focus": {"type": "string", "description": "Optional company or topic to focus the ranking on."},
        }}}},
]

TOOL_FUNCS = {"search_news": search_news, "rank_interesting_news": rank_interesting_news}

# the system prompt carries the law-firm lens and the grounding rules (prompt engineering)
SYSTEM_PROMPT = (
    "You are a news monitor for a global law firm. You report ONLY on the client articles "
    "returned by your tools, never from outside knowledge. For a topic or company request, "
    "call search_news. For 'most interesting' or 'biggest' news, call rank_interesting_news "
    "and then rank the articles, most important first. "
    "For a law firm, 'interesting' means legally or financially significant: lawsuits and "
    "litigation, regulators and investigations, mergers and acquisitions, data breaches, "
    "financial distress, and leadership changes. For each item give one line on why it matters, "
    "and cite the company, date, and source URL. If the tools return nothing relevant, say so "
    "plainly instead of inventing news. Keep answers easy to skim."
)

#  chat 
if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = [
        {"role": "assistant", "content": "Hi! Ask me what's most interesting, or about a specific client."}
    ]

# show the conversation so far
for m in st.session_state.hw7_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("e.g. what is the most interesting news this week?"):
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # short memory, then the system prompt first so it's never dropped
    buffered = st.session_state.hw7_messages[-BUFFER_MESSAGES:]
    working = [{"role": "system", "content": SYSTEM_PROMPT}] + buffered

    # CALL 1: let the model pick a tool and write its arguments
    first = client.chat.completions.create(
        model=CHAT_MODEL, messages=working, tools=TOOLS, tool_choice="auto",
    )
    msg = first.choices[0].message

    with st.chat_message("assistant"):
        if msg.tool_calls:
            working.append(msg)                      # record the tool request
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = TOOL_FUNCS[tc.function.name](**args)   # run my tool
                working.append({"role": "tool", "tool_call_id": tc.id,
                                "content": json.dumps(result)})  # hand results back
            # CALL 2: no tools, so it writes the answer from the articles and can't loop
            stream = client.chat.completions.create(
                model=CHAT_MODEL, messages=working, stream=True,
            )
            answer = st.write_stream(stream)
        else:
            # model didn't need a tool (e.g. a greeting), so just show what it said
            answer = msg.content or ""
            st.markdown(answer)

    st.session_state.hw7_messages.append({"role": "assistant", "content": answer})
