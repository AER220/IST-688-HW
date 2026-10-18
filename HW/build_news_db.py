# build_news_db.py
# HW 7 - this builds my news vector database ONCE, outside the app.
# the assignment suggested doing it this way so the app just loads the db at startup
# instead of re-embedding 1,290 articles every time, which would be slow for the user.
#
# i run this one time from the terminal:   python build_news_db.py
# it reads news.csv, embeds every article with openai, and writes the ./news_chroma folder.
# it needs my OPENAI_API_KEY, either in the environment or in .streamlit/secrets.toml.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
import os
import csv
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# where things live. i keep the csv and the db folder next to this script.
HERE = Path(__file__).parent
CSV_PATH = HERE / "news.csv"
DB_PATH = str(HERE / "news_chroma")          # the folder my app loads later
COLLECTION_NAME = "news_articles"
EMBED_MODEL = "text-embedding-3-small"        # openai embeddings model for the vectors


def get_api_key():
    # check the environment first, then fall back to my .streamlit/secrets.toml file
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    secrets = HERE / ".streamlit" / "secrets.toml"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No OPENAI_API_KEY found (set the env var or add it to .streamlit/secrets.toml).")


def split_document(document):
    # the Document column looks like "Title Description: blurb", so i split it into the
    # two parts on the word "Description:".
    document = (document or "").strip()
    if "Description:" in document:
        title, desc = document.split("Description:", 1)
        return title.strip(), desc.strip()
    return document, ""


def read_articles():
    # read every row of the csv into a clean little dict, one per article
    rows = []
    with open(CSV_PATH, encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f)):
            company = (row.get("company_name") or "").strip()   # strip trailing spaces
            title, desc = split_document(row.get("Document"))
            if not title:
                continue
            rows.append({
                "id": str(i),
                "company": company,
                "date": (row.get("Date") or "")[:10],   # keep just YYYY-MM-DD
                "url": (row.get("URL") or "").strip(),
                "title": title,
                "description": desc,
            })
    return rows


def build():
    api_key = get_api_key()

    # chroma will call openai to make the embeddings for me on add() and query()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=DB_PATH)   # writes the db to disk
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
    )

    # if it already has data, don't pay to embed everything again
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} articles. Nothing to do.")
        return

    articles = read_articles()
    print(f"Embedding {len(articles)} articles...")

    documents, ids, metadatas = [], [], []
    for a in articles:
        # i put the company name into the embedded text so a search like "news about Apple"
        # still matches even when the headline doesn't repeat the company name.
        documents.append(f"{a['company']} | {a['title']}\n{a['description']}")
        ids.append(a["id"])                              # row number as the key
        metadatas.append({                               # keep these for filtering + citing
            "company": a["company"], "date": a["date"],
            "url": a["url"], "title": a["title"],
        })

    # add in batches of 100 so i don't blow past the request-size limit
    for start in range(0, len(documents), 100):
        collection.add(
            documents=documents[start:start + 100],
            ids=ids[start:start + 100],
            metadatas=metadatas[start:start + 100],
        )
        print(f"  added {min(start + 100, len(documents))}/{len(documents)}")

    print(f"Done. Vector DB written to {DB_PATH}")


if __name__ == "__main__":
    build()
