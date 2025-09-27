import os, warnings
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# load env before any client init
load_dotenv()

from strands import tool
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document

INDEX_DIR = Path("indexes/faiss")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment (.env).")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)

if not INDEX_DIR.exists():
    raise RuntimeError(f"FAISS index not found at {INDEX_DIR}. Build it with: python src/build_index.py")

_vectorstore = FAISS.load_local(
    str(INDEX_DIR),
    _embeddings,
    allow_dangerous_deserialization=True
)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

def _compact(text: str, limit=420) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "…"

@tool
def kb_search(query: str, k: int = 4) -> List[str]:
    """
    Retrieve up to k Biocom knowledge snippets from the FAISS index.
    Returns compact, bracket-cited snippets like: [leadership_team.md] ...
    """
    docs = _retriever.invoke(query)[:k]   # new API
    out = []
    for d in docs:
        src = d.metadata.get("source", "knowledge")
        out.append(f"[{src}] {_compact(d.page_content)}")
    return out