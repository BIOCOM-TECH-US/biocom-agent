import os, glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()
KB_DIR = Path("knowledge")
OUT_DIR = Path("indexes/faiss")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load docs
docs = []
for fp in glob.glob(str(KB_DIR / "*.md")):
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read()
    docs.append(Document(page_content=text, metadata={"source": Path(fp).name}))

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120, separators=["\n## ", "\n### ", "\n", " "]
)
chunks = splitter.split_documents(docs)

# Embed + index
emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vs = FAISS.from_documents(chunks, emb)
vs.save_local(str(OUT_DIR))

print(f"Indexed {len(chunks)} chunks → {OUT_DIR}")