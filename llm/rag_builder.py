import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config.constants import IGNORE_DIRS, SUPPORTED_EXTENSIONS



def build_vector_store():
    docs = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            path = os.path.join(root, fname)
            if os.path.splitext(fname)[1] not in SUPPORTED_EXTENSIONS:
                continue
            try:
                docs += TextLoader(path).load()
            except:
                pass

    if not docs: return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, emb)
