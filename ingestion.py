from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import json
import streamlit as st

api_key = st.secrets["openai_api_key"]

def ingest_documents(doc_folder="data/documents", db_folder="data/vectordb"):
    
    index_file = os.path.join(doc_folder, "indexed_files.json")
    indexed = set()

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            indexed = set(json.load(f))

    new_files = [f for f in os.listdir(doc_folder)
                 if f.endswith(".pdf") and f not in indexed]

    if not new_files:
        return -1
    documents = []
    
    for filename in new_files:
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(doc_folder, filename))
            docs = loader.load()
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        persist_directory=db_folder,
    )
    vectordb.persist()
    
    indexed.update(new_files)
    with open(index_file, "w") as f:
        json.dump(list(indexed), f)
    return 0


def ingest_report_examples(folder="data/rapports_exemples", db_folder="data/vectordb_exemples"):
    index_file = os.path.join(folder, "indexed_examples.json")
    indexed = set()

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            indexed = set(json.load(f))

    new_files = [f for f in os.listdir(folder)
                 if f.endswith(".pdf") and f not in indexed]

    if not new_files:
        return -1

    documents = []
    for filename in new_files:
        loader = PyMuPDFLoader(os.path.join(folder, filename))
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        persist_directory=db_folder
    )
    vectordb.persist()

    indexed.update(new_files)
    with open(index_file, "w") as f:
        json.dump(list(indexed), f)

    return 0
