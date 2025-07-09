from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import json
import streamlit as st
import supabase
from supabase import create_client
import uuid
import tempfile

api_key = st.secrets["openai_api_key"]


url = st.secrets["supabase_url"]
key = st.secrets["supabase_key"]
supabase = create_client(url, key)

def upload_pdf_to_supabase(file):
    existing = supabase.table("documents")\
        .select("id")\
        .eq("original_name", file.name)\
        .execute()

    if existing.data:
        return 0
    
    filename = f"{uuid.uuid4()}.pdf"
    supabase.storage().from_("documents").upload(filename, file, {"content-type": "application/pdf"})
    supabase.table("documents").insert({"filename": filename}).execute()
    return 1

def get_new_documents(database):
    response = supabase.table(database).select("*").eq("vectorized", False).execute()
    return response.data

def ingest_documents(doc_folder="data/documents", db_folder="data/vectordb"):
    
    new_docs = get_new_documents("documents")

    if not new_docs:
        return -1
    
    documents = []
    for doc in new_docs:
        filename = doc["filename"]
        # Télécharger dans un fichier temporaire
        file_content = supabase.storage().from_("documents").download(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            loader = PyMuPDFLoader(tmp_file.name)
            docs = loader.load()
            documents.extend(docs)

    # Split et embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        persist_directory=db_folder,
    )
    vectordb.persist()
    
    # Mettre à jour la base Supabase
    for doc in new_docs:
        supabase.table("documents").update({"vectorized": True}).eq("filename", doc["filename"]).execute()

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
