import streamlit as st
from ingestion import ingest_documents, ingest_report_examples
from chat import get_qa_chain
from report import generate_report
import os


st.set_page_config(page_title="FireGPT - Prévention Incendie", layout="wide")
st.title("🔥 Assistant Prévention Incendie Belgique")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.sidebar.header("📁 Ajouter des documents")

# 1. Upload de fichiers
uploaded_files = st.sidebar.file_uploader(
    "Ajoute un ou plusieurs fichiers PDF",
    type=["pdf"],
    accept_multiple_files=True
)

# 2. Sauvegarde dans data/documents/
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("data/documents", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"{len(uploaded_files)} fichier(s) ajouté(s).")
    
    
if st.sidebar.button("Intégration des PDF"):
    if ingest_documents() == 0:
        st.sidebar.success("Nouveaux documents intégrés !")
    else:
        st.sidebar.warning("Ces documents sont déjà intégrés")
        
        
st.sidebar.header("📄 Ajouter des exemples de rapports")
uploaded_examples = st.sidebar.file_uploader(
    "Ajoute un ou plusieurs rapports types (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    key="upload_examples"
)

example_dir = "data/rapports_exemples"
os.makedirs(example_dir, exist_ok=True)

if uploaded_examples:
    for file in uploaded_examples:
        path = os.path.join(example_dir, file.name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(file.read())
    st.sidebar.success(f"{len(uploaded_examples)} rapport(s) ajouté(s).")
    
if st.sidebar.button("Intégrer les rapports types"):
    from ingestion import ingest_report_examples
    if ingest_report_examples() == 0:
        st.sidebar.success("Rapports exemples intégrés !")
    else:
        st.sidebar.warning("Ces rapports sont déjà intégrés")    


if "data_ingested" not in st.session_state:
    with st.spinner("Chargement et intégration des documents..."):
        ingest_documents()
        ingest_report_examples()
        st.session_state["data_ingested"] = True
qa_chain = get_qa_chain()

st.subheader("💬 Dialogue avec FireGPT")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_qa_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Pose ta question...")

if user_input:
    with st.spinner("Analyse en cours..."):
        response = st.session_state.qa_chain.run({"question": user_input})
        st.session_state.chat_history.append(("👤", user_input))
        st.session_state.chat_history.append(("🤖", response))

for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)


if any("Je suis prêt à rédiger le rapport" in msg for _, msg in st.session_state.chat_history):
    st.sidebar.markdown("✅ L'assistant est prêt à générer le rapport.")
    if st.sidebar.button("📄 Générer rapport"):
        final_report = generate_report(st.session_state.chat_history)
        st.download_button("📥 Télécharger le rapport", final_report, file_name="rapport_prevention.md")