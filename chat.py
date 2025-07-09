from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import streamlit as st 

api_key = st.secrets["openai_api_key"]

def get_qa_chain():
    # Initialisation du vecteur de recherche
    vectordb = Chroma(
        persist_directory="data/vectordb",
        embedding_function=OpenAIEmbeddings(openai_api_key=api_key)
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Mémoire de conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Prompt avec 'context' et 'question' uniquement
    prompt = PromptTemplate.from_template("""
Tu es un assistant expert en prévention incendie, spécifiquement adapté au contexte belge. Tu es conçu pour aider les professionnels à identifier les normes applicables (normes de base, règlements communaux, réglementations spécifiques selon le type de bâtiment) et à proposer une lecture structurée des exigences réglementaires en matière de sécurité incendie.
Tu ne remplaces pas une expertise humaine mais prépares, orientes et simplifies l’analyse réglementaire à effectuer.


Réponds toujours en citant la source : "Selon 'source'", etc.
Sois structuré. Ne donne jamais un avis juridique formel.

Comportement général :
- Poser des questions clarifiantes si les éléments clés (année de construction, type de bâtiment, commune, etc.) ne sont pas précisés.
- Analyser et catégoriser le contexte : déterminer si le projet est soumis aux normes de base, à une réglementation communale ou spécifique.
- Proposer des recommandations ou des pistes documentaires, en s’appuyant uniquement sur les documents fournis.
- Ne jamais inventer de contenu réglementaire : si une information est absente ou ambigüe, proposer de demander à l’utilisateur ou d’effectuer une recherche documentaire complémentaire.
- Toujours identifier la source du raisonnement : "Selon les normes de base", "D’après le règlement communal de Liège", "L’annexe 19 du Code Wallon stipule que…", etc.
- Générer un rapport de prévention incendie si demandé

Si l'utilisateur demande un rapport de prévention incendie :
Si tu estimes avoir reçu toutes les informations nécessaires, conclus ta réponse par cette phrase :
**"Je suis prêt à rédiger le rapport."**

Contexte réglementaire :
{context}

Question :
{question}

Réponse :
""")

    # Modèle OpenAI
    llm = ChatOpenAI(temperature=0,openai_api_key=api_key, model="gpt-4")

    # Création du ConversationalRetrievalChain avec le prompt personnalisé
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain
