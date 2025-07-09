from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

def generate_report(chat_history):
    # Accès aux rapports exemples
    vectordb_exemples = Chroma(
        persist_directory="data/vectordb_exemples",
        embedding_function=OpenAIEmbeddings()
    )
    retriever_exemples = vectordb_exemples.as_retriever(search_kwargs={"k": 3})
    exemples = retriever_exemples.get_relevant_documents("structure de rapport de prévention incendie")
    exemples_text = "\n\n".join([doc.page_content for doc in exemples])

    # Accès aux documents réglementaires
    vectordb_reglement = Chroma(
        persist_directory="data/vectordb",
        embedding_function=OpenAIEmbeddings()
    )
    retriever_reglement = vectordb_reglement.as_retriever(search_kwargs={"k": 5})
    reglements = retriever_reglement.get_relevant_documents("base réglementaire pour rapport de prévention incendie")
    reglements_text = "\n\n".join([doc.page_content for doc in reglements])

    # Reformatage de l’historique utilisateur
    historique_text = "\n".join([
        f"Utilisateur : {q}\nAssistant : {r}" for q, r in chat_history
    ])

    # Prompt combiné
    prompt = f"""
Tu es un assistant expert chargé de rédiger un rapport de prévention incendie en Belgique.

Tu dois :
- Reprendre le style et la structure des rapports exemples fournis.
- Utiliser les références réglementaires disponibles dans les documents techniques.
- Construire un rapport clair, rigoureux, structuré, et adapté au contexte de l’échange avec l’utilisateur.

--- Exemples de rapports ---
{exemples_text}

--- Extraits de documents réglementaires ---
{reglements_text}

--- Dialogue avec l'utilisateur ---
{historique_text}

Rédige un rapport clair, professionnel, avec des titres et des sous-titres.
Si des informations sont manquantes, indique-les entre crochets : [À compléter].
Sois rigoureux.
"""

    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    rapport = llm.predict(prompt)

    return rapport
