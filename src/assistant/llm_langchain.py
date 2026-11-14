"""Le modèle de chat, version LangChain.

Avant (projet 1 et 2) : src/rag/llm.py créait un client Groq, et chaque appel s'écrivait
    client.chat.completions.create(model=MODELE, messages=[...], ...)
Maintenant : un objet « modèle de chat » LangChain, qui s'utilise avec .invoke(messages).
La clé GROQ_API_KEY est lue automatiquement dans l'environnement (fichier .env).
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.rag.llm import MODELE  # même modèle qu'avant : openai/gpt-oss-20b

load_dotenv()

modele = ChatGroq(model=MODELE, temperature=0)