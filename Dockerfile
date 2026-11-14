# ============================================================================
# Image Docker de l'assistant de candidature
# Une seule image, deux usages (voir docker-compose.yml) : l'API et l'interface.
# ============================================================================

# 1. Point de départ : Python 3.12, version « slim » (Debian minimal, image plus légère)
FROM python:3.12-slim

# Pas de fichiers .pyc, et les print() s'affichent tout de suite dans les logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 2. PyTorch en version CPU, AVANT le reste.
#    Sinon, sentence-transformers installe la version GPU (CUDA) : plusieurs Go inutiles ici.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Les dépendances, AVANT le code : Docker garde cette couche en cache.
#    Tant que requirements.txt ne change pas, modifier le code ne réinstalle rien.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Télécharger le modèle d'embeddings PENDANT la construction (une fois pour toutes),
#    et pas au premier démarrage du conteneur.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# 5. Le code et les données (.dockerignore exclut .env, .venv, caches...)
COPY . .

# 6. Ne pas tourner en root : un utilisateur sans privilèges (bonne pratique de sécurité)
RUN useradd --create-home appli && chown -R appli /app
USER appli

EXPOSE 8000 8501

# Docker vérifie toutes les 30 s que l'API répond (statut « healthy » dans docker ps)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/sante')" || exit 1

# Commande par défaut : l'API. docker-compose.yml la remplace pour l'interface.
CMD ["uvicorn", "src.assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]