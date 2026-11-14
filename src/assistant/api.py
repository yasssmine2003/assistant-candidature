"""
Étape 10 : l'API REST de l'assistant (FastAPI).

Pourquoi une API en plus de l'interface Streamlit ?
L'interface sert une personne devant un navigateur. L'API sert des PROGRAMMES : une application
mobile, un site web en React, un autre service... Ils envoient du JSON et reçoivent du JSON.
Même logique (src/), deuxième « porte d'entrée ».

Lancement, depuis la racine de assistant-candidature :
    uvicorn src.assistant.api:app --reload
puis ouvre http://localhost:8000/docs : la documentation est générée automatiquement,
et tu peux tester chaque route depuis le navigateur (bouton « Try it out »).

Les routes (une ressource = une URL, une action = un verbe HTTP) :
    GET  /sante                              l'API répond-elle ?
    POST /sessions                           envoyer un CV -> une session (1 appel au LLM)
    GET  /sessions/{identifiant}/offres      classement des offres (aucun appel au LLM)
    POST /sessions/{identifiant}/questions   poser une question à l'agent
    POST /sessions/{identifiant}/lettres     rédiger une lettre (rédacteur + relecteur)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from groq import APIError
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.assistant.affichage import outils_appeles, tableau_offres
from src.assistant.candidature import evaluer_offre
from src.assistant.session import CVSuspect, Session, creer_session, obtenir_session

RACINE = Path(__file__).resolve().parent.parent.parent
RESSOURCES: dict = {}   # fiches, textes, moteur : chargés UNE fois au démarrage
AGENTS: dict = {}       # identifiant de session -> son agent LangGraph (qui garde la mémoire du fil)


# ----------------------------------------------------------------------------
# Démarrage : charger les ressources lourdes une seule fois (équivalent de @st.cache_resource)
# ----------------------------------------------------------------------------
@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.offres import charger_base
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    RESSOURCES["fiches"] = charger_base(RACINE / "data" / "offres.json")
    RESSOURCES["textes"] = {d.source: d.contenu for d in charger_documents(RACINE / "data" / "documents")}
    RESSOURCES["moteur"] = MoteurProtege(RechercheHybride(BaseVectorielle()))
    yield  # l'API tourne ici ; ce qui suivrait le yield s'exécuterait à l'arrêt


app = FastAPI(title="Assistant de candidature",
              description="Agent IA : analyse de CV, classement d'offres, questions, lettres relues.",
              version="1.0", lifespan=cycle_de_vie)


# ----------------------------------------------------------------------------
# Les schémas d'entrée et de sortie : FastAPI valide le JSON reçu avec Pydantic
# ----------------------------------------------------------------------------
class NouvelleSession(BaseModel):
    texte_cv: str = Field(min_length=50, description="Le texte complet du CV")


class SessionCreee(BaseModel):
    identifiant: str
    nom: str | None
    resume: str
    alertes_ancrage: list[str]


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class Reponse(BaseModel):
    reponse: str
    outils: list[str]


class DemandeLettre(BaseModel):
    source: str = Field(description="Nom du fichier de l'offre, par exemple offre_stage_devops.md")


class LettreRelue(BaseModel):
    lettre: str | None
    alertes: list[str]
    essais: int
    journal: list[str]


# ----------------------------------------------------------------------------
# Petites aides
# ----------------------------------------------------------------------------
def session_ou_404(identifiant: str) -> Session:
    """Une session inconnue n'est pas une erreur du serveur : c'est une ressource absente -> 404."""
    try:
        return obtenir_session(identifiant)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session inconnue : {identifiant}")


def creer_agent(session: Session):
    from src.assistant.agent_graphe import construire_agent
    from src.assistant.candidature import creer_outils_candidature
    from src.assistant.lettre import creer_outil_lettre
    from src.assistant.outils_langchain import creer_outils

    fiches, textes, moteur = RESSOURCES["fiches"], RESSOURCES["textes"], RESSOURCES["moteur"]
    outils = (creer_outils(moteur, fiches) + creer_outils_candidature(session, fiches)
              + [creer_outil_lettre(session, fiches, moteur, textes)])
    return construire_agent(outils, session)


# ----------------------------------------------------------------------------
# Les routes
# ----------------------------------------------------------------------------
@app.get("/sante")
def sante() -> dict:
    return {"statut": "ok", "offres": len(RESSOURCES.get("fiches", []))}


@app.post("/sessions", status_code=201, response_model=SessionCreee)
def nouvelle_session(demande: NouvelleSession):
    session = creer_session()
    try:
        session.charger_cv(demande.texte_cv)
    except CVSuspect as erreur:
        # 422 : la requête est bien formée, mais son contenu est refusé
        raise HTTPException(status_code=422, detail=str(erreur))
    except APIError as erreur:
        # 503 : c'est un service extérieur (Groq) qui ne répond pas, pas la faute du client
        raise HTTPException(status_code=503, detail=f"LLM indisponible : {erreur}")
    AGENTS[session.identifiant] = creer_agent(session)
    return SessionCreee(identifiant=session.identifiant, nom=session.cv.nom,
                        resume=session.resume_cv(), alertes_ancrage=session.alertes_ancrage)


# ============================================================================
# À TOI DE JOUER : la route du classement des offres
# ============================================================================
@app.get("/sessions/{identifiant}/offres")
def classement_offres(identifiant: str) -> list[dict]:
    """Classe les offres pour le CV de cette session, de la plus adaptée à la moins adaptée.

    Étapes (aucun appel au LLM) :
    1. session = session_ou_404(identifiant)          (404 si la session n'existe pas)
    2. evaluations = une évaluation par fiche :
           evaluer_offre(session.cv.competences, fiche)  pour fiche dans RESSOURCES["fiches"]
    3. lignes = tableau_offres(evaluations)            (déjà triées, voir affichage.py)
    4. renvoie les lignes, en gardant la clé "_source" sous le nom "source" et en retirant "_taux"
       (le client a besoin du nom de fichier pour demander une lettre ensuite).

    Exemple de ligne renvoyée :
        {"Entreprise": "BanqueNord", "Poste": "Data Scientist", "Correspondance": "100%",
         "Exigences": "3/3", "Manquantes": "aucune", "source": "offre_stage_data_science.md"}
    """
    session=session_ou_404(identifiant)
    evaluations=[evaluer_offre(session.cv.competences, fiche) for fiche in RESSOURCES["fiches"]]
    lignes=tableau_offres(evaluations)
    for ligne in lignes:
        ligne["source"]=ligne.pop("_source")
        del ligne["_taux"]
    return lignes


@app.post("/sessions/{identifiant}/questions", response_model=Reponse)
def poser_question(identifiant: str, demande: Question):
    session = session_ou_404(identifiant)
    agent = AGENTS[session.identifiant]
    config = {"configurable": {"thread_id": session.identifiant}, "recursion_limit": 12}
    avant = len(agent.get_state(config).values.get("messages", []))
    try:
        resultat = agent.invoke({"messages": [HumanMessage(demande.question)]}, config)
    except APIError as erreur:
        raise HTTPException(status_code=503, detail=f"LLM indisponible : {erreur}")
    return Reponse(reponse=resultat["messages"][-1].content,
                   outils=outils_appeles(resultat["messages"][avant:]))


@app.post("/sessions/{identifiant}/lettres", response_model=LettreRelue)
def rediger_lettre(identifiant: str, demande: DemandeLettre):
    from src.assistant.lettre import mettre_en_forme
    from src.assistant.multi_agents import construire_relecture

    session = session_ou_404(identifiant)
    if demande.source not in RESSOURCES["textes"]:
        raise HTTPException(status_code=404, detail=f"Offre inconnue : {demande.source}")
    graphe = construire_relecture(session, RESSOURCES["fiches"], RESSOURCES["moteur"], RESSOURCES["textes"])
    etat = graphe.invoke({"source": demande.source, "lettre": None, "contexte": "", "alertes": [],
                          "corrections": [], "essais": 0, "journal": []})
    texte = mettre_en_forme(etat["lettre"], session.cv.nom) if etat["lettre"] else None
    return LettreRelue(lettre=texte, alertes=etat["alertes"], essais=etat["essais"], journal=etat["journal"])