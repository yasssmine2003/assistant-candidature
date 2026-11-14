"""
Étape 9 : l'interface web de l'assistant de candidature (Streamlit).

Lancement, depuis la racine de assistant-candidature :
    streamlit run app.py
puis ouvre http://localhost:8501 dans le navigateur.

Comment marche Streamlit (à retenir) :
- Le script est RELANCÉ EN ENTIER de haut en bas à chaque clic ou message.
- Donc tout ce qui doit survivre entre deux clics va dans st.session_state (un dictionnaire),
  et tout ce qui est long à créer (connexion Qdrant, modèle d'embeddings) est mis en cache
  avec @st.cache_resource : créé une seule fois, réutilisé ensuite.
- Ce fichier ne contient que de l'AFFICHAGE : toute la logique est dans src/ (déjà testée).
"""

from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from src.assistant.affichage import outils_appeles, tableau_offres

RACINE = Path(__file__).resolve().parent
CV_EXEMPLE = RACINE / "data" / "exemples" / "cv_exemple.txt"

st.set_page_config(page_title="Assistant candidature", page_icon="🎯", layout="wide")


# ----------------------------------------------------------------------------
# Ressources partagées : créées UNE fois pour toute l'application
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connexion à Qdrant Cloud et chargement du modèle d'embeddings...")
def ressources():
    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.offres import charger_base
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    fiches = charger_base(RACINE / "data" / "offres.json")
    textes = {d.source: d.contenu for d in charger_documents(RACINE / "data" / "documents")}
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    return fiches, textes, moteur


def analyser_cv(texte: str) -> None:
    """Crée une session pour ce CV et l'agent qui va avec. Remet la conversation à zéro."""
    from src.assistant.agent_graphe import construire_agent
    from src.assistant.candidature import creer_outils_candidature
    from src.assistant.lettre import creer_outil_lettre
    from src.assistant.outils_langchain import creer_outils
    from src.assistant.session import CVSuspect, creer_session

    fiches, textes, moteur = ressources()
    session = creer_session()
    try:
        with st.spinner("Lecture du CV par le LLM..."):
            session.charger_cv(texte)
    except CVSuspect as erreur:
        st.sidebar.error(f"CV refusé : {erreur}")
        return
    except Exception as erreur:  # Groq indisponible, limite de débit... : on l'affiche, sans planter
        st.sidebar.error(f"Impossible d'analyser le CV : {erreur}")
        return
    outils = (creer_outils(moteur, fiches) + creer_outils_candidature(session, fiches)
              + [creer_outil_lettre(session, fiches, moteur, textes)])
    st.session_state.session = session
    st.session_state.agent = construire_agent(outils, session)
    st.session_state.conversation = []  # [(rôle, texte, outils utilisés)]
    st.session_state.lettre = None


# ----------------------------------------------------------------------------
# Barre latérale : le CV
# ----------------------------------------------------------------------------
st.sidebar.title("🎯 Assistant candidature")
choix = st.sidebar.radio("CV à analyser", ["CV d'exemple", "Mon CV (fichier .txt)"])
texte_cv = None
if choix == "CV d'exemple":
    texte_cv = CV_EXEMPLE.read_text(encoding="utf-8")
else:
    fichier = st.sidebar.file_uploader("Ton CV", type=["txt"])
    if fichier is not None:
        texte_cv = fichier.read().decode("utf-8", errors="replace")

if st.sidebar.button("Analyser le CV", type="primary", disabled=texte_cv is None):
    analyser_cv(texte_cv)

session = st.session_state.get("session")
if session is not None:
    st.sidebar.success(f"CV chargé : {session.cv.nom}")
    st.sidebar.markdown(session.resume_cv().replace("\n", "  \n"))
    for alerte in session.alertes_ancrage or []:
        st.sidebar.warning(f"À vérifier : {alerte}")
st.sidebar.caption("Offre gratuite de Groq : 8 000 tokens par minute. "
                   "En cas d'erreur 429, attends une minute.")

if session is None:
    st.title("Assistant de candidature")
    st.info("Commence par analyser un CV dans la barre de gauche.")
    st.stop()  # rien d'autre à afficher tant qu'il n'y a pas de CV

fiches, textes, moteur = ressources()
onglet_offres, onglet_chat, onglet_lettre = st.tabs(["🎯 Offres", "💬 Assistant", "✉️ Lettre de motivation"])

# ----------------------------------------------------------------------------
# Onglet 1 : le classement des offres (aucun appel au LLM : tout est calculé dans le code)
# ----------------------------------------------------------------------------
with onglet_offres:
    from src.assistant.candidature import evaluer_offre

    st.subheader("Les offres, de la plus adaptée à la moins adaptée")
    lignes = tableau_offres([evaluer_offre(session.cv.competences, f) for f in fiches])
    st.dataframe([{k: v for k, v in l.items() if not k.startswith("_")} for l in lignes],
                 use_container_width=True, hide_index=True)
    st.caption("Calculé dans le code à partir des fiches structurées (étape 1) : aucun appel au LLM.")

# ----------------------------------------------------------------------------
# Onglet 2 : discuter avec l'agent LangGraph (étape 5), qui garde la mémoire du fil
# ----------------------------------------------------------------------------
with onglet_chat:
    for role, texte, outils in st.session_state.conversation:
        with st.chat_message(role):
            st.markdown(texte.replace("<br>", " "))
            if outils:
                st.caption("Outils utilisés : " + " → ".join(outils))

    question = st.chat_input("Ex. : Quels stages me correspondent ? Qu'est-ce qui me manque pour Nexalia ?")
    if question:
        st.session_state.conversation.append(("user", question, []))
        with st.chat_message("user"):
            st.markdown(question)
        agent = st.session_state.agent
        config = {"configurable": {"thread_id": session.identifiant}, "recursion_limit": 12}
        avant = len(agent.get_state(config).values.get("messages", []))
        with st.chat_message("assistant"):
            try:
                with st.spinner("L'agent réfléchit..."):
                    resultat = agent.invoke({"messages": [HumanMessage(question)]}, config)
                nouveaux = resultat["messages"][avant:]
                reponse, outils = resultat["messages"][-1].content, outils_appeles(nouveaux)
            except Exception as erreur:
                reponse, outils = f"Désolée, une erreur est survenue : {erreur}", []
            st.markdown(reponse)
            if outils:
                st.caption("Outils utilisés : " + " → ".join(outils))
        st.rerun()  # réaffiche toute la conversation au-dessus de la zone de saisie

# ----------------------------------------------------------------------------
# Onglet 3 : la lettre, rédigée et relue par les deux agents (étape 7)
# ----------------------------------------------------------------------------
with onglet_lettre:
    from src.assistant.lettre import mettre_en_forme
    from src.assistant.multi_agents import construire_relecture

    noms = {f.source: f"{f.entreprise} - {f.poste}" for f in fiches}
    source = st.selectbox("Offre visée", list(noms), format_func=noms.get)
    if st.button("Rédiger la lettre (rédacteur + relecteur)", type="primary"):
        graphe = construire_relecture(session, fiches, moteur, textes)
        try:
            with st.spinner("Rédaction puis relecture (de 2 à 6 appels au LLM, jusqu'à une minute)..."):
                etat = graphe.invoke({"source": source, "lettre": None, "contexte": "", "alertes": [],
                                      "corrections": [], "essais": 0, "journal": []})
            st.session_state.lettre = {"source": source, "etat": etat}
        except Exception as erreur:
            st.error(f"La rédaction a échoué : {erreur}")

    resultat = st.session_state.lettre
    if resultat and resultat["source"] == source and resultat["etat"]["lettre"]:
        etat = resultat["etat"]
        if etat["alertes"]:
            st.warning("Alertes restantes, à corriger toi-même :\n- " + "\n- ".join(etat["alertes"]))
        else:
            st.success(f"Lettre approuvée par le relecteur en {etat['essais']} rédaction(s).")
        # Une zone modifiable : l'étudiante garde le dernier mot sur SA lettre
        texte_lettre = st.text_area("Ta lettre (modifiable)", mettre_en_forme(etat["lettre"], session.cv.nom),
                                    height=480)
        st.download_button("Télécharger (.txt)", texte_lettre, file_name=f"lettre_{source.replace('.md', '')}.txt")
        with st.expander("Journal des deux agents"):
            st.text("\n".join(etat["journal"]))