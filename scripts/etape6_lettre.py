"""
Projet 3, étape 6 : rédiger une lettre de motivation ancrée.

Ce script :
0. vérifie TA fonction chiffres_inventes (sans LLM) ;
1. prépare la session avec le CV de Yasmine (1 appel au LLM) ;
2. appelle directement l'outil rediger_lettre pour Nexalia (1 appel) : la lettre et ses vérifications ;
3. demande la même chose à l'agent LangGraph, en langage naturel (2 à 3 appels).

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape6_lettre.py
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.assistant.lettre import chiffres_inventes


def verifier_ma_fonction() -> None:
    assert chiffres_inventes("6 mois, 1000 TND, +20 %", "Durée : 6 mois. 1000 TND. Réf 2027") == ["20"], \
        "« 20 » n'est dans aucune source"
    assert chiffres_inventes("En 2026, 20 %", "Année 2026") == ["20"], \
        "Comparer des nombres entiers : « 20 » n'est pas « 2026 »"
    assert chiffres_inventes("3 projets et 3 stages, 5 ans", "rien") == ["3", "5"], \
        "Sans doublon, dans l'ordre d'apparition"
    assert chiffres_inventes("Aucun nombre ici", "1000") == [], "Pas de nombre : liste vide"
    print("Ta fonction chiffres_inventes passe les 4 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    from langchain_core.messages import HumanMessage

    from src.assistant.agent_graphe import construire_agent
    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.candidature import creer_outils_candidature
    from src.assistant.lettre import creer_outil_lettre
    from src.assistant.offres import charger_base
    from src.assistant.outils_langchain import creer_outils
    from src.assistant.session import creer_session
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    print("=" * 70)
    print("1. Préparation (1 appel au LLM pour extraire le CV)")
    print("=" * 70)
    fiches = charger_base(RACINE / "data" / "offres.json")
    textes_offres = {d.source: d.contenu for d in charger_documents(RACINE / "data" / "documents")}
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    session = creer_session()
    session.charger_cv((RACINE / "data" / "exemples" / "cv_exemple.txt").read_text(encoding="utf-8"))
    rediger_lettre = creer_outil_lettre(session, fiches, moteur, textes_offres)
    print(f"Session {session.identifiant} : {session.cv.nom}\n")

    print("=" * 70)
    print("2. L'outil seul : lettre pour Nexalia (1 appel)")
    print("=" * 70)
    print(rediger_lettre.invoke({"source": "offre_stage_ia_generative.md"}))
    print()

    print("=" * 70)
    print("3. Par l'agent LangGraph, en langage naturel")
    print("=" * 70)
    outils = creer_outils(moteur, fiches) + creer_outils_candidature(session, fiches) + [rediger_lettre]
    agent = construire_agent(outils, session)
    config = {"configurable": {"thread_id": session.identifiant}, "recursion_limit": 12}
    resultat = agent.invoke({"messages": [HumanMessage("Écris-moi une lettre de motivation pour le stage de TelcoMed.")]}, config)
    for m in resultat["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"   -> outil {tc['name']}({tc['args']})")
    print(resultat["messages"][-1].content)


if __name__ == "__main__":
    main()