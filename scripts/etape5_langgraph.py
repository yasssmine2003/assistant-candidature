"""
Projet 3, étape 5 : l'agent orchestrateur avec LangGraph.

Ce script :
0. vérifie TA fonction classer (sans LLM) ;
1. prépare tout : moteur Qdrant, fiches, session avec le CV de Yasmine (1 appel au LLM), outils ;
2. affiche le graphe de l'agent (à coller sur mermaid.live pour le voir dessiné) ;
3. mène une conversation de 3 questions, dans le MÊME fil : l'agent se souvient ;
4. pose une question floue dans un NOUVEAU fil : l'agent n'a pas le contexte.

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape5_langgraph.py
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.assistant.candidature import classer


def verifier_ma_fonction() -> None:
    evaluations = [
        {"source": "a.md", "entreprise": "Alpha", "poste": "Dev", "satisfaites": ["Python"], "manquantes": ["SQL", "Git"], "total": 3},
        {"source": "b.md", "entreprise": "Beta", "poste": "Data", "satisfaites": ["Python", "SQL"], "manquantes": [], "total": 2},
        {"source": "c.md", "entreprise": "Gamma", "poste": "Vide", "satisfaites": [], "manquantes": [], "total": 0},
    ]
    attendu = ("1. Beta (Data) : 100 % des exigences (2/2) [b.md]\n"
               "2. Alpha (Dev) : 33 % des exigences (1/3) [a.md]\n"
               "3. Gamma (Vide) : 0 % des exigences (0/0) [c.md]")
    obtenu = classer(evaluations)
    assert obtenu == attendu, f"\n  attendu :\n{attendu}\n  obtenu :\n{obtenu}"
    assert classer([]) == "Aucune offre à classer.", "Aucune évaluation : message prévu"
    print("Ta fonction classer passe les 2 tests.\n")


def poser(agent, question: str, fil: str) -> None:
    """Pose une question dans un fil de conversation, et affiche ce que l'agent a fait."""
    config = {"configurable": {"thread_id": fil}, "recursion_limit": 12}  # 12 = garde-fou (MAX_TOURS)
    avant = len(agent.get_state(config).values.get("messages", []))
    resultat = agent.invoke({"messages": [HumanMessage(question)]}, config)
    nouveaux = resultat["messages"][avant:]

    print(f"Étudiante : {question}")
    for m in nouveaux:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"   -> outil {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            print(f"      résultat : {m.content[:90].replace(chr(10), ' | ')}...")
    print(f"Assistant : {resultat['messages'][-1].content}")
    print(f"   ({len(resultat['messages'])} messages en mémoire dans ce fil)\n")


def main() -> None:
    verifier_ma_fonction()

    from src.assistant.agent_graphe import construire_agent
    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.candidature import creer_outils_candidature
    from src.assistant.offres import charger_base
    from src.assistant.outils_langchain import creer_outils
    from src.assistant.session import creer_session
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    print("=" * 70)
    print("1. Préparation (1 appel au LLM pour extraire le CV)")
    print("=" * 70)
    fiches = charger_base(RACINE / "data" / "offres.json")
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    session = creer_session()
    session.charger_cv((RACINE / "data" / "exemples" / "cv_exemple.txt").read_text(encoding="utf-8"))
    outils = creer_outils(moteur, fiches) + creer_outils_candidature(session, fiches)
    print(f"Session {session.identifiant} : {session.cv.nom}")
    print(f"Outils de l'agent : {[o.name for o in outils]}\n")

    agent = construire_agent(outils, session)

    print("=" * 70)
    print("2. Le graphe de l'agent (colle ce texte sur https://mermaid.live)")
    print("=" * 70)
    print(agent.get_graph().draw_mermaid())

    print("=" * 70)
    print("3. Une conversation dans le MÊME fil : l'agent se souvient")
    print("=" * 70)
    fil = session.identifiant  # un fil par session
    poser(agent, "Quels stages me correspondent le mieux ?", fil)
    poser(agent, "Et pour celui de Nexalia, qu'est-ce qui me manque exactement ?", fil)
    poser(agent, "Comment répondre à une question comportementale en entretien ?", fil)

    print("=" * 70)
    print("4. La même question floue dans un NOUVEAU fil : pas de contexte")
    print("=" * 70)
    poser(agent, "Et pour celui-là, qu'est-ce qui me manque ?", "nouveau-fil")


if __name__ == "__main__":
    main()