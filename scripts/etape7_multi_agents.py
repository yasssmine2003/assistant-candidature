"""
Projet 3, étape 7 : deux agents (rédacteur + relecteur) contre un agent unique.

Ce script :
0. vérifie TES fonctions apres_relecture et preuve_valide (sans LLM) ;
1. prépare la session (1 appel au LLM) ;
2. AGENT UNIQUE : la lettre TelcoMed de l'étape 6, avec ses seules vérifications par règles (1 appel) ;
3. DEUX AGENTS : la même lettre, rédigée puis relue, corrigée si besoin (2 appels par tour) ;
4. le bilan : appels au LLM, alertes restantes.

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape7_multi_agents.py
"""

import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from langgraph.graph import END

from src.assistant.multi_agents import MAX_ESSAIS, apres_relecture, preuve_valide

OFFRE = "offre_stage_devops.md"  # le cas difficile : aucune compétence demandée dans le CV


def verifier_ma_fonction() -> None:
    assert apres_relecture({"alertes": [], "essais": 1}) == END, "Aucune alerte : lettre approuvée -> END"
    assert apres_relecture({"alertes": ["x"], "essais": 1}) == "rediger", "Des alertes, essais restants -> rediger"
    assert apres_relecture({"alertes": ["x"], "essais": MAX_ESSAIS}) == END, "Essais épuisés -> END"
    assert apres_relecture({"alertes": [], "essais": MAX_ESSAIS}) == END
    print("Ta fonction apres_relecture passe les 4 tests.")

    source = "Stage chez ABC Tech : développement d’API REST avec FastAPI. Travail en équipe."
    assert preuve_valide("développement d'API REST avec FastAPI", source), "Apostrophe différente : doit être acceptée"
    assert preuve_valide("Développement d’API  REST", source), "Majuscule et double espace : doit être acceptée"
    assert not preuve_valide("salué par mes encadrants", source), "Absent des sources : refusé"
    assert not preuve_valide("", source), "Preuve vide : refusée"
    assert not preuve_valide("FastAPI", source), "Un seul mot ne prouve pas une phrase : refusé"
    print("Ta fonction preuve_valide passe les 5 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.lettre import creer_outil_lettre
    from src.assistant.multi_agents import construire_relecture
    from src.assistant.offres import charger_base
    from src.assistant.session import creer_session
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    print("=" * 70)
    print("1. Préparation (1 appel au LLM pour extraire le CV)")
    print("=" * 70)
    fiches = charger_base(RACINE / "data" / "offres.json")
    textes = {d.source: d.contenu for d in charger_documents(RACINE / "data" / "documents")}
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    session = creer_session()
    session.charger_cv((RACINE / "data" / "exemples" / "cv_exemple.txt").read_text(encoding="utf-8"))
    print(f"Session {session.identifiant} : {session.cv.nom}\n")

    print("=" * 70)
    print("2. AGENT UNIQUE (étape 6) : rédiger, vérifier par règles (1 appel)")
    print("=" * 70)
    sortie = creer_outil_lettre(session, fiches, moteur, textes).invoke({"source": OFFRE})
    print(sortie.split("---")[-1].strip())
    print("(La relecture du SENS n'existe pas : une phrase inventée sans chiffre passe.)\n")

    print("Pause de 60 s : l'offre gratuite de Groq limite le nombre de tokens par minute.\n")
    time.sleep(60)

    print("=" * 70)
    print("3. DEUX AGENTS : rédacteur + relecteur")
    print("=" * 70)
    graphe = construire_relecture(session, fiches, moteur, textes)
    print(graphe.get_graph().draw_mermaid())
    etat = graphe.invoke({"source": OFFRE, "lettre": None, "contexte": "", "alertes": [],
                          "corrections": [], "essais": 0, "journal": []})
    print("\n".join(etat["journal"]))
    print("\n--- Lettre finale ---\n")
    print(mettre_en_forme_finale(etat, session) if etat["lettre"] else "(aucune lettre : la rédaction a échoué)")

    print("\n" + "=" * 70)
    print("4. Bilan")
    print("=" * 70)
    print(f"Agent unique : 1 appel au LLM pour la lettre, relecture du sens : aucune")
    print(f"Deux agents  : {etat['essais']} rédaction(s) + {etat['essais']} relecture(s) = {2 * etat['essais']} appels")
    print(f"Alertes restantes : {len(etat['alertes'])}" + (" (approuvée)" if not etat["alertes"] else " (essais épuisés)"))


def mettre_en_forme_finale(etat, session) -> str:
    from src.assistant.lettre import mettre_en_forme
    return mettre_en_forme(etat["lettre"], session.cv.nom)


if __name__ == "__main__":
    main()