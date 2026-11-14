"""
Étape 0 du projet 3 : vérifier que les briques copiées fonctionnent dans le nouveau projet.

Le script ne fait AUCUN appel au LLM (zéro token dépensé) : il vérifie la clé Groq,
les imports, reconstruit l'index et refait une recherche dont on connaît la réponse ;
puis, si src/candidature/ existe, il vérifie les briques du projet 1.

Lancement, depuis la racine de assistant-candidature :
    python scripts/verifier_installation.py
"""

import os
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))


def etape(message: str, ok: bool) -> None:
    print(f"  [{'OK' if ok else 'ÉCHEC'}] {message}")
    if not ok:
        sys.exit(1)


def main() -> None:
    print("1. Clé Groq")
    from dotenv import load_dotenv
    load_dotenv(RACINE / ".env")
    etape("GROQ_API_KEY présente dans .env", bool(os.environ.get("GROQ_API_KEY")))

    print("2. Imports des briques du projet 2")
    from src.rag.chargement import charger_documents
    from src.rag.decoupage import decouper_document
    from src.rag.index import IndexVectoriel
    from src.rag.rag_classique import formater_contexte  # noqa: F401 (on vérifie juste l'import)
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege
    etape("tous les modules de src/rag/ s'importent", True)

    print("3. Index et recherche (le modèle se charge : quelques secondes)")
    documents = charger_documents(RACINE / "data" / "documents")
    etape(f"{len(documents)} documents chargés", len(documents) == 6)
    index = IndexVectoriel()
    index.ajouter([c for doc in documents for c in decouper_document(doc)])
    index.sauvegarder(RACINE / "data" / "index")
    etape(f"index construit et sauvegardé ({len(index)} passages)", len(index) > 0)

    moteur = MoteurProtege(RechercheHybride(index))
    premier = moteur.rechercher("Nexalia stipend", k=1)[0].chunk
    print(f"     « Nexalia stipend » -> [{premier.source} > {premier.section}]")
    etape("la recherche retrouve l'offre Nexalia, comme dans le projet 2",
          premier.source == "offre_stage_ia_generative.md")

    print("\nBriques du projet 2 : OK.")

    if not (RACINE / "src" / "candidature").exists():
        print("(src/candidature/ absent : lance scripts/preparer_candidature.py pour la suite)")
        return

    print("4. Briques du projet 1 (aucun appel au LLM)")
    import src.candidature.extraction  # noqa: F401
    import src.candidature.memoire  # noqa: F401
    import src.candidature.modeles  # noqa: F401
    import src.candidature.securite  # noqa: F401
    from src.candidature import outils
    etape("tous les modules de src/candidature/ s'importent", True)
    etape(f"seuil de similarité réglé à 0.6 (actuel : {outils.SEUIL_SIMILARITE})",
          abs(outils.SEUIL_SIMILARITE - 0.6) < 1e-9)
    r = outils.comparer_competences(["Machine Learning", "SQL"], ["ML", "Java"])
    print(f"     CV [Machine Learning, SQL] / offre [ML, Java] -> présentes {r['presentes']}, manquantes {r['manquantes']}")
    etape("« ML » reconnu, « Java » manquant (correction de l'étape 8)",
          r["presentes"] == ["ML"] and r["manquantes"] == ["Java"])

    print("\nTout fonctionne : les briques des projets 1 et 2 sont réunies dans le projet 3.")


if __name__ == "__main__":
    main()