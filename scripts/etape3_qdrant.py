"""
Projet 3, étape 3 : remplacer l'index NumPy par une base vectorielle en ligne (Qdrant Cloud).

Ce script :
0. vérifie TA fonction convertir_resultats (sans connexion ni modèle) ;
1. envoie les 26 passages dans Qdrant Cloud ;
2. compare NumPy et Qdrant sur les questions du projet 2 : mêmes résultats ?
3. montre le filtre par métadonnées, appliqué par Qdrant ;
4. retire puis remet un document, sans tout reconstruire ;
5. branche la recherche hybride du projet 2 sur Qdrant, sans modifier une ligne ;
6. affiche un vecteur stocké, pour le retrouver ensuite dans le tableau de bord.

Prérequis : QDRANT_URL et QDRANT_API_KEY dans le fichier .env.

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape3_qdrant.py
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.assistant.base_vectorielle import BaseVectorielle, convertir_resultats, identifiant_point
from src.rag.chargement import charger_documents
from src.rag.decoupage import decouper_document
from src.rag.index import IndexVectoriel
from src.rag.recherche import RechercheHybride

QUESTIONS = [
    "Quelle est la gratification du stage chez Nexalia ?",
    "Quels stages demandent du Machine Learning ?",
    "Comment répondre à une question comportementale en entretien ?",
    "Combien de mots doit faire une lettre de motivation ?",
    "Kubernetes",
]


def verifier_ma_fonction() -> None:
    faux_points = [
        SimpleNamespace(score=0.75, payload={"chunk_id": "a.md#0", "source": "a.md", "section": "Intro",
                                              "texte": "texte A", "nb_mots": 2, "type": "offre"}),
        SimpleNamespace(score=0.4, payload={"chunk_id": "b.md#3", "source": "b.md", "section": "Profil",
                                             "texte": "texte B", "nb_mots": 5, "type": "guide"}),
    ]
    r = convertir_resultats(faux_points)
    assert len(r) == 2, "Deux résultats attendus"
    assert r[0].chunk.id == "a.md#0", "L'id du Chunk est dans payload['chunk_id']"
    assert r[0].chunk.source == "a.md" and r[0].chunk.section == "Intro"
    assert r[1].chunk.texte == "texte B" and r[1].chunk.nb_mots == 5
    assert abs(r[0].score - 0.75) < 1e-9, "Qdrant donne déjà la similarité : score = point.score"
    assert convertir_resultats([]) == [], "Aucun point : liste vide"
    print("Ta fonction convertir_resultats passe les 5 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    print("=" * 70)
    print("1. Envoyer les passages dans Qdrant Cloud")
    print("=" * 70)
    debut = time.perf_counter()
    base = BaseVectorielle()
    chunks = [c for d in charger_documents(RACINE / "data" / "documents") for c in decouper_document(d)]
    base.ajouter(chunks)
    print(f"{len(base)} passages dans la collection « passages » ({time.perf_counter() - debut:.1f} s)\n")

    print("=" * 70)
    print("2. NumPy (projet 2) contre Qdrant : mêmes résultats ?")
    print("=" * 70)
    numpy_index = IndexVectoriel.charger(RACINE / "data" / "index")
    identiques = 0
    for q in QUESTIONS:
        n = numpy_index.rechercher(q, k=3)
        c = base.rechercher(q, k=3)
        meme = [x.chunk.id for x in n] == [x.chunk.id for x in c]
        identiques += meme
        print(f"« {q} »")
        print(f"   NumPy  n°1 : {n[0].score:.3f} [{n[0].chunk.source} > {n[0].chunk.section}]")
        print(f"   Qdrant n°1 : {c[0].score:.3f} [{c[0].chunk.source} > {c[0].chunk.section}]")
        print(f"   top 3 identique : {'oui' if meme else 'NON'}")
    print(f"\n{identiques}/{len(QUESTIONS)} questions avec exactement le même top 3.\n")

    print("=" * 70)
    print("3. Filtre par métadonnées, appliqué par Qdrant")
    print("=" * 70)
    q = "Quels stages demandent du Machine Learning ?"
    for filtre in (None, "offre"):
        print(f"Filtre {filtre!r} :")
        for r in base.rechercher(q, k=3, type_document=filtre):
            print(f"   {r.score:.3f} [{r.chunk.source} > {r.chunk.section}]")
    print()

    print("=" * 70)
    print("4. Retirer puis remettre un document, sans tout reconstruire")
    print("=" * 70)
    avant = len(base)
    base.supprimer_document("guide_entretien_technique.md")
    print(f"Après suppression du guide d'entretien : {len(base)} passages (avant : {avant})")
    print(f"   « question comportementale » -> {base.rechercher(QUESTIONS[2], k=1)[0].chunk.source}")
    base.ajouter([c for c in chunks if c.source == "guide_entretien_technique.md"])
    print(f"Après remise : {len(base)} passages\n")

    print("=" * 70)
    print("5. La recherche hybride du projet 2, branchée sur Qdrant")
    print("=" * 70)
    moteur = RechercheHybride(base)  # aucune modification de RechercheHybride
    for q in ("Kubernetes", "Nexalia stipend"):
        r = moteur.rechercher(q, k=1)[0]
        print(f"« {q} » -> [{r.chunk.source} > {r.chunk.section}] (rang sens {r.rang_sens}, rang mots {r.rang_mots})")
    print()

    print("=" * 70)
    print("6. Voir un vecteur stocké")
    print("=" * 70)
    chunk_id = "guide_lettre_motivation.md#0"
    point = base.client.retrieve(base.nom, ids=[identifiant_point(chunk_id)], with_vectors=True)[0]
    print(f"Passage    : {chunk_id}")
    print(f"Id Qdrant  : {point.id}")
    print(f"Vecteur    : {len(point.vector)} nombres, les 6 premiers : {[round(x, 4) for x in point.vector[:6]]}")
    print("Retrouve ce point dans le tableau de bord : Collections > passages.")


if __name__ == "__main__":
    main()