"""
Projet 3, étape 1 : construire la base d'offres structurée.

Ce script :
0. vérifie la fonction lister_offres (sans LLM) ;
1. extrait une fiche par offre (3 appels au LLM, une seule fois) et la vérifie ;
2. sauvegarde la base dans data/offres.json ;
3. montre ce que l'agent verra avec lister_offres ;
4. compare les offres DANS LE CODE : exact, instantané, gratuit.

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape1_offres.py
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.assistant.offres import (FicheOffre, charger_base, extraire_fiche, lister_offres,
                                  sauvegarder_base, verifier_fiche)
from src.rag.chargement import charger_documents

FICHIER_BASE = RACINE / "data" / "offres.json"


def verifier_ma_fonction() -> None:
    fiche = FicheOffre(
        source="a.md", entreprise="Acme", poste="Dev", ville=None, modalite="sur site",
        duree_mois_min=4, duree_mois_max=4, gratification_tnd_mois=700, date_limite="2027-01-20",
        profil=[], langues=[],
        competences_requises=[["Python"], ["FastAPI", "Flask"]], competences_bonus=[], reference=None,
    )
    attendu = "[a.md] Acme | Dev | ? | 700 TND/mois | limite 2027-01-20 | Python, FastAPI ou Flask"
    obtenu = lister_offres([fiche])
    assert obtenu == attendu, f"Une offre :\n  attendu : {attendu}\n  obtenu  : {obtenu}"
    sans_gratif = fiche.model_copy(update={"gratification_tnd_mois": None, "source": "b.md"})
    assert lister_offres([fiche, sans_gratif]).split("\n")[1].startswith("[b.md] Acme | Dev | ? | ? TND/mois"), \
        "Deux offres : une ligne chacune, séparées par \\n ; gratification absente -> « ? »"
    assert lister_offres([]) == "Aucune offre dans la base.", "Aucune fiche : message prévu"
    print("lister_offres passe les 3 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    print("=" * 70)
    print("1. Extraction et vérification des fiches (3 appels au LLM)")
    print("=" * 70)
    documents = [d for d in charger_documents(RACINE / "data" / "documents") if d.source.startswith("offre_")]
    fiches = []
    for doc in documents:
        fiche = extraire_fiche(doc)
        fiches.append(fiche)
        print(f"\n[{fiche.source}]")
        for champ, valeur in fiche.model_dump(exclude={"source"}).items():
            print(f"   {champ:<22} {valeur}")
        alertes = verifier_fiche(fiche, doc.contenu)
        print(f"   -> vérification : {'OK' if not alertes else 'ALERTES : ' + ' ; '.join(alertes)}")

    sauvegarder_base(fiches, FICHIER_BASE)
    fiches = charger_base(FICHIER_BASE)  # on relit, comme le fera l'assistant
    print(f"\nBase sauvegardée puis relue : {len(fiches)} fiches dans {FICHIER_BASE.name}\n")

    print("=" * 70)
    print("2. Ce que l'agent verra avec lister_offres()")
    print("=" * 70)
    print(lister_offres(fiches))
    print()

    print("=" * 70)
    print("3. Comparer dans le code (sans LLM)")
    print("=" * 70)
    payees = [f for f in fiches if f.gratification_tnd_mois is not None]
    mieux = max(payees, key=lambda f: f.gratification_tnd_mois)
    moins = min(payees, key=lambda f: f.gratification_tnd_mois)
    proche = min((f for f in fiches if f.date_limite), key=lambda f: f.date_limite)
    print(f"Le mieux payé       : {mieux.entreprise} ({mieux.gratification_tnd_mois} TND/mois)")
    print(f"Le moins payé       : {moins.entreprise} ({moins.gratification_tnd_mois} TND/mois)")
    print(f"Date limite proche  : {proche.entreprise} ({proche.date_limite})")
    print("\nAucun token dépensé pour ces trois réponses, et elles portent sur TOUTES les offres.")


if __name__ == "__main__":
    main()