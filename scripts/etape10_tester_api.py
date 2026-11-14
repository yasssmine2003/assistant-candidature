"""
Projet 3, étape 10 : tester l'API sans lancer de serveur (TestClient de FastAPI).

Le TestClient envoie de vraies requêtes HTTP à l'application, directement en mémoire.
`with TestClient(app)` déclenche aussi le démarrage (chargement de Qdrant et des offres).

Ce script : 2 appels au LLM (le CV, puis une question).

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape10_tester_api.py
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from fastapi.testclient import TestClient

from src.assistant.api import app


def titre(texte: str) -> None:
    print("\n" + "=" * 70 + f"\n{texte}\n" + "=" * 70)


def main() -> None:
    with TestClient(app) as client:
        titre("1. GET /sante")
        r = client.get("/sante")
        print(r.status_code, r.json())
        assert r.status_code == 200

        titre("2. Une session qui n'existe pas -> 404 (et pas 500)")
        r = client.get("/sessions/inconnue/offres")
        print(r.status_code, r.json())
        assert r.status_code == 404, "Une ressource absente doit renvoyer 404"

        titre("3. Un CV trop court -> 422 (validé par Pydantic, AVANT tout appel au LLM)")
        r = client.post("/sessions", json={"texte_cv": "trop court"})
        print(r.status_code, r.json()["detail"][0]["msg"])
        assert r.status_code == 422

        titre("4. POST /sessions avec le CV d'exemple (1 appel au LLM)")
        cv = (RACINE / "data" / "exemples" / "cv_exemple.txt").read_text(encoding="utf-8")
        r = client.post("/sessions", json={"texte_cv": cv})
        print(r.status_code, r.json())
        assert r.status_code == 201
        identifiant = r.json()["identifiant"]

        titre(f"5. GET /sessions/{identifiant}/offres (TA route, aucun appel au LLM)")
        r = client.get(f"/sessions/{identifiant}/offres")
        assert r.status_code == 200, r.text
        offres = r.json()
        for o in offres:
            print(f"   {o['Correspondance']:>5}  {o['Entreprise']:<12} manquantes : {o['Manquantes']}  [{o['source']}]")
        assert all("source" in o and "_taux" not in o and "_source" not in o for o in offres), \
            "Chaque ligne doit avoir 'source', et plus de clé commençant par '_'"
        print("Ta route classement_offres est correcte.")

        titre("6. POST /sessions/{id}/questions (1 appel au LLM ou plus)")
        r = client.post(f"/sessions/{identifiant}/questions",
                        json={"question": "Qu'est-ce qui me manque pour le stage de Nexalia ?"})
        print(r.status_code)
        print("Outils :", r.json().get("outils"))
        print("Réponse :", r.json().get("reponse", r.text)[:400])

    print("\nL'API fonctionne. Lance-la pour de vrai : uvicorn src.assistant.api:app --reload")


if __name__ == "__main__":
    main()