"""
Visualiser le contenu de la base Qdrant : un tableau des points, puis un nuage en 2D.

Chaque point = un passage (chunk). Son vecteur a 384 dimensions : impossible à dessiner tel quel.
On le projette en 2 dimensions avec une ACP (analyse en composantes principales) :
on garde les 2 directions où les vecteurs varient le plus. Deux passages proches par le sens
restent généralement proches sur le dessin.

Lancement, depuis la racine de assistant-candidature :
    python scripts/visualiser_qdrant.py
Le dessin s'ouvre dans une fenêtre et est enregistré dans data/nuage_passages.png.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

RACINE = Path(__file__).resolve().parent.parent
load_dotenv(RACINE / ".env")


def main() -> None:
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    points, _ = client.scroll("passages", limit=10_000, with_payload=True, with_vectors=True)
    points.sort(key=lambda p: p.payload["chunk_id"])

    # 1. Le tableau : ce que contient chaque point
    print(f"{len(points)} points dans la collection « passages »\n")
    print(f"{'type':<12}{'source':<32}{'section':<36}{'vecteur (3 premiers nombres)'}")
    print("-" * 110)
    for p in points:
        m = p.payload
        debut = ", ".join(f"{x:+.3f}" for x in p.vector[:3])
        print(f"{m['type']:<12}{m['source']:<32}{m['section'][:34]:<36}[{debut}, ...] ({len(p.vector)} nombres)")

    # 2. La projection en 2D (ACP faite à la main avec NumPy)
    X = np.array([p.vector for p in points])
    X = X - X.mean(axis=0)                       # centrer les vecteurs
    _, _, axes = np.linalg.svd(X, full_matrices=False)
    coords = X @ axes[:2].T                      # projeter sur les 2 axes principaux

    # 3. Le dessin, une couleur par type de document
    couleurs = {"offre": "tab:blue", "guide": "tab:orange", "referentiel": "tab:green"}
    fig, ax = plt.subplots(figsize=(11, 8))
    for type_doc, couleur in couleurs.items():
        idx = [i for i, p in enumerate(points) if p.payload["type"] == type_doc]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=couleur, s=80, label=type_doc)
    for (x, y), p in zip(coords, points):
        etiquette = f"{p.payload['source'].split('.')[0].replace('offre_stage_', '').replace('guide_', '')}\n{p.payload['section'][:22]}"
        ax.annotate(etiquette, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"Les {len(points)} passages de Qdrant, projetés en 2D (ACP)")
    ax.legend()
    sortie = RACINE / "data" / "nuage_passages.png"
    fig.savefig(sortie, dpi=150, bbox_inches="tight")
    print(f"\nDessin enregistré dans {sortie}")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())