"""
Étape 0 du projet 3 : copier les briques du projet 2 dans src/rag/ et adapter leurs imports.

Pourquoi un script plutôt qu'un copier-coller à la main ?
- on ne copie QUE les briques utiles (pas l'agent ni l'API du projet 2 : l'assistant aura les siens) ;
- les imports sont adaptés automatiquement, sans risque d'en oublier un ;
- on peut relancer le script à tout moment : le résultat est toujours le même (reproductible).

Lancement, depuis la racine de assistant-candidature :
    python scripts/preparer_rag.py C:\\Users\\bouch\\Desktop\\agent-rag
"""

import re
import shutil
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# Les briques du projet 2 dont l'assistant a besoin
FICHIERS = [
    "embeddings.py",     # texte -> vecteur
    "chargement.py",     # lecture des documents
    "decoupage.py",      # passages (avec ta fonction decouper_texte)
    "index.py",          # index vectoriel (avec ta méthode rechercher_par_vecteur)
    "recherche.py",      # recherche hybride (avec ta fonction fusion_rrf)
    "securite.py",       # filtre anti-injection
    "llm.py",            # client Groq
    "rag_classique.py",  # pour formater_contexte (ta fonction) et le prompt ancré
]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage : python scripts/preparer_rag.py CHEMIN_VERS_agent-rag")
        sys.exit(1)

    projet2 = Path(sys.argv[1]).resolve()
    if not (projet2 / "src" / "index.py").exists():
        print(f"Je ne trouve pas src/index.py dans {projet2} : vérifie le chemin du projet 2.")
        sys.exit(1)

    # 1. Les dossiers deviennent des « paquets » Python grâce aux fichiers __init__.py
    dossier_rag = RACINE / "src" / "rag"
    dossier_rag.mkdir(parents=True, exist_ok=True)
    for paquet in (RACINE / "src", dossier_rag):
        (paquet / "__init__.py").touch()

    # 2. Copier chaque brique en adaptant ses imports :
    #    « from src.index import ... »  devient  « from src.rag.index import ... »
    for nom in FICHIERS:
        texte = (projet2 / "src" / nom).read_text(encoding="utf-8")
        texte_adapte, nb = re.subn(r"\bfrom src\.(\w+) import", r"from src.rag.\1 import", texte)
        (dossier_rag / nom).write_text(texte_adapte, encoding="utf-8")
        print(f"  {nom:<18} copié ({nb} import(s) adapté(s))")

    # 3. Les documents et la clé Groq
    shutil.copytree(projet2 / "data" / "documents", RACINE / "data" / "documents", dirs_exist_ok=True)
    print("  data/documents/    copié")
    if (projet2 / ".env").exists() and not (RACINE / ".env").exists():
        shutil.copy(projet2 / ".env", RACINE / ".env")
        print("  .env               copié")

    print("\nTerminé. Lance maintenant : python scripts/verifier_installation.py")


if __name__ == "__main__":
    main()