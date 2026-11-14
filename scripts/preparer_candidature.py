"""
Étape 0 du projet 3 (suite) : copier les briques du projet 1 dans src/candidature/.

Le projet 1 importait « à plat » (from modeles import ...), car on lançait les scripts
depuis src/. Le projet 3 importe depuis la racine (from src.candidature.modeles import ...).
Le script adapte les imports, et redirige embeddings et llm vers src/rag/ pour ne garder
qu'UN modèle d'embeddings et UN client Groq dans tout le projet.

Lancement, depuis la racine de assistant-candidature :
    python scripts/preparer_candidature.py C:\\Users\\bouch\\Downloads\\agent-candidature\\agent-candidature
"""

import re
import shutil
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

FICHIERS = ["modeles.py", "extraction.py", "outils.py", "securite.py", "memoire.py"]
COPIES = {f.removesuffix(".py") for f in FICHIERS}
VERS_RAG = {"embeddings", "llm"}                       # déjà présents dans src/rag/
NON_COPIES = {"agent", "api", "main", "calibrer_seuil"}  # volontairement laissés de côté


def adapter_imports(texte: str) -> tuple[str, int]:
    """Réécrit les imports à plat du projet 1 en imports depuis la racine du projet 3."""
    def remplacer(m: re.Match) -> str:
        retrait, module = m.group(1), m.group(2)
        if module in COPIES:
            return f"{retrait}from src.candidature.{module} import"
        if module in VERS_RAG:
            return f"{retrait}from src.rag.{module} import"
        return m.group(0)  # bibliothèque externe (pydantic, re...) : on ne touche à rien
    return re.subn(r"^(\s*)from (\w+) import", remplacer, texte, flags=re.MULTILINE)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage : python scripts/preparer_candidature.py CHEMIN_VERS_agent-candidature")
        sys.exit(1)

    projet1 = Path(sys.argv[1]).resolve()
    if not (projet1 / "src" / "modeles.py").exists():
        print(f"Je ne trouve pas src/modeles.py dans {projet1} : vérifie le chemin du projet 1.")
        sys.exit(1)

    dossier = RACINE / "src" / "candidature"
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "__init__.py").touch()

    problemes = []
    for nom in FICHIERS:
        texte = (projet1 / "src" / nom).read_text(encoding="utf-8")
        adapte, _ = adapter_imports(texte)
        nb = sum(1 for a, b in zip(texte.splitlines(), adapte.splitlines()) if a != b)
        (dossier / nom).write_text(adapte, encoding="utf-8")
        print(f"  {nom:<14} copié ({nb} import(s) adapté(s))")
        # Un fichier copié dépend-il d'un fichier qu'on n'a pas copié ?
        for module in re.findall(r"^\s*(?:from|import) (\w+)", adapte, flags=re.MULTILINE):
            if module in NON_COPIES:
                problemes.append(f"{nom} importe « {module} », qui n'a pas été copié")

    # Les fichiers d'exemple (CV, offre) du projet 1, s'il y en a à la racine
    exemples = RACINE / "data" / "exemples"
    for fichier in projet1.glob("*.txt"):
        exemples.mkdir(parents=True, exist_ok=True)
        shutil.copy(fichier, exemples / fichier.name)
        print(f"  {fichier.name:<14} copié dans data/exemples/")

    if problemes:
        print("\nATTENTION :")
        for p in problemes:
            print(f"  - {p}")
        print("Envoie ce message à ton mentor : il faudra déplacer cette dépendance.")
    else:
        print("\nTerminé, aucune dépendance manquante. Lance : python scripts/verifier_installation.py")


if __name__ == "__main__":
    main()