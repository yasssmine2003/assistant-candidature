"""
Comparer le CV de la session aux offres : le lien entre les projets 1 et 2.

On réutilise comparer_competences du projet 1 (réparée avec le seuil de 0,6), en tenant
compte des alternatives de l'étape 1 : une exigence ["FastAPI", "Flask"] est satisfaite
si le CV contient FastAPI OU Flask.

Le calcul se fait DANS LE CODE (exact, gratuit), pas dans le LLM : le LLM ne fait
qu'expliquer le résultat à l'étudiante.
"""

from langchain_core.tools import tool

from src.assistant.offres import FicheOffre
from src.candidature.outils import comparer_competences


def evaluer_offre(competences_cv: list[str], fiche: FicheOffre, comparer=comparer_competences) -> dict:
    """Compare les compétences du CV aux exigences d'une offre.

    Toutes les alternatives sont comparées en un seul appel (un seul calcul d'embeddings),
    puis chaque exigence est satisfaite si au moins une de ses alternatives est présente.
    """
    alternatives = [alt for groupe in fiche.competences_requises for alt in groupe]
    resultat = comparer(competences_cv, alternatives)
    presentes = set(resultat["presentes"])
    via = {c["offre"]: c["cv"] for c in resultat["correspondances"]}  # quelle compétence du CV a servi

    satisfaites, manquantes = [], []
    for groupe in fiche.competences_requises:
        trouvees = [alt for alt in groupe if alt in presentes]
        nom = " ou ".join(groupe)
        if trouvees:
            preuve = via.get(trouvees[0], trouvees[0])
            satisfaites.append(nom if preuve == trouvees[0] else f"{nom} (grâce à « {preuve} »)")
        else:
            manquantes.append(nom)

    return {
        "source": fiche.source, "entreprise": fiche.entreprise, "poste": fiche.poste,
        "satisfaites": satisfaites, "manquantes": manquantes,
        "total": len(fiche.competences_requises),
    }


# ============================================================================
# À TOI DE JOUER : classer les offres
# ============================================================================
def classer(evaluations: list[dict]) -> str:
    if not evaluations:
        return "Aucune offre à classer."

    def taux(e):
        return len(e["satisfaites"]) / e["total"] if e["total"] else 0

    lignes = []
    for rang, e in enumerate(sorted(evaluations, key=taux, reverse=True), start=1):
        pourcentage = f"{taux(e):.0%}".replace("%", " %")
        lignes.append(f"{rang}. {e['entreprise']} ({e['poste']}) : {pourcentage} des exigences "
                      f"({len(e['satisfaites'])}/{e['total']}) [{e['source']}]")
    return "\n".join(lignes)


def creer_outils_candidature(session, fiches: list[FicheOffre]) -> list:
    """Les outils qui ont besoin du CV : la closure enferme la SESSION (étape 2)."""
    par_source = {f.source: f for f in fiches}

    def competences_cv() -> list[str]:
        if not session.cv_charge:
            raise ValueError("Aucun CV chargé dans la session.")
        return session.cv.competences

    @tool
    def classer_offres() -> str:
        """Classe TOUTES les offres de la plus adaptée à la moins adaptée au CV de l'étudiante, avec le pourcentage d'exigences satisfaites. À utiliser pour « quels stages me correspondent ? »."""
        return classer([evaluer_offre(competences_cv(), f) for f in fiches])

    @tool(parse_docstring=True)
    def analyser_candidature(source: str) -> str:
        """Analyse en détail une offre face au CV de l'étudiante : compétences présentes et manquantes.

        Args:
            source: Le nom du fichier de l'offre, par exemple offre_stage_ia_generative.md (donné par lister_offres ou classer_offres).
        """
        if source not in par_source:
            return f"Offre inconnue : {source}. Offres disponibles : {', '.join(par_source)}"
        e = evaluer_offre(competences_cv(), par_source[source])
        return (f"[{e['source']}] {e['entreprise']} - {e['poste']}\n"
                f"Exigences satisfaites : {len(e['satisfaites'])}/{e['total']}\n"
                f"Présentes : {' ; '.join(e['satisfaites']) or 'aucune'}\n"
                f"Manquantes : {' ; '.join(e['manquantes']) or 'aucune'}")

    return [classer_offres, analyser_candidature]