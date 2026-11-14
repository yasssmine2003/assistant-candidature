"""Les outils de l'assistant, version LangChain.

Avant (projet 2, outils.py) : pour chaque outil, on écrivait À LA MAIN
  1. la fonction Python ;
  2. un dictionnaire TOOLS décrivant au LLM son nom, son rôle et ses arguments ;
  3. un registre {nom: fonction} pour l'exécuter.
Maintenant : le décorateur @tool fabrique les trois à partir de la fonction elle-même :
  - le NOM vient du nom de la fonction ;
  - la DESCRIPTION vient de la docstring (première phrase) ;
  - le SCHÉMA des arguments vient des annotations de type, et la description de chaque
    argument vient de la section « Args: » de la docstring (parse_docstring=True).

La docstring n'est donc plus un simple commentaire : c'est ce que le LLM lit.
"""

from typing import Literal

from langchain_core.tools import tool

from src.assistant.offres import FicheOffre, lister_offres as lister_offres_texte
from src.rag.rag_classique import formater_contexte


def creer_outils(moteur, fiches: list[FicheOffre]) -> list:
    """Fabrique les outils liés à UN moteur de recherche et UNE base d'offres.
    C'est la closure du projet 2 : les outils « se souviennent » du moteur et des fiches,
    sans que le LLM ait à les fournir."""

    @tool
    def lister_offres() -> str:
        """Liste TOUTES les offres de stage (entreprise, poste, ville, gratification, date limite, compétences). À utiliser pour comparer des offres ou savoir lesquelles existent."""
        return lister_offres_texte(fiches)

    # ========================================================================
    # À TOI DE JOUER : l'outil de recherche documentaire
    # ========================================================================
    @tool(parse_docstring=True)
    def rechercher_documents(
        requete: str,
        type_document: Literal["offre", "guide", "referentiel"] | None = None,
    ) -> str:
        """Cherche les passages les plus pertinents dans les offres de stage, les guides de candidature et le référentiel de compétences.

        Args:
            requete: Les mots à chercher. Certaines offres sont en anglais : reformule dans la langue du document visé, et développe les acronymes (ML devient machine learning).
            type_document: Facultatif. Limite la recherche aux offres, aux guides ou au référentiel.
        """
        resultats = moteur.rechercher(requete, k=3, type_document=type_document)
        if not resultats:
            return "Aucun passage trouvé pour cette requête."
        return formater_contexte(resultats)
    return [lister_offres, rechercher_documents]