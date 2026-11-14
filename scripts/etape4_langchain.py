"""
Projet 3, étape 4 : traduire tes briques en LangChain, et comparer.

Ce script :
0. vérifie TON outil rechercher_documents (sans LLM ni Qdrant) ;
1. le modèle de chat LangChain : un appel simple (1 appel au LLM) ;
2. les sorties structurées en une ligne : extraire une fiche d'offre (1 appel) ;
3. le schéma généré par @tool, comparé au dictionnaire TOOLS écrit à la main au projet 2 ;
4. le tool calling avec LangChain : le LLM choisit un outil, on l'exécute, il répond (2 appels).

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape4_langchain.py
"""

import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from groq import BadRequestError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.assistant.offres import PROMPT_EXTRACTION, FicheExtraite, charger_base
from src.assistant.outils_langchain import creer_outils
from src.rag.decoupage import Chunk
from src.rag.recherche import ResultatHybride


class FauxMoteur:
    """Remplace le vrai moteur pour tester l'outil sans Qdrant ni modèle."""

    def __init__(self):
        self.appels = []

    def rechercher(self, requete, k=3, type_document=None):
        self.appels.append((requete, type_document))
        if requete == "introuvable":
            return []
        chunk = Chunk(id="a.md#0", source="a.md", section="Intro", texte="Texte A", nb_mots=2)
        return [ResultatHybride(chunk=chunk, score_rrf=0.0, rang_sens=1, rang_mots=None, score_sens=0.5)]


def verifier_mon_outil() -> None:
    moteur = FauxMoteur()
    _, rechercher = creer_outils(moteur, [])
    assert rechercher.name == "rechercher_documents"
    assert "____" not in rechercher.description and len(rechercher.description) > 20, \
        "Écris une vraie description dans la docstring : c'est ce que le LLM lit"
    schema = convert_to_openai_tool(rechercher)["function"]["parameters"]["properties"]
    assert "____" not in schema["requete"].get("description", "____"), "Décris l'argument requete dans « Args: »"
    assert "____" not in schema["type_document"].get("description", "____"), "Décris l'argument type_document"
    texte = rechercher.invoke({"requete": "python", "type_document": "offre"})
    assert moteur.appels[-1] == ("python", "offre"), "Transmets requete ET type_document au moteur"
    assert texte == '<passage source="a.md" section="Intro">\nTexte A\n</passage>', \
        "Des passages trouvés : renvoyer formater_contexte(resultats)"
    assert rechercher.invoke({"requete": "introuvable"}) == "Aucun passage trouvé pour cette requête."
    print("Ton outil rechercher_documents passe les 6 tests.\n")


def main() -> None:
    verifier_mon_outil()

    # Les imports qui se connectent (Groq, Qdrant) viennent après les tests
    from src.assistant.base_vectorielle import BaseVectorielle
    from src.assistant.llm_langchain import modele
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    print("=" * 70)
    print("1. Le modèle de chat LangChain (1 appel)")
    print("=" * 70)
    reponse = modele.invoke([HumanMessage("Présente-toi en une phrase, en français.")])
    print(f"Type de la réponse : {type(reponse).__name__}")
    print(f"Texte  : {reponse.content}")
    print(f"Tokens : {reponse.usage_metadata}\n")

    print("=" * 70)
    print("2. Sorties structurées en une ligne (1 appel)")
    print("=" * 70)
    extracteur = (
        modele.with_structured_output(FicheExtraite, method="json_schema", strict=True)
        # Le réessai qu'on avait écrit à la main à l'étape 1, en une ligne.
        # Limite : il relance la MÊME demande, sans montrer l'erreur au modèle.
        .with_retry(retry_if_exception_type=(BadRequestError,), stop_after_attempt=3)
    )
    offre = next(d for d in charger_documents(RACINE / "data" / "documents") if d.source == "offre_stage_devops.md")
    fiche = extracteur.invoke([SystemMessage(PROMPT_EXTRACTION), HumanMessage(f"<offre>\n{offre.contenu}\n</offre>")])
    print(f"Type : {type(fiche).__name__} (directement l'objet Pydantic, déjà validé)")
    print(f"{fiche.entreprise} | {fiche.poste} | {fiche.gratification_tnd_mois} TND | {fiche.competences_requises}\n")

    print("=" * 70)
    print("3. Le schéma fabriqué par @tool (ce que le LLM reçoit)")
    print("=" * 70)
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    outils = creer_outils(moteur, charger_base(RACINE / "data" / "offres.json"))
    print(json.dumps(convert_to_openai_tool(outils[1]), ensure_ascii=False, indent=2))
    print("Compare avec le dictionnaire TOOLS écrit à la main dans le projet 2 (src/outils.py).\n")

    print("=" * 70)
    print("4. Le tool calling avec LangChain (2 appels)")
    print("=" * 70)
    par_nom = {o.name: o for o in outils}
    modele_outils = modele.bind_tools(outils)  # le modèle « connaît » maintenant les outils
    messages = [
        SystemMessage("Tu aides une étudiante à choisir un stage. Utilise les outils pour répondre, "
                      "et cite les fichiers sources entre crochets."),
        HumanMessage("Quel stage paie le mieux, et quelle est sa date limite ?"),
    ]
    ai = modele_outils.invoke(messages)
    print(f"Le LLM demande : {[(tc['name'], tc['args']) for tc in ai.tool_calls] or 'aucun outil'}")
    messages.append(ai)
    for tc in ai.tool_calls:
        resultat = par_nom[tc["name"]].invoke(tc)  # renvoie directement un ToolMessage
        print(f"Résultat de {tc['name']} : {len(resultat.content)} caractères")
        messages.append(resultat)
    finale = modele_outils.invoke(messages)
    print(f"\nRéponse finale : {finale.content}")
    print("\nIci, c'est encore TOI qui as fait la boucle (appeler, exécuter, renvoyer).")
    print("À l'étape 5, LangGraph la fera à ta place.")


if __name__ == "__main__":
    main()