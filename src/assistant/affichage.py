"""
Étape 9 : petites fonctions d'affichage pour l'interface, testables SANS Streamlit.

Principe : l'interface (app.py) ne contient que de l'affichage. Toute logique, même petite,
vit dans une fonction normale qu'on peut tester. Une interface est difficile à tester ;
une fonction, non.

Tester ce fichier (depuis la racine du projet) :
    python -m src.assistant.affichage
"""

from langchain_core.messages import AIMessage, BaseMessage


# ============================================================================
# À TOI DE JOUER
# ============================================================================
def outils_appeles(messages: list[BaseMessage]) -> list[str]:
    noms = []
    for m in messages:
        for appel in getattr(m, "tool_calls", []):
            noms.append(appel["name"])
    return noms


def tableau_offres(evaluations: list[dict]) -> list[dict]:
    """Les évaluations (evaluer_offre) en lignes de tableau, de la meilleure à la moins bonne."""
    lignes = []
    for e in evaluations:
        taux = len(e["satisfaites"]) / e["total"] if e["total"] else 0
        lignes.append({"Entreprise": e["entreprise"], "Poste": e["poste"], "Correspondance": f"{taux:.0%}",
                       "Exigences": f"{len(e['satisfaites'])}/{e['total']}",
                       "Manquantes": ", ".join(e["manquantes"]) or "aucune", "_taux": taux,
                       "_source": e["source"]})
    return sorted(lignes, key=lambda l: l["_taux"], reverse=True)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, ToolMessage

    conversation = [
        HumanMessage("Qu'est-ce qui me manque pour Nexalia ?"),
        AIMessage("", tool_calls=[{"name": "lister_offres", "args": {}, "id": "1"}]),
        ToolMessage("...", tool_call_id="1"),
        AIMessage("", tool_calls=[{"name": "analyser_candidature", "args": {"source": "x.md"}, "id": "2"}]),
        ToolMessage("...", tool_call_id="2"),
        AIMessage("Il te manque Docker."),
    ]
    assert outils_appeles(conversation) == ["lister_offres", "analyser_candidature"]
    assert outils_appeles([HumanMessage("Bonjour"), AIMessage("Bonjour !")]) == [], "Aucun outil"
    assert outils_appeles([]) == []
    deux_en_un = AIMessage("", tool_calls=[{"name": "classer_offres", "args": {}, "id": "a"},
                                           {"name": "lister_offres", "args": {}, "id": "b"}])
    assert outils_appeles([deux_en_un]) == ["classer_offres", "lister_offres"], "Deux outils dans un seul message"
    print("Ta fonction outils_appeles passe les 4 tests.")