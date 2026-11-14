"""
Projet 3, étape 8 : évaluer le relecteur sur des lettres piégées.

Ce script :
0. vérifie TA fonction detectee (sans LLM) ;
1. prépare la session et les sources (1 appel au LLM pour extraire le CV) ;
2. relit la lettre PROPRE (témoin) puis 5 lettres PIÉGÉES (6 appels au relecteur) ;
3. affiche qui a trouvé chaque piège : les règles, le relecteur, ou personne ;
4. calcule les métriques et les sauvegarde dans data/evaluation_relecteur.json.

Durée : environ 3 minutes (pauses de 25 s entre les appels, à cause de la limite gratuite de Groq).

Lancement, depuis la racine de assistant-candidature :
    python scripts/etape8_evaluation.py
"""

import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.assistant.evaluation import LETTRE_PROPRE, PIEGES, detectee, pieger

OFFRE = "offre_stage_devops.md"
PAUSE = 25  # secondes entre deux appels au relecteur


def creer_relecteur_robuste():
    """Le relecteur de l'étape 7, plus un PLAN B (with_fallbacks).

    Problème vu au 1er lancement : avec temperature=0, le modèle est (presque) déterministe.
    S'il produit un JSON invalide, le réessai lui renvoie EXACTEMENT la même question... et il
    refait EXACTEMENT la même erreur, 4 fois de suite. Réessayer à l'identique ne sert à rien.

    Plan B : si le relecteur principal échoue, LangChain passe au secours, qui a une température
    un peu plus haute (0.4). Sa réponse sera légèrement différente, donc a des chances d'être valide.
    """
    from groq import BadRequestError, RateLimitError
    from langchain_groq import ChatGroq

    from src.assistant.multi_agents import Relecture
    from src.rag.llm import MODELE

    def version(temperature: float, essais: int):
        modele = ChatGroq(model=MODELE, temperature=temperature, reasoning_effort="low")
        return modele.with_structured_output(Relecture, method="json_schema", strict=True).with_retry(
            retry_if_exception_type=(BadRequestError, RateLimitError), stop_after_attempt=essais,
            exponential_jitter_params={"initial": 5, "max": 30})

    return version(0, 2).with_fallbacks([version(0.4, 3)], exceptions_to_handle=(BadRequestError, RateLimitError))


def verifier_ma_fonction() -> None:
    piege = "Chez XYZ AI, j’ai encadré une équipe de cinq stagiaires."
    assert detectee(piege, ["affirmation non soutenue : « j'ai encadré une équipe de cinq stagiaires »"]), \
        "L'alerte recopie la phrase (apostrophe différente) : détectée"
    assert detectee(piege, ["Encadrement d'une équipe de cinq stagiaires"]), \
        "Reformulée mais avec la moitié des mots (equipe, cinq, stagiaires) : détectée"
    assert not detectee(piege, ["TelcoMed, reconnue pour la santé"]), "Une autre phrase : pas détectée"
    assert not detectee(piege, []), "Aucune alerte : pas détectée"
    print("Ta fonction detectee passe les 4 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    from src.assistant.base_vectorielle import BaseVectorielle
    from groq import BadRequestError, RateLimitError

    from src.assistant.evaluation import construire_contexte, relire_seul
    from src.assistant.offres import charger_base
    from src.assistant.session import creer_session
    from src.rag.chargement import charger_documents
    from src.rag.recherche import RechercheHybride
    from src.rag.securite import MoteurProtege

    print("=" * 70)
    print("1. Préparation (1 appel au LLM pour extraire le CV)")
    print("=" * 70)
    fiche = {f.source: f for f in charger_base(RACINE / "data" / "offres.json")}[OFFRE]
    texte_offre = {d.source: d.contenu for d in charger_documents(RACINE / "data" / "documents")}[OFFRE]
    moteur = MoteurProtege(RechercheHybride(BaseVectorielle()))
    session = creer_session()
    session.charger_cv((RACINE / "data" / "exemples" / "cv_exemple.txt").read_text(encoding="utf-8"))
    contexte = construire_contexte(session, fiche, texte_offre, moteur)
    relecteur = creer_relecteur_robuste()
    print(f"Session {session.identifiant} : {session.cv.nom} | offre : {fiche.entreprise}\n")

    def relire(lettre):
        return relire_seul(relecteur, lettre, contexte, session.texte_cv, texte_offre, fiche.reference)

    print("=" * 70)
    print(f"2. Relectures : 1 lettre propre + {len(PIEGES)} lettres piégées (pause de {PAUSE} s entre chaque)")
    print("=" * 70)
    time.sleep(PAUSE)
    temoin = relire(LETTRE_PROPRE)
    print(f"Lettre PROPRE : {temoin['nb_affirmations']} affirmations relevées, "
          f"{len(temoin['regles'])} alerte(s) de règles, {len(temoin['sens'])} alerte(s) du relecteur")
    for a in temoin["regles"] + temoin["sens"]:
        print(f"   - faux positif ? {a}")

    resultats, echecs = [], []
    for piege in PIEGES:
        time.sleep(PAUSE)
        try:
            r = relire(pieger(LETTRE_PROPRE, piege))
        except (BadRequestError, RateLimitError) as erreur:
            # Un cas qui plante ne doit pas arrêter toute l'évaluation : on le note et on continue.
            echecs.append(piege["nom"])
            print(f"Piège « {piege['nom']} » : ÉCHEC technique (Groq : {str(erreur)[:70]}...)")
            continue
        # Les règles ont trouvé le piège si elles donnent PLUS d'alertes que sur la lettre propre
        par_regles = len(r["regles"]) > len(temoin["regles"])
        par_relecteur = detectee(piege["phrase"], r["sens"])
        autres = [a for a in r["sens"] if not detectee(piege["phrase"], [a])]
        resultats.append({"piege": piege["nom"], "phrase": piege["phrase"], "regles": par_regles,
                          "relecteur": par_relecteur, "alertes_regles": r["regles"],
                          "alertes_relecteur": r["sens"], "autres_alertes": len(autres)})
        print(f"Piège « {piege['nom']} » : règles {'OUI' if par_regles else 'non'} | "
              f"relecteur {'OUI' if par_relecteur else 'non'} | autres alertes du relecteur : {len(autres)}")

    print("\n" + "=" * 70)
    print("3. Tableau : qui trouve quoi ?")
    print("=" * 70)
    print(f"{'Piège':<20}{'Règles':<10}{'Relecteur':<12}{'Les deux réunis'}")
    for r in resultats:
        print(f"{r['piege']:<20}{'OUI' if r['regles'] else '-':<10}{'OUI' if r['relecteur'] else '-':<12}"
              f"{'OUI' if r['regles'] or r['relecteur'] else 'MANQUÉ'}")

    print("\n" + "=" * 70)
    print("4. Métriques")
    print("=" * 70)
    n = len(resultats)
    if n == 0:
        print("Aucun piège n'a pu être relu : relance le script dans une minute.")
        return
    if echecs:
        print(f"Attention : {len(echecs)} piège(s) non évalué(s) ({', '.join(echecs)}), "
              f"les taux portent sur {n} pièges seulement.")
    metriques = {
        "detection_regles": sum(r["regles"] for r in resultats) / n,
        "detection_relecteur": sum(r["relecteur"] for r in resultats) / n,
        "detection_combinee": sum(r["regles"] or r["relecteur"] for r in resultats) / n,
        "faux_positifs_lettre_propre": len(temoin["regles"]) + len(temoin["sens"]),
        "alertes_hors_piege": sum(r["autres_alertes"] for r in resultats),
        "echecs_techniques": echecs,
    }
    print(f"Taux de détection (rappel) - règles seules : {metriques['detection_regles']:.0%}")
    print(f"Taux de détection (rappel) - relecteur seul : {metriques['detection_relecteur']:.0%}")
    print(f"Taux de détection (rappel) - les deux       : {metriques['detection_combinee']:.0%}")
    print(f"Faux positifs sur la lettre propre          : {metriques['faux_positifs_lettre_propre']}")
    print(f"Alertes du relecteur hors piège             : {metriques['alertes_hors_piege']} "
          f"(sur {n} lettres : du bruit, ou une vraie faiblesse de la lettre « propre » ?)")

    sortie = RACINE / "data" / "evaluation_relecteur.json"
    sortie.write_text(json.dumps({"metriques": metriques, "temoin": temoin, "resultats": resultats},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats sauvegardés : {sortie.relative_to(RACINE)} (pour comparer après une modification)")


if __name__ == "__main__":
    main()