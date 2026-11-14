"""
Étape 8 : évaluer le relecteur sur des lettres PIÉGÉES.

Idée : on ne sait pas si un vérificateur marche tant qu'on ne l'a pas mesuré sur des erreurs
connues d'avance. On part d'une lettre propre (celle que ton graphe a approuvée à l'étape 7),
on y glisse UNE phrase inventée à la fois, et on regarde qui la repère :
    - les RÈGLES (verifier_lettre, étape 6 : chiffres, compétences citées, longueur, référence) ;
    - le RELECTEUR (LLM de l'étape 7 + ta fonction preuve_valide).

La lettre propre sert aussi de témoin : toute alerte sur elle est un FAUX POSITIF.

Remarque d'architecte : relire_seul reprend la logique du nœud « relire » de multi_agents.py.
C'est une duplication assumée pour cette étape (pour ne pas toucher à ton fichier) ; à l'étape 9,
on la factorisera dans une seule fonction utilisée par les deux.
"""

import re

from groq import BadRequestError, RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.assistant.candidature import evaluer_offre
from src.assistant.lettre import Lettre, mettre_en_forme, verifier_lettre
from src.assistant.multi_agents import PROMPT_RELECTEUR, Relecture, normaliser, preuve_valide
from src.rag.llm import MODELE
from src.rag.rag_classique import formater_contexte

# ----------------------------------------------------------------------------
# Le jeu de test : une lettre propre + des pièges
# ----------------------------------------------------------------------------
# La lettre TelcoMed approuvée par ton graphe à l'étape 7 (recopiée telle quelle).
LETTRE_PROPRE = Lettre(
    objet="Stage Ingénieur DevOps / Cloud – Réf. DEVOPS-27-02",
    paragraphe_vous=(
        "Je souhaite vivement rejoindre TelcoMed pour ce stage d’ingénieur DevOps / Cloud. L’opportunité "
        "d’automatiser le déploiement des applications internes, de mettre en place des pipelines CI/CD avec "
        "GitLab et de superviser la plateforme correspond exactement aux missions que vous proposez. Votre "
        "environnement, basé sur la conteneurisation et la supervision, représente un cadre idéal pour "
        "approfondir mes compétences techniques tout en contribuant à la stabilité et à la performance des "
        "services internes."),
    paragraphe_moi=(
        "Au cours de mon stage chez XYZ AI, j’ai prototypé un système RAG, ce qui m’a permis de consolider ma "
        "maîtrise de Python et de travailler efficacement en équipe. Par ailleurs, mon expérience en tant que "
        "développeuse Python chez ABC Tech m’a amenée à concevoir et déployer des API REST avec FastAPI, en "
        "utilisant Git pour le contrôle de version et en collaborant étroitement avec les équipes de "
        "développement. Ces expériences m’ont appris à structurer des projets, à gérer des dépendances et à "
        "assurer la qualité du code grâce à des pratiques de travail en équipe. Elles démontrent ma capacité à "
        "apprendre rapidement de nouveaux outils et à les appliquer dans un contexte professionnel."),
    paragraphe_nous=(
        "Je suis convaincue que mon expérience en développement Python, combinée à ma volonté d’apprendre les "
        "technologies DevOps, me permettra de contribuer efficacement à la mise en place de pipelines CI/CD, à "
        "la conteneurisation des services et à la supervision de la plateforme chez TelcoMed. Je souhaite mettre "
        "à profit ma rigueur et mon esprit d’équipe pour assurer la continuité des services et soutenir les "
        "équipes de développement dans l’optimisation de leurs processus. Je serais ravie de discuter plus en "
        "détail de la manière dont je pourrais apporter une valeur ajoutée à votre équipe lors d’un entretien."),
    competences_citees=["Python", "FastAPI", "Git"],
)

# Chaque piège = UNE phrase inventée, d'un type différent, glissée dans un paragraphe.
PIEGES = [
    {"nom": "compétence", "paragraphe": "paragraphe_moi",
     "phrase": "Chez ABC Tech, j’ai également déployé des clusters Kubernetes en production."},
    {"nom": "chiffre", "paragraphe": "paragraphe_moi",
     "phrase": "Ce travail a réduit de 40 % le temps de déploiement de l’équipe."},
    {"nom": "jugement", "paragraphe": "paragraphe_moi",
     "phrase": "Mon sérieux a été salué par mes encadrants, qui m’ont proposé un poste à la fin du stage."},
    {"nom": "entreprise", "paragraphe": "paragraphe_vous",
     "phrase": "TelcoMed est d’ailleurs le leader tunisien de la santé numérique."},
    {"nom": "nombre en lettres", "paragraphe": "paragraphe_moi",
     "phrase": "Chez XYZ AI, j’ai encadré une équipe de cinq stagiaires."},
]


def pieger(lettre: Lettre, piege: dict) -> Lettre:
    """Copie de la lettre avec la phrase piège ajoutée à la fin du paragraphe choisi."""
    texte = getattr(lettre, piege["paragraphe"])
    return lettre.model_copy(update={piege["paragraphe"]: f"{texte} {piege['phrase']}"})


# ----------------------------------------------------------------------------
# Savoir si une alerte désigne bien la phrase piège
# ----------------------------------------------------------------------------
def mots(texte: str) -> set[str]:
    """Les mots « significatifs » d'un texte : normalisés (sans accents, minuscules), 4 lettres ou plus.
    mots("J’ai encadré une équipe de cinq stagiaires.") -> {"encadre", "equipe", "cinq", "stagiaires"}"""
    return {m for m in re.findall(r"[a-z0-9]+", normaliser(texte)) if len(m) >= 4}


# ============================================================================
# À TOI DE JOUER
# ============================================================================
def detectee(phrase_piege: str, alertes: list[str]) -> bool:
    cible = mots(phrase_piege)

    for alerte in alertes:
        communs = cible & mots(alerte)
        if len(communs) >= len(cible) / 2:
            return True

    return False


# ----------------------------------------------------------------------------
# Lancer le relecteur seul sur une lettre (même logique que le nœud « relire »)
# ----------------------------------------------------------------------------
def creer_relecteur():
    modele = ChatGroq(model=MODELE, temperature=0, reasoning_effort="low")
    return modele.with_structured_output(Relecture, method="json_schema", strict=True).with_retry(
        retry_if_exception_type=(BadRequestError, RateLimitError), stop_after_attempt=4,
        exponential_jitter_params={"initial": 5, "max": 30})


def construire_contexte(session, fiche, texte_offre: str, moteur) -> str:
    """Les mêmes sources que le nœud « preparer » de l'étape 7."""
    analyse = evaluer_offre(session.cv.competences, fiche)
    guide = formater_contexte(moteur.rechercher("structure et longueur d'une lettre de motivation",
                                                k=3, type_document="guide"))
    return (f"<cv>\n{session.texte_cv}\n</cv>\n\n<offre>\n{texte_offre}\n</offre>\n\n"
            f"<analyse>\nPrésentes : {', '.join(analyse['satisfaites']) or 'aucune'}\n"
            f"Manquantes : {', '.join(analyse['manquantes']) or 'aucune'}\n</analyse>\n\n"
            f"<guide>\n{guide}\n</guide>")


def relire_seul(relecteur, lettre: Lettre, contexte: str, texte_cv: str, texte_offre: str,
                reference: str | None) -> dict:
    """Renvoie séparément les alertes des RÈGLES et celles du relecteur (SENS), pour les comparer."""
    regles = verifier_lettre(lettre, texte_cv, texte_offre, reference)
    relecture = relecteur.invoke([
        SystemMessage(PROMPT_RELECTEUR),
        HumanMessage(f"{contexte}\n\n<lettre>\n{mettre_en_forme(lettre, None)}\n</lettre>"),
    ])
    sens = ([a.phrase for a in relecture.affirmations if not a.soutenue]
            + [a.phrase for a in relecture.affirmations if a.soutenue and not preuve_valide(a.preuve, contexte)])
    return {"regles": regles, "sens": sens, "nb_affirmations": len(relecture.affirmations)}