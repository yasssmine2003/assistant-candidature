"""
La session : le dossier personnel d'UNE utilisatrice pendant sa conversation.

Deux sortes de données dans l'assistant :
- partagées (base d'offres, index) : les mêmes pour tout le monde, préparées au démarrage ;
- de session (CV, fiche extraite, historique) : propres à une utilisatrice, jamais mélangées.

Le CV est contrôlé puis extrait UNE SEULE FOIS, au chargement. Ensuite, tous les outils
lisent la fiche gardée dans la session : aucun nouvel appel au LLM pour le CV.

On réutilise les briques du projet 1 (dans src/candidature/) sans les modifier.
"""

import uuid
from dataclasses import dataclass, field

from src.candidature.extraction import extraire_cv
from src.candidature.modeles import CV
from src.candidature.securite import detecter_injection, verifier_ancrage


class CVSuspect(Exception):
    """Levée quand un CV contient des formulations qui ressemblent à une injection."""


@dataclass
class Session:
    # Un identifiant unique, pour retrouver la session (utile plus tard pour l'API)
    identifiant: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    texte_cv: str | None = None
    cv: CV | None = None                                       # la fiche extraite, gardée pour toute la conversation
    alertes_ancrage: list[str] = field(default_factory=list)  # compétences extraites absentes du texte
    historique: list[dict] = field(default_factory=list)       # les messages de la conversation (étape 4)

    @property
    def cv_charge(self) -> bool:
        return self.cv is not None

    def charger_cv(self, texte: str) -> CV:
        """Les gestes de la session : contrôler, extraire une fois, vérifier, garder."""
        # 1. Contrôler AVANT tout appel au LLM : un CV est une entrée écrite par l'utilisatrice
        suspects = detecter_injection(texte)
        if suspects:
            raise CVSuspect(f"Formulations suspectes dans le CV : {suspects}")

        # 2. Extraire une seule fois (un appel au LLM, fonction du projet 1)
        cv = extraire_cv(texte)

        # 3. Vérifier que les compétences extraites sont bien écrites dans le CV
        self.alertes_ancrage = verifier_ancrage(cv, texte)

        # 4. Garder dans la session
        self.texte_cv, self.cv = texte, cv
        return cv

    # ========================================================================
    # À TOI DE JOUER : le résumé du CV que l'agent verra
    # ========================================================================
    def resume_cv(self) -> str:
        """Résumé du CV en texte, pour l'agent (même idée que lister_offres à l'étape 1).

        Format exact :
            Nom : Yasmine Bouchami
            Compétences : Python, Machine Learning, SQL
            Expériences : Développeuse Python (ABC Tech, 2025-01 -> 2025-06) ; Stagiaire IA (XYZ AI, 2026-09 -> en cours)

        Règles :
        - aucun CV chargé -> "Aucun CV chargé."
        - nom absent -> "?" ; entreprise absente -> "?" ; date_fin absente -> "en cours"
        - aucune expérience -> "Expériences : aucune"
        - les trois lignes sont séparées par "\\n"
        """
        if self.cv is None:
            return "Aucun CV chargé."
        exps = []
        for e in self.cv.experiences:
            exps.append(f"{e.poste} ({e.entreprise or '?'}, {e.date_debut} -> {e.date_fin or 'en cours'})")
        return (f"Nom : {self.cv.nom or '?'}\n"
                f"Compétences : {', '.join(self.cv.competences)}\n"
                f"Expériences : {' ; '.join(exps) if exps else 'aucune'}")


# ============================================================================
# Le registre des sessions : une session par conversation
# ============================================================================
SESSIONS: dict[str, Session] = {}


def creer_session() -> Session:
    session = Session()
    SESSIONS[session.identifiant] = session
    return session


def obtenir_session(identifiant: str) -> Session:
    if identifiant not in SESSIONS:
        raise KeyError(f"Session inconnue : {identifiant}")
    return SESSIONS[identifiant]