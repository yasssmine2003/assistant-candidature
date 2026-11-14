"""
Base vectorielle Qdrant Cloud : remplace l'index NumPy écrit à la main au projet 2.

Ce qui NE change pas :
- le modèle d'embeddings (on fournit nous-mêmes les vecteurs, avec encoder du projet 2) ;
- l'interface : ajouter(), rechercher(), len(), chunks -> on peut la brancher partout
  où l'on utilisait IndexVectoriel, y compris dans RechercheHybride, sans rien modifier.

Ce que Qdrant fait à notre place :
- le stockage, en ligne, consultable dans un tableau de bord ;
- la recherche rapide à grande échelle (index HNSW, recherche approximative) ;
- les filtres par métadonnées (appelées « payload » dans Qdrant) ;
- l'ajout, la mise à jour et la suppression d'un document sans tout reconstruire.

La connexion se fait avec QDRANT_URL et QDRANT_API_KEY, lus dans le fichier .env.
"""

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.rag.decoupage import Chunk
from src.rag.embeddings import encoder
from src.rag.index import ResultatRecherche

load_dotenv()
DIMENSION = 384  # taille des vecteurs de notre modèle d'embeddings


def type_document(source: str) -> str:
    """'offre_stage_devops.md' -> 'offre' (le préfixe du nom de fichier)."""
    return source.split("_")[0]


def identifiant_point(chunk_id: str) -> str:
    """Qdrant n'accepte que des entiers ou des UUID comme identifiants de points.
    uuid5 transforme toujours le même texte en le même UUID : relancer ne crée pas de doublons."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def filtre_egal(cle: str, valeur: str) -> models.Filter:
    """Un filtre Qdrant « le champ `cle` du payload vaut `valeur` »."""
    return models.Filter(must=[models.FieldCondition(key=cle, match=models.MatchValue(value=valeur))])


# ============================================================================
# À TOI DE JOUER : traduire la réponse de Qdrant dans nos objets
# ============================================================================
def convertir_resultats(points) -> list[ResultatRecherche]:
    """Transforme les points renvoyés par Qdrant en liste de ResultatRecherche.

    Chaque point a deux attributs utiles :
        point.score    -> la similarité cosinus (déjà un score : pas de conversion à faire)
        point.payload  -> le dictionnaire des métadonnées :
                          {"chunk_id": "offre_stage_devops.md#2", "source": "...",
                           "section": "...", "texte": "...", "nb_mots": 12, "type": "offre"}

    Pour chaque point, reconstruire :
        Chunk(id=..., source=..., section=..., texte=..., nb_mots=...)
    puis :
        ResultatRecherche(chunk=..., score=...)
    Aucun point -> liste vide.
    """
    resultats = []
    for p in points:
        m = p.payload
        chunk = Chunk(id=m["chunk_id"], source=m["source"], section=m["section"], texte=m["texte"], nb_mots=m["nb_mots"])
        resultats.append(ResultatRecherche(chunk=chunk, score=p.score))
    return resultats


class BaseVectorielle:
    def __init__(self, nom: str = "passages") -> None:
        self.nom = nom
        self.client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        if not self.client.collection_exists(nom):
            # Une collection = une « table » de vecteurs, tous de la même taille
            self.client.create_collection(
                nom, vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE)
            )
        # Index sur les champs qu'on filtre : Qdrant Cloud l'exige pour filtrer rapidement
        for champ in ("type", "source"):
            self.client.create_payload_index(nom, field_name=champ, field_schema=models.PayloadSchemaType.KEYWORD)

    def __len__(self) -> int:
        return self.client.count(self.nom, exact=True).count

    def ajouter(self, chunks: list[Chunk]) -> None:
        """Ajoute (ou met à jour) des passages. upsert : relancer ne crée pas de doublons."""
        if not chunks:
            return
        vecteurs = encoder([c.texte for c in chunks])  # MÊME modèle qu'avant
        points = [
            models.PointStruct(
                id=identifiant_point(c.id),
                vector=v.tolist(),
                payload={"chunk_id": c.id, "source": c.source, "section": c.section,
                         "texte": c.texte, "nb_mots": c.nb_mots, "type": type_document(c.source)},
            )
            for c, v in zip(chunks, vecteurs)
        ]
        self.client.upsert(self.nom, points=points)

    def supprimer_document(self, source: str) -> None:
        """Retire tous les passages d'un fichier, sans toucher au reste."""
        self.client.delete(self.nom, points_selector=models.FilterSelector(filter=filtre_egal("source", source)))

    def rechercher(self, requete: str, k: int = 3, type_document: str | None = None) -> list[ResultatRecherche]:
        """Même rôle que IndexVectoriel.rechercher, avec un filtre appliqué PAR Qdrant."""
        reponse = self.client.query_points(
            self.nom,
            query=encoder([requete])[0].tolist(),
            limit=k,
            query_filter=filtre_egal("type", type_document) if type_document else None,
            with_payload=True,
        )
        return convertir_resultats(reponse.points)

    @property
    def chunks(self) -> list[Chunk]:
        """Tous les passages : RechercheHybride en a besoin pour construire BM25."""
        points, _ = self.client.scroll(self.nom, limit=10_000, with_payload=True)
        return [
            Chunk(id=p.payload["chunk_id"], source=p.payload["source"], section=p.payload["section"],
                  texte=p.payload["texte"], nb_mots=p.payload["nb_mots"])
            for p in points
        ]