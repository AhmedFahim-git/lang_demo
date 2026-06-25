from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from itertools import chain
from typing import Any, Sequence

from config import (
    EMBED_DIM,
    EMBED_MODEL,
    ES_INDEX_NAME,
    ES_URL,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)
from elasticsearch import Elasticsearch, helpers
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
)
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from qdrant_client import QdrantClient, models


class BaseSearchStore(ABC):
    @abstractmethod
    def add_delete_nodes(
        self,
        nodes_to_add: Sequence[BaseNode] = [],
        node_ids_to_delete: Sequence[str] = [],
    ) -> None: ...

    @abstractmethod
    def search_query(
        self,
        query: str,
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
        size: int = 5,
    ) -> VectorStoreQueryResult: ...

    @abstractmethod
    def sync(
        self,
        ground_truth: set[str],
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
    ) -> None: ...


class ElasticsearchStore(BaseSearchStore):
    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        es_url: str = ES_URL,
        # ca_certs="http_ca.crt",
        # username: str = "elastic",
        # password: str = "KXW-oa-1CH+cJu7ODM35",
        delete_index_if_exists: bool = False,
    ) -> None:
        mappings = {
            i: {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            }
            for i in ["file_path", "file_name", "file_type"]
        } | {"content": {"type": "text"}, "file_size": {"type": "integer"}}
        self.index_name = index_name

        self.client = Elasticsearch(
            es_url,
            # ca_certs=ca_certs,
            # basic_auth=(username, password),
        )

        index_exist = self.client.indices.exists(index=self.index_name)

        if index_exist and delete_index_if_exists:
            self.client.indices.delete(index=self.index_name)
            index_exist = False
        if not index_exist:
            self.client.indices.create(
                index=self.index_name, mappings={"properties": mappings}
            )

    # Taken from https://github.com/run-llama/llama_index/blob/9aa5ee5cd2a1ecff6ffa2a8cd6af46b87af674b9/llama-index-integrations/vector_stores/llama-index-vector-stores-elasticsearch/llama_index/vector_stores/elasticsearch/base.py
    @staticmethod
    def _to_llama_similarities(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]
        return [(x - min_score) / (max_score - min_score) for x in scores]

    def _generate_docs_to_add(
        self, nodes: Sequence[BaseNode]
    ) -> Generator[dict[str, Any]]:
        for node in nodes:
            yield {
                "_index": self.index_name,
                "_id": node.node_id,
                "content": node.get_content(metadata_mode=MetadataMode.NONE),
                "file_path": node.metadata.get("file_path"),
                "file_name": node.metadata.get("file_name"),
                "file_type": node.metadata.get("file_type"),
                "file_size": node.metadata.get("file_size"),
            }

    def _generate_docs_to_delete(
        self,
        node_ids: Sequence[str],
    ) -> Generator[dict[str, Any]]:
        for node_id in node_ids:
            yield {"_op_type": "delete", "_index": self.index_name, "_id": node_id}

    def add_delete_nodes(
        self,
        nodes_to_add: Sequence[BaseNode] = [],
        node_ids_to_delete: Sequence[str] = [],
    ) -> None:
        actions = []
        if node_ids_to_delete:
            actions.append(self._generate_docs_to_delete(node_ids=node_ids_to_delete))
        if nodes_to_add:
            actions.append(self._generate_docs_to_add(nodes=nodes_to_add))
        if actions:
            helpers.bulk(
                client=self.client,
                actions=chain(*actions),
                refresh=True,
                ignore_status=404,
            )

    @staticmethod
    def _es_hits_to_query_result(
        hits: list[dict], node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]]
    ) -> VectorStoreQueryResult:
        top_scores = []
        top_ids = []
        for hit in hits:
            top_ids.append(hit["_id"])
            top_scores.append(hit["_score"])
        node_dict = node_fetch_fn(top_ids)
        top_nodes = [node_dict[i] for i in top_ids]

        return VectorStoreQueryResult(
            nodes=top_nodes,
            similarities=ElasticsearchStore._to_llama_similarities(top_scores),
            ids=top_ids,
        )

    def search_query(
        self,
        query: str,
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
        size: int = 5,
    ) -> VectorStoreQueryResult:
        results = self.client.search(
            index=self.index_name,
            query={"match": {"content": query}},
            size=size,
        )
        return self._es_hits_to_query_result(
            hits=results["hits"]["hits"], node_fetch_fn=node_fetch_fn
        )

    def search_detailed_es_query(
        self,
        query: dict,
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
        size: int = 5,
    ) -> VectorStoreQueryResult:
        results = self.client.search(
            index=self.index_name,
            query=query,
            size=size,
        )
        return self._es_hits_to_query_result(
            hits=results["hits"]["hits"], node_fetch_fn=node_fetch_fn
        )

    def sync(
        self,
        ground_truth: set[str],
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
    ) -> None:
        es_nodes: set[str] = {
            doc["_id"]
            for doc in helpers.scan(
                client=self.client,
                index=self.index_name,
                query={"query": {"match_all": {}}},
                _source=False,
            )
        }
        node_ids_to_delete = list(es_nodes - ground_truth)
        nodes_to_add = list(node_fetch_fn(list(ground_truth - es_nodes)).values())
        self.add_delete_nodes(
            nodes_to_add=nodes_to_add, node_ids_to_delete=node_ids_to_delete
        )


class QdrantStore(BaseSearchStore):
    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
        embedding_model: BaseEmbedding = EMBED_MODEL,
        embedding_dim: int = EMBED_DIM,
        delete_collection_if_exists: bool = False,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        col_exist = self.client.collection_exists(collection_name=self.collection_name)
        if col_exist and delete_collection_if_exists:
            self.client.delete_collection(collection_name=self.collection_name)
            col_exist = False
        if not col_exist:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(m=16),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_size",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        self.embedding_model = embedding_model

    def add_delete_nodes(
        self,
        nodes_to_add: Sequence[BaseNode] = [],
        node_ids_to_delete: Sequence[str] = [],
    ) -> None:
        self.client.update_collection(
            collection_name=self.collection_name, hnsw_config=models.HnswConfigDiff(m=0)
        )
        actions: list[models.UpdateOperation] = []
        if node_ids_to_delete:
            actions.append(
                models.DeleteOperation(
                    delete=models.PointIdsList(points=list(node_ids_to_delete))
                )
            )
        if nodes_to_add:
            vectors = self.embedding_model.get_text_embedding_batch(
                [i.get_content(metadata_mode=MetadataMode.EMBED) for i in nodes_to_add]
            )
            actions.append(
                models.UpsertOperation(
                    upsert=models.PointsList(
                        points=[
                            models.PointStruct(
                                id=node.node_id,
                                vector=vectors[i],
                                payload={
                                    "file_path": node.metadata.get("file_path"),
                                    "file_name": node.metadata.get("file_name"),
                                    "file_type": node.metadata.get("file_type"),
                                    "file_size": node.metadata.get("file_size"),
                                },
                            )
                            for i, node in enumerate(nodes_to_add)
                        ]
                    )
                )
            )
        if actions:
            self.client.batch_update_points(
                collection_name=self.collection_name, update_operations=actions
            )
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(m=16),
        )

    @staticmethod
    def _qdrant_points_to_query_result(
        points: list[models.ScoredPoint],
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
    ) -> VectorStoreQueryResult:
        top_scores = []
        top_ids = []
        for point in points:
            point = point.model_dump()
            top_ids.append(point["id"])
            top_scores.append(point["score"])
        node_dict = node_fetch_fn(top_ids)
        top_nodes = [node_dict[i] for i in top_ids]

        return VectorStoreQueryResult(
            nodes=top_nodes,
            similarities=top_scores,
            ids=top_ids,
        )

    def search_query(
        self,
        query: str,
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
        size: int = 5,
    ) -> VectorStoreQueryResult:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding_model.get_text_embedding(query),
            limit=size,
        )
        return self._qdrant_points_to_query_result(
            points=results.points, node_fetch_fn=node_fetch_fn
        )

    def search_detailed_qdrant_query(
        self,
        query: str,
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
        query_filter: models.Filter | None = None,
        search_params: models.SearchParams | None = None,
        size: int = 5,
    ) -> VectorStoreQueryResult:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding_model.get_text_embedding(query),
            limit=size,
            query_filter=query_filter,
            search_params=search_params,
        )
        return self._qdrant_points_to_query_result(
            points=results.points, node_fetch_fn=node_fetch_fn
        )

    def sync(
        self,
        ground_truth: set[str],
        node_fetch_fn: Callable[[list[str]], dict[str, BaseNode]],
    ) -> None:
        qdrant_nodes: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            qdrant_nodes.update(str(point.id) for point in points)
            if offset is None:
                break
        node_ids_to_delete = list(qdrant_nodes - ground_truth)
        nodes_to_add = list(node_fetch_fn(list(ground_truth - qdrant_nodes)).values())
        self.add_delete_nodes(
            nodes_to_add=nodes_to_add, node_ids_to_delete=node_ids_to_delete
        )
