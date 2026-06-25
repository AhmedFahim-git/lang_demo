from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator
from copy import deepcopy
from datetime import datetime, timezone
from functools import reduce
from io import BytesIO
from itertools import chain
from typing import Any, Literal, NamedTuple, Optional, Sequence

import filetype
import fsspec
import pymongo
import requests
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from fastapi import FastAPI
from llama_index.core import Document
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from pydantic import BaseModel
from pymongo import MongoClient
from qdrant_client import QdrantClient, models

EMBED_MODEL = OpenAILikeEmbedding(
    model_name="Qwen0.6B", api_base="http://localhost:8080/v1/"
)
EMBED_DIM = 1024

# URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{os.environ['MONGO_SERVICE_HOSTNAME']}:27017/?authSource=admin"  # "mongodb://root:example@localhost:27017/?authSource=admin"
MONGO_URI = "mongodb://root:example@localhost:27017/?authSource=admin"
ES_INDEX_NAME = "es_index"
RERANK_BASE_URL = "http://localhost:8081/v1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "collection_name"
app = FastAPI()


def rerank_docs(
    query: str, docs: list[str], base_url: str = RERANK_BASE_URL
) -> list[tuple[int, float]]:
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    r = requests.post(base_url + "rerank", json={"query": query, "documents": docs})
    r.raise_for_status()
    res_json = r.json()
    return [(i["index"], i["relevance_score"]) for i in res_json["results"]]


def rerank_nodes(
    query: str,
    nodes: Sequence[NodeWithScore],
    top_k: int | None = None,
    base_url: str = RERANK_BASE_URL,
) -> list[NodeWithScore]:
    docs = [i.get_content(metadata_mode=MetadataMode.LLM) for i in nodes]
    ranked_docs = rerank_docs(query=query, docs=docs, base_url=base_url)
    if top_k:
        ranked_docs = ranked_docs[:top_k]
    new_list = []
    for i, score in ranked_docs:
        cur_node = nodes[i]
        cur_node.score = score
        new_list.append(cur_node)
    return new_list


class LlamaElasticsearch:
    @staticmethod
    def get_client(
        index_name: str = ES_INDEX_NAME,
        ca_certs="http_ca.crt",
        username: str = "elastic",
        password: str = "yNZKq+=90SMk5yl-8y95",
        delete_index_if_exists: bool = True,
    ) -> Elasticsearch:
        metadata_mappings = {
            "document_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "ref_doc_id": {"type": "keyword"},
        }

        client = Elasticsearch(
            "https://localhost:9200",
            ca_certs=ca_certs,
            basic_auth=(username, password),
        )

        index_exist = client.indices.exists(index=index_name)

        if index_exist and delete_index_if_exists:
            client.indices.delete(index=index_name)
            index_exist = False
        if not index_exist:
            client.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        "content": {"type": "text"},
                        "metadata": {"type": "object", "properties": metadata_mappings},
                    }
                },
            )
        return client

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

    @staticmethod
    def _generate_docs_for_es(nodes: Sequence[BaseNode]) -> Generator[dict[str, Any]]:
        for node in nodes:
            yield {
                "_index": ES_INDEX_NAME,
                "_id": node.node_id,
                "content": node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": node_to_metadata_dict(node, remove_text=True),
            }

    @staticmethod
    def put_nodes_in_es(client: Elasticsearch, nodes: Sequence[BaseNode]) -> None:
        helpers.bulk(
            client=client,
            actions=LlamaElasticsearch._generate_docs_for_es(nodes=nodes),
            refresh=True,
        )

    @staticmethod
    def source_to_node(source: dict) -> BaseNode:
        metadata, text = source["metadata"], source["content"]
        return metadata_dict_to_node(metadata=metadata, text=text)

    @staticmethod
    def search_es(
        client: Elasticsearch, query: dict, index: str = ES_INDEX_NAME, size: int = 5
    ) -> VectorStoreQueryResult:
        results = client.search(
            index=index,
            query=query,
            size=size,
        )
        top_nodes = []
        top_scores = []
        top_ids = []
        for hit in results["hits"]["hits"]:
            top_ids.append(hit["_id"])
            top_scores.append(hit["_score"])
            top_nodes.append(LlamaElasticsearch.source_to_node(hit["_source"]))

        return VectorStoreQueryResult(
            nodes=top_nodes,
            similarities=LlamaElasticsearch._to_llama_similarities(top_scores),
            ids=top_ids,
        )

    @staticmethod
    def get_node_by_id(
        client: Elasticsearch, id: str, index: str = ES_INDEX_NAME
    ) -> BaseNode | None:
        try:
            res = client.get(index=index, id=id)
            return LlamaElasticsearch.source_to_node(res["_source"])
        except NotFoundError:
            return None


def dedup_nodes_with_score(nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
    hash_dict: dict[str, NodeWithScore] = dict()
    for node in nodes:
        hash = node.node.hash
        if (hash in hash_dict) and (hash_dict[hash].get_score() > node.get_score()):
            continue
        hash_dict[hash] = node
    return sorted(list(hash_dict.values()), key=lambda x: x.get_score(), reverse=True)


class MergeConsecutiveNodesPostprocessor(BaseNodePostprocessor):
    @classmethod
    def class_name(cls) -> str:
        return "MergeConsecutiveNodesPostprocessor"

    @staticmethod
    def _join_nodes(
        prev_node_with_score: NodeWithScore, next_node_with_score: NodeWithScore
    ) -> NodeWithScore:
        prev_node_with_score = deepcopy(prev_node_with_score)
        prev_node = prev_node_with_score.node
        assert isinstance(prev_node, TextNode)
        prev_node_start_end = prev_node.get_node_info()
        next_node = next_node_with_score.node
        assert isinstance(next_node, TextNode)
        next_node_start_end = next_node.get_node_info()
        if (prev_node_start_end.get("end") is not None) and (
            next_node_start_end.get("start") is not None
        ):
            new_content = (
                prev_node.get_content(metadata_mode=MetadataMode.NONE)
                + next_node.get_content(metadata_mode=MetadataMode.NONE)[
                    prev_node_start_end["end"] - next_node_start_end["start"] :
                ]
            )
        else:
            new_content = (
                prev_node.get_content(metadata_mode=MetadataMode.NONE)
                + "\n"
                + next_node.get_content(metadata_mode=MetadataMode.NONE)
            )
        new_score = max(
            next_node_with_score.get_score(),
            prev_node_with_score.get_score(),
        )
        prev_node.set_content(new_content)
        prev_node_with_score.score = new_score
        if (NodeRelationship.NEXT in next_node.relationships) and (
            next_node.next_node is not None
        ):
            prev_node.relationships[NodeRelationship.NEXT] = next_node.next_node
        else:
            prev_node.relationships.pop(NodeRelationship.NEXT)
        # if NodeRelationship.NEXT in next_node.relationships:
        #     prev_node.relationships[NodeRelationship.NEXT] = next_node.relationships[
        #         NodeRelationship.NEXT
        #     ]
        prev_node.end_char_idx = next_node_start_end.get("end")
        return prev_node_with_score

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        nodes = dedup_nodes_with_score(nodes=nodes)
        all_nodes: dict[str, NodeWithScore] = {
            node.node.node_id: node for node in nodes
        }
        # for node in nodes:
        #     if (node.node.node_id in all_nodes) and (
        #         node.get_score() <= all_nodes[node.node.node_id].get_score()
        #     ):
        #         continue
        #     all_nodes[node.node.node_id] = node
        node_id_set = set(all_nodes.keys())
        new_list = []
        while node_id_set:
            node_id = node_id_set.pop()
            node_with_score = all_nodes[node_id]
            node = node_with_score.node
            node_deque = deque([node_id])
            while True:
                if NodeRelationship.PREVIOUS in node.relationships:
                    prev_node_info = node.prev_node
                    if prev_node_info is not None:
                        prev_node_id = prev_node_info.node_id
                        if prev_node_id in all_nodes:
                            prev_node_with_score = all_nodes[prev_node_id]
                            assert (
                                (
                                    NodeRelationship.NEXT
                                    in prev_node_with_score.node.relationships
                                )
                                and (prev_node_with_score.node.next_node is not None)
                                and (
                                    prev_node_with_score.node.next_node.node_id
                                    == node_deque[0]
                                )
                            ), (
                                f"{node_deque}\n Cur node: relation{node.relationships}, metadata{node.metadata}, content:{node.get_content(MetadataMode.NONE)}, ID: {node_id}\n Prev node: relation{prev_node_with_score.node.relationships} metadata: {prev_node_with_score.node.metadata}, content:{prev_node_with_score.node.get_content(MetadataMode.NONE)}, ID: {prev_node_id}"
                            )
                            node_deque.appendleft(prev_node_id)
                            node_with_score = self._join_nodes(
                                prev_node_with_score=prev_node_with_score,
                                next_node_with_score=node_with_score,
                            )
                            node = node_with_score.node
                            node_id_set.discard(prev_node_id)
                            continue
                break
            while True:
                if NodeRelationship.NEXT in node.relationships:
                    next_node_info = node.next_node
                    if next_node_info is not None:
                        next_node_id = next_node_info.node_id
                        if next_node_id in all_nodes:
                            next_node_with_score = all_nodes[next_node_id]
                            assert (
                                (
                                    NodeRelationship.PREVIOUS
                                    in next_node_with_score.node.relationships
                                )
                                and (next_node_with_score.node.prev_node is not None)
                                and (
                                    next_node_with_score.node.prev_node.node_id
                                    == node_deque[-1]
                                )
                            )
                            node_deque.append(next_node_id)
                            node_with_score = self._join_nodes(
                                prev_node_with_score=node_with_score,
                                next_node_with_score=next_node_with_score,
                            )
                            node = node_with_score.node
                            node_id_set.discard(next_node_id)
                            continue
                break
            new_list.append(node_with_score)
        return new_list


class PrevNextNodePostprocessor(BaseNodePostprocessor):
    """
    Previous/Next Node post-processor.

    Allows users to fetch additional nodes from the document store,
    based on the relationships of the nodes.

    NOTE: this is a beta feature.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.

    """

    id_node_fn: Callable[[str], BaseNode | None]
    num_nodes: int = Field(default=1)
    mode: str = Field(default="next")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        """Validate mode."""
        if v not in ["next", "previous", "both"]:
            raise ValueError(f"Invalid mode: {v}")
        return v

    @classmethod
    def class_name(cls) -> str:
        return "PrevNextNodePostprocessor"

    @staticmethod
    def get_for_back_nodes(
        node_with_score: NodeWithScore,
        num_nodes: int,
        id_node_fn: Callable[[str], BaseNode | None],
        direction: Literal["prev", "next"],
    ) -> dict[str, NodeWithScore]:
        """Get forward nodes."""
        node = node_with_score.node
        nodes: dict[str, NodeWithScore] = {node.node_id: node_with_score}
        cur_count = 0
        # get forward and backward nodes in an iterative manner
        while cur_count < num_nodes:
            if direction == "next":
                if NodeRelationship.NEXT not in node.relationships:
                    break
                new_node_info = node.next_node
            elif direction == "prev":
                if NodeRelationship.PREVIOUS not in node.relationships:
                    break
                new_node_info = node.prev_node

            if new_node_info is None:
                break
            new_node_id = new_node_info.node_id
            new_node = id_node_fn(new_node_id)
            if new_node is None:
                break

            next_node_info = node.next_node
            if next_node_info is None:
                break

            nodes[new_node.node_id] = NodeWithScore(node=new_node)
            node = new_node
            cur_count += 1
        return nodes

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        all_nodes: dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node
            if self.mode == "next":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
            elif self.mode == "previous":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="prev",
                    )
                )
            elif self.mode == "both":
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
                all_nodes.update(
                    self.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="prev",
                    )
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        all_nodes_values: list[NodeWithScore] = list(all_nodes.values())
        sorted_nodes: list[NodeWithScore] = []
        for node in all_nodes_values:
            # variable to check if cand node is inserted
            node_inserted = False
            for i, cand in enumerate(sorted_nodes):
                node_id = node.node.node_id
                # prepend to current candidate
                prev_node_info = cand.node.prev_node
                next_node_info = cand.node.next_node
                if prev_node_info is not None and node_id == prev_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i, node)
                    break
                # append to current candidate
                elif next_node_info is not None and node_id == next_node_info.node_id:
                    node_inserted = True
                    sorted_nodes.insert(i + 1, node)
                    break

            if not node_inserted:
                sorted_nodes.append(node)

        return sorted_nodes


# Adapted from https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/fusion_retriever.py#L113
def reciprocal_rerank_fusion(
    results: Sequence[Sequence[NodeWithScore]],
) -> list[NodeWithScore]:
    """
    Apply reciprocal rank fusion.

    The original paper uses k=60 for best results:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    hash_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results:
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            hash = node_with_score.node.hash
            hash_to_node[hash] = node_with_score
            if hash not in fused_scores:
                fused_scores[hash] = 0.0
            fused_scores[hash] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: list[NodeWithScore] = []
    for hash, score in reranked_results.items():
        reranked_nodes.append(hash_to_node[hash])
        reranked_nodes[-1].score = score

    return reranked_nodes


# fs = fsspec.filesystem("s3", endpoint_url="http://localhost:4566")
# converter = DocumentConverter()


def get_doc_from_fpath(
    file_path: str,
    fs: fsspec.AbstractFileSystem,
    converter: DocumentConverter,
) -> Document:
    excluded_metadata_keys = [
        "file_name",
        "file_type",
        "file_size",
        "creation_date",
        "last_modified_date",
        "last_accessed_date",
    ]
    f_info = fs.info(file_path)
    metadata = dict()
    metadata["file_path"] = f_info.get("name")
    metadata["file_name"] = file_path.split("/")[-1]
    metadata["file_type"] = f_info.get("ContentType")
    metadata["file_size"] = f_info.get("size")
    metadata["creation_date"] = f_info.get(
        "LastModified", datetime.now(timezone.utc)
    ).isoformat()
    metadata["last_modified_date"] = f_info.get(
        "LastModified", datetime.now(timezone.utc)
    ).isoformat()
    # The issue is that using datetime.now is problematic as datetime changes each time ingestion is done. So node hash changes even if node content did not change.
    # metadata["last_accessed_date"] = datetime.now(timezone.utc).isoformat()
    with fs.open(file_path, "rb") as f:
        f_bytes = f.read()
        assert isinstance(f_bytes, bytes)
        if metadata["file_type"] is None:
            metadata["file_type"] = filetype.guess_mime(f_bytes)
        f_bytes = BytesIO(f_bytes)
    loaded_doc = converter.convert(
        DocumentStream(name=metadata["file_name"], stream=f_bytes)
    )
    doc = Document(text=loaded_doc.document.export_to_markdown(), metadata=metadata)
    doc.excluded_llm_metadata_keys.extend(excluded_metadata_keys)
    doc.excluded_embed_metadata_keys.extend(excluded_metadata_keys)
    return doc


def get_docs_from_dir(
    dir_name: str,
    fs: fsspec.AbstractFileSystem,
    converter: DocumentConverter,
) -> list[Document]:
    docs = []
    for file in fs.ls(dir_name):
        docs.append(get_doc_from_fpath(file_path=file, fs=fs, converter=converter))
    return docs


def postprocess_ingestion_nodes(nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
    nodes = deepcopy(nodes)
    for i in range(1, len(nodes)):
        prev_node, next_node = nodes[i - 1], nodes[i]
        if prev_node.metadata["file_path"] == next_node.metadata["file_path"]:
            assert (
                (NodeRelationship.NEXT in prev_node.relationships)
                and (prev_node.next_node is not None)
                and (prev_node.next_node.node_id == next_node.node_id)
            ) or (
                (NodeRelationship.PREVIOUS in next_node.relationships)
                and (next_node.prev_node is not None)
                and (next_node.prev_node.node_id == prev_node.node_id)
            ), "At least one side should have the required relation"
            prev_node.relationships[NodeRelationship.NEXT] = (
                next_node.as_related_node_info()
            )
            next_node.relationships[NodeRelationship.PREVIOUS] = (
                prev_node.as_related_node_info()
            )
    return nodes


class NodeDiff(NamedTuple):
    node_ids_delete: list[str]
    nodes_add: list[BaseNode]


class BaseDocstore(ABC):
    @abstractmethod
    def get_nodes_diff(self, nodes: Sequence[BaseNode]) -> NodeDiff: ...
    @abstractmethod
    def get_node(self, node_id: str) -> BaseNode | None: ...
    @abstractmethod
    def get_nodes(self, node_ids: list[str]) -> dict[str, BaseNode]: ...
    @abstractmethod
    def get_all_node_ids(self) -> set[str]: ...


# def get_node_hash(node: BaseNode) -> str:
#     metadata_keys_to_use: list[str] = ["file_name", "file_type", "file_size"]
#     metadata_str = node.metadata_separator.join(
#         [
#             node.metadata_template.format(key=key, value=str(value))
#             for key, value in node.metadata.items()
#             if key in metadata_keys_to_use
#         ]
#     )
#     doc_identity = str(node.get_content(metadata_mode=MetadataMode.NONE)) + metadata_str
#     return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())


class MongoDocstore(BaseDocstore):
    def __init__(
        self,
        uri: str = MONGO_URI,
        database_name: str = "init_db_mongo",
        doc_collection_name: str = "doc_collection_name",
        node_collection_name: str = "node_collection_name",
    ) -> None:
        self.client = MongoClient(uri)
        self.db = self.client.get_database(database_name)
        self.doc_collection = self.db.get_collection(doc_collection_name)
        self.node_collection = self.db.get_collection(node_collection_name)

    def get_nodes_diff(self, nodes: Sequence[BaseNode]) -> NodeDiff:
        node_dict = {node.node_id: node for node in nodes}
        print("num items in doc collection", self.doc_collection.count_documents({}))
        print("num items in node collection", self.node_collection.count_documents({}))

        docs_to_add = dict()
        docs_to_delete = []
        nodes_to_add = []
        nodes_to_delete = []
        for node in nodes:
            if (NodeRelationship.SOURCE in node.relationships) and (
                node.source_node is not None
            ):
                source_node = node.source_node
                if self.doc_collection.find_one({"hash": source_node.hash}) is not None:
                    # If doc matches we consider that it is already added
                    continue
                res = self.doc_collection.find_one(
                    {
                        "file_name": source_node.metadata.get("file_name"),
                        "file_type": source_node.metadata.get("file_type"),
                    }
                )
                if res is not None:
                    if source_node.node_id not in docs_to_add:
                        nodes_to_delete.extend(res.get("child_nodes", []))
                        docs_to_delete.append(res.get("_id"))
                if source_node.node_id not in docs_to_add:
                    docs_to_add[source_node.node_id] = source_node.metadata | {
                        "hash": source_node.hash,
                        "child_nodes": [node.node_id],
                    }
                else:
                    docs_to_add[source_node.node_id]["child_nodes"].append(node.node_id)
            elif self.node_collection.find_one({"hash": node.hash}) is not None:
                continue
            nodes_to_add.append(
                {
                    "_id": node.node_id,
                    "content": node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": node_to_metadata_dict(node, remove_text=True),
                    "hash": node.hash,
                }
            )
        doc_operations = []
        if docs_to_delete:
            doc_operations.append(pymongo.DeleteMany({"_id": {"$in": docs_to_delete}}))
        if docs_to_add:
            doc_operations.extend(
                [pymongo.InsertOne({"_id": k} | v) for k, v in docs_to_add.items()]
            )
        # doc_operations = [pymongo.DeleteMany({"_id": {"$in": docs_to_delete}})] + [
        #     pymongo.InsertOne({"_id": k} | v) for k, v in docs_to_add.items()
        # ]
        node_operations = []
        if nodes_to_delete:
            node_operations.append(
                pymongo.DeleteMany({"_id": {"$in": nodes_to_delete}})
            )
        if nodes_to_add:
            node_operations.extend([pymongo.InsertOne(i) for i in nodes_to_add])
        # node_operations = [pymongo.DeleteMany({"_id": {"$in": nodes_to_delete}})] + [
        #     pymongo.InsertOne(i) for i in nodes_to_add
        # ]
        print(
            f"before docstore operations. Doc operations: {len(doc_operations)}\nNode operations:{len(node_operations)}"
        )
        # dict_return = {
        #     "delete_nodes": list(self.get_nodes(nodes_to_delete).values()),
        #     "add_nodes": [node_dict[i.get("_id")] for i in nodes_to_add],
        # }
        try:
            if doc_operations:
                self.doc_collection.bulk_write(doc_operations)
                print("doc operations done")
            if node_operations:
                self.node_collection.bulk_write(node_operations)
                print("node operations done")
        except Exception as e:
            print(e)
        print("num items in doc  collection", self.doc_collection.count_documents({}))
        print("num items in node collection", self.node_collection.count_documents({}))
        print("Len nodes_to_delete", len(nodes_to_delete))
        print("Len nodes_to_add", len(nodes_to_add))
        # print(
        #     "Number of nodes to delete:", len([node_dict[i] for i in nodes_to_delete])
        # )
        # print(
        #     "Number of nodes to add:",
        #     len([node_dict[i.get("_id")] for i in nodes_to_add]),
        # )
        # dict_return = {
        #     "delete_nodes": [node_dict[i] for i in nodes_to_delete],
        #     "add_nodes": [node_dict[i.get("_id")] for i in nodes_to_add],
        # }
        # print("in mango", dict_return)
        node_diff_return = NodeDiff(
            node_ids_delete=nodes_to_delete,
            nodes_add=[node_dict[i.get("_id")] for i in nodes_to_add],
        )
        return node_diff_return

    def get_node(self, node_id: str) -> BaseNode | None:
        res = self.node_collection.find_one({"_id": node_id})
        if res is None:
            return None
        return metadata_dict_to_node(
            metadata=res.get("metadata"), text=res.get("content")
        )

    def get_nodes(self, node_ids: list[str]) -> dict[str, BaseNode]:
        res = self.node_collection.find({"_id": {"$in": node_ids}})
        node_dict = dict()
        for item in res:
            node_dict[item.get("_id")] = metadata_dict_to_node(
                metadata=item.get("metadata"), text=item.get("content")
            )
        return node_dict

    def get_all_node_ids(self) -> set[str]:
        return {doc["_id"] for doc in self.node_collection.find({}, {"_id": 1})}


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
        ca_certs="http_ca.crt",
        username: str = "elastic",
        password: str = "KXW-oa-1CH+cJu7ODM35",
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
            "https://localhost:9200",
            ca_certs=ca_certs,
            basic_auth=(username, password),
        )

        index_exist = self.client.indices.exists(index=self.index_name)
        print(index_exist)

        if index_exist and delete_index_if_exists:
            self.client.indices.delete(index=self.index_name)
            index_exist = False
        if not index_exist:
            self.client.indices.create(
                index=self.index_name, mappings={"properties": mappings}
            )
        print("Init Es document count:", self.client.count(index=self.index_name))

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
        print("Es document before count:", self.client.count(index=self.index_name))
        print(
            f"Num nodes to delete: {len(node_ids_to_delete)}, Num nodes to add: {len(nodes_to_add)}"
        )
        actions = []
        if node_ids_to_delete:
            actions.append(self._generate_docs_to_delete(node_ids=node_ids_to_delete))
        if nodes_to_add:
            actions.append(self._generate_docs_to_add(nodes=nodes_to_add))
        if actions:
            print("Es actions", actions)
            helpers.bulk(
                client=self.client,
                actions=chain(*actions),
                refresh=True,
                ignore_status=404,
            )
            print("actions done")
        print("Es document after count:", self.client.count(index=self.index_name))

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
        print("Es document count:", self.client.count(index=self.index_name))
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
        # actions = []
        # if nodes_to_delete:
        #     actions.append(self._generate_docs_to_delete(node_ids=nodes_to_delete))
        # if nodes_to_add:
        #     actions.append(self._generate_docs_to_add(nodes=nodes_to_add))
        # if actions:
        #     print("Es actions", actions)
        #     helpers.bulk(
        #         client=self.client,
        #         actions=chain(*actions),
        #         refresh=True,
        #         ignore_status=404,
        #     )


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
        print(
            "Qdrant collection info pre:",
            self.client.get_collection(collection_name=self.collection_name),
        )
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
            # self.client.delete(
            #     collection_name=self.collection_name,
            #     points_selector=models.PointIdsList(
            #         points=[i.node_id for i in nodes_to_delete]
            #     ),
            # )
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
            # self.client.upsert(
            #     collection_name=self.collection_name,
            #     points=[
            #         models.PointStruct(
            #             id=node.node_id,
            #             vector=vectors[i],
            #             payload={
            #                 "file_path": node.metadata.get("file_path"),
            #                 "file_name": node.metadata.get("file_name"),
            #                 "file_type": node.metadata.get("file_type"),
            #                 "file_size": node.metadata.get("file_size"),
            #             },
            #         )
            #         for i, node in enumerate(nodes_to_add)
            #     ],
            # )
        if actions:
            self.client.batch_update_points(
                collection_name=self.collection_name, update_operations=actions
            )
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(m=16),
        )
        print(
            "Qdrant collection info after:",
            self.client.get_collection(collection_name=self.collection_name),
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
        print(
            "Qdrant collection info:",
            self.client.get_collection(collection_name=self.collection_name),
        )
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
        # self.client.update_collection(
        #     collection_name=self.collection_name, hnsw_config=models.HnswConfigDiff(m=0)
        # )
        # actions: list[models.UpdateOperation] = []
        # if node_ids_to_delete:
        #     actions.append(
        #         models.DeleteOperation(
        #             delete=models.PointIdsList(points=[i for i in node_ids_to_delete])
        #         )
        #     )
        #     # self.client.delete(
        #     #     collection_name=self.collection_name,
        #     #     points_selector=models.PointIdsList(
        #     #         points=[i.node_id for i in nodes_to_delete]
        #     #     ),
        #     # )
        # if nodes_to_add:
        #     vectors = self.embedding_model.get_text_embedding_batch(
        #         [i.get_content(metadata_mode=MetadataMode.EMBED) for i in nodes_to_add]
        #     )
        #     actions.append(
        #         models.UpsertOperation(
        #             upsert=models.PointsList(
        #                 points=[
        #                     models.PointStruct(
        #                         id=node.node_id,
        #                         vector=vectors[i],
        #                         payload={
        #                             "file_path": node.metadata.get("file_path"),
        #                             "file_name": node.metadata.get("file_name"),
        #                             "file_type": node.metadata.get("file_type"),
        #                             "file_size": node.metadata.get("file_size"),
        #                         },
        #                     )
        #                     for i, node in enumerate(nodes_to_add)
        #                 ]
        #             )
        #         )
        #     )
        #     # self.client.upsert(
        #     #     collection_name=self.collection_name,
        #     #     points=[
        #     #         models.PointStruct(
        #     #             id=node.node_id,
        #     #             vector=vectors[i],
        #     #             payload={
        #     #                 "file_path": node.metadata.get("file_path"),
        #     #                 "file_name": node.metadata.get("file_name"),
        #     #                 "file_type": node.metadata.get("file_type"),
        #     #                 "file_size": node.metadata.get("file_size"),
        #     #             },
        #     #         )
        #     #         for i, node in enumerate(nodes_to_add)
        #     #     ],
        #     # )
        # if actions:
        #     self.client.batch_update_points(
        #         collection_name=self.collection_name, update_operations=actions
        #     )
        # self.client.update_collection(
        #     collection_name=self.collection_name,
        #     hnsw_config=models.HnswConfigDiff(m=16),
        # )


def ingest_docs(
    docs: Sequence[Document],
    docstore: BaseDocstore,
    searchstores: Sequence[BaseSearchStore],
) -> None:
    pipe = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=100, chunk_overlap=50),
        ]
    )
    nodes = pipe.run(documents=docs)

    nodes = postprocess_ingestion_nodes(nodes=nodes)

    nodes_diff = NodeDiff([], [])
    print("about to put objects in docstore")
    try:
        nodes_diff = docstore.get_nodes_diff(nodes=nodes)
        print("Nodes diff", len(nodes_diff))
    except Exception as e:
        print(e)
    print("objects put in docstore")
    print(f"Nodes diff add: {len(nodes_diff.nodes_add)}")
    print(f"Nodes diff delete: {len(nodes_diff.node_ids_delete)}")

    for store in searchstores:
        print(type(store))
        store.add_delete_nodes(
            nodes_to_add=nodes_diff.nodes_add,
            node_ids_to_delete=nodes_diff.node_ids_delete,
        )


class RagDir(BaseModel):
    dir_name: str


class RagFiles(BaseModel):
    file_paths: str | list[str]


@app.post("/ingest_docs_from_dir/")
async def ingest_docs_from_dir(dir_name: RagDir):
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    try:
        fs = fsspec.filesystem("s3", endpoint_url="http://localhost:4566")
        converter = DocumentConverter()
        docs = get_docs_from_dir(dir_name=dir_name.dir_name, fs=fs, converter=converter)
        ingest_docs(docs, docstore=docstore, searchstores=searchstores)
        print("here i am")
        return {"status": "Success"}
    except Exception as e:
        return {"status": "failed", "error": e}


@app.post("/ingest_doc_from_file/")
async def ingest_doc_from_file(file_path: RagFiles):
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    fs = fsspec.filesystem("s3", endpoint_url="http://localhost:4566")
    converter = DocumentConverter()
    file_paths: list[str] = []
    if isinstance(file_path.file_paths, str):
        file_paths = [file_path.file_paths]
    else:
        file_paths = file_path.file_paths
    docs = [
        get_doc_from_fpath(file_path=f_path, fs=fs, converter=converter)
        for f_path in file_paths
    ]
    ingest_docs(docs, docstore=docstore, searchstores=searchstores)
    return {"status": "Success"}


@app.get("/sync/")
async def sync():
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    all_node_ids = docstore.get_all_node_ids()
    for store in searchstores:
        store.sync(ground_truth=all_node_ids, node_fetch_fn=docstore.get_nodes)
    return {"status": "Success"}


# docs = get_docs_from_dir(dir_name="my-bucket/pdfs/")
#
# pipe = IngestionPipeline(
#     transformations=[
#         MarkdownNodeParser(),
#         SentenceSplitter(chunk_size=100, chunk_overlap=50),
#     ]
# )
# nodes = pipe.run(documents=docs)
#
# nodes = postprocess_ingestion_nodes(nodes=nodes)
#
# nodes_diff = docstore.get_nodes_diff(nodes=nodes)
#
# for store in searchstores:
#     store.add_delete_nodes(
#         nodes_to_add=nodes_diff["add_nodes"], nodes_to_delete=nodes_diff["delete_nodes"]
#     )


class SearchQuery(BaseModel):
    query: str
    fusion_type: Literal["reciprocal_rank_fusion", "rerank_model"] = (
        "reciprocal_rank_fusion"
    )
    retriever_top_k: int = 10
    final_top_k: int = 5


@app.post("/query/")
async def query(search_query: SearchQuery):
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    postprocesors: list[BaseNodePostprocessor] = [
        PrevNextNodePostprocessor(
            id_node_fn=docstore.get_node, num_nodes=1, mode="both"
        ),
        MergeConsecutiveNodesPostprocessor(),
    ]
    query_results: list[VectorStoreQueryResult] = [
        store.search_query(
            query=search_query.query,
            node_fetch_fn=docstore.get_nodes,
            size=search_query.retriever_top_k,
        )
        for store in searchstores
    ]
    # print([len(res.nodes) for res in query_results])

    list_list_nodes_with_score: list[list[NodeWithScore]] = []
    for res in query_results:
        if not res.nodes:
            continue
        # assert res.nodes
        scores = []
        if res.similarities is None:
            scores = [None] * len(res.nodes)
        else:
            scores = res.similarities
        nodes = res.nodes
        list_list_nodes_with_score.append(
            [
                NodeWithScore(node=node, score=score)
                for node, score in zip(nodes, scores)
            ]
        )
    if not list_list_nodes_with_score:
        return {"status": "Failed", "Error": "No Query Results returned"}
    combined_nodes: list[NodeWithScore] = []
    if search_query.fusion_type == "reciprocal_rank_fusion":
        combined_nodes = reciprocal_rerank_fusion(results=list_list_nodes_with_score)
    elif search_query.fusion_type == "rerank_model":
        combined_nodes = rerank_nodes(
            query=search_query.query,
            nodes=dedup_nodes_with_score(list(chain(*list_list_nodes_with_score))),
        )
    postprocess_combined_nodes = reduce(
        lambda v, p_processor: p_processor.postprocess_nodes(v),
        postprocesors,
        combined_nodes,
    )
    return {
        "status": "Success",
        "rag_texts": [
            node.get_content(metadata_mode=MetadataMode.LLM)
            for node in postprocess_combined_nodes
        ],
    }


# fusion_nodes = reciprocal_rerank_fusion(list_list_nodes_with_score)
#
# reranked_nodes = rerank_nodes(
#     query=query, nodes=dedup_nodes_with_score(list(chain(*list_list_nodes_with_score)))
# )
#
# postprocesors: list[BaseNodePostprocessor] = [
#     PrevNextNodePostprocessor(id_node_fn=docstore.get_node, num_nodes=1, mode="both"),
#     MergeConsecutiveNodesPostprocessor(),
# ]
#
# postprocess_fusion_nodes = reduce(
#     lambda v, p_processor: p_processor.postprocess_nodes(v), postprocesors, fusion_nodes
# )
# postprocess_rerank_nodes = reduce(
#     lambda v, p_processor: p_processor.postprocess_nodes(v),
#     postprocesors,
#     reranked_nodes,
# )
#
#
# post_processed_nodes: list[list[NodeWithScore]] = []
#
# for node_list in list_list_nodes_with_score:
#     for postprocessor in postprocesors:
#         post_processed_nodes.append(postprocessor.postprocess_nodes(nodes=node_list))
#
#
# # print('"' + nodes[0].get_content() + '"')
# # print('"' + nodes[1].get_content() + '"')
# #
# node_dict = {
#     node.node_id: NodeWithScore(node=node, score=(i + 1) / len(nodes))
#     for i, node in enumerate(nodes)
# }
#
# post_processor = MergeConsecutiveNodesPostprocessor()
#
# processed_nodes = post_processor.postprocess_nodes(nodes=list(node_dict.values()))
# #
# # print('"' + processed_nodes[0].node.get_content()[:1500] + '"')
# print(len(processed_nodes))
# print('"' + node_dict[processed_nodes[0].node.node_id].get_content()[:1500] + '"')
# # print('"' + processed_nodes[1].node.get_content() + '"')
#
# ranked_nodes = rerank_nodes(
#     query="What is the main result?", nodes=list(node_dict.values())[:10]
# )
#
# print('"' + ranked_nodes[0].get_content() + '"')

# client = LlamaElasticsearch.get_client()
# LlamaElasticsearch.put_nodes_in_es(client, nodes=nodes)
#
# final_result = LlamaElasticsearch.search_es(
#     client=client, query={"match": {"content": "Invarian 4 dimensional Ricci Flow"}}
# )
#
# print(final_result)
