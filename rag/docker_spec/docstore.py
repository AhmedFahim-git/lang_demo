from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence

import pymongo
from config import MONGO_URI
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pymongo import MongoClient


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
        node_operations = []
        if nodes_to_delete:
            node_operations.append(
                pymongo.DeleteMany({"_id": {"$in": nodes_to_delete}})
            )
        if nodes_to_add:
            node_operations.extend([pymongo.InsertOne(i) for i in nodes_to_add])
        if doc_operations:
            self.doc_collection.bulk_write(doc_operations)
        if node_operations:
            self.node_collection.bulk_write(node_operations)
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
