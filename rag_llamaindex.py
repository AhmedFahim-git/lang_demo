from collections import deque
from collections.abc import Callable, Generator
from copy import deepcopy
from glob import glob
from typing import Any, Literal, Optional, Sequence

import requests
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from llama_index.core import (
    SimpleDirectoryReader,
)
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.readers.docling import DoclingReader

ES_INDEX_NAME = "es_index"
RERANK_BASE_URL = "http://localhost:8080/v1"


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
    nodes: list[NodeWithScore],
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
        prev_node_start_end = prev_node.get_node_info()
        next_node = next_node_with_score.node
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
        all_nodes: dict[str, NodeWithScore] = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node
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
                            )
                            node_deque.appendleft(prev_node_id)
                            node_with_score = (
                                MergeConsecutiveNodesPostprocessor._join_nodes(
                                    prev_node_with_score=prev_node_with_score,
                                    next_node_with_score=node_with_score,
                                )
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
                            node_with_score = (
                                MergeConsecutiveNodesPostprocessor._join_nodes(
                                    prev_node_with_score=node_with_score,
                                    next_node_with_score=next_node_with_score,
                                )
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
                    PrevNextNodePostprocessor.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
            elif self.mode == "previous":
                all_nodes.update(
                    PrevNextNodePostprocessor.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="prev",
                    )
                )
            elif self.mode == "both":
                all_nodes.update(
                    PrevNextNodePostprocessor.get_for_back_nodes(
                        node,
                        self.num_nodes,
                        id_node_fn=self.id_node_fn,
                        direction="next",
                    )
                )
                all_nodes.update(
                    PrevNextNodePostprocessor.get_for_back_nodes(
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


pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)

files = []
for file in glob("pdfs/*"):
    files.append(file)

dir_reader = SimpleDirectoryReader(
    input_files=files[:2], file_extractor={".pdf": pdf_reader}
)

docs = dir_reader.load_data()
pipe = IngestionPipeline(
    transformations=[
        MarkdownNodeParser(),
        SentenceSplitter(chunk_size=100, chunk_overlap=50),
    ]
)
nodes = pipe.run(documents=docs)

relation_dict = dict()
for node in nodes:
    relation_dict.update(node.relationships)


print(relation_dict)


# print('"' + nodes[0].get_content() + '"')
# print('"' + nodes[1].get_content() + '"')
#
node_dict = {
    node.node_id: NodeWithScore(node=node, score=(i + 1) / len(nodes))
    for i, node in enumerate(nodes)
}

post_processor = MergeConsecutiveNodesPostprocessor()

processed_nodes = post_processor.postprocess_nodes(nodes=list(node_dict.values()))
#
# print('"' + processed_nodes[0].node.get_content()[:1500] + '"')
print(len(processed_nodes))
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
