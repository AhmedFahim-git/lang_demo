from typing import Sequence

import requests
from config import RERANK_BASE_URL
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
)


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
