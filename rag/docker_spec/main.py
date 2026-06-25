from functools import reduce
from itertools import chain
from typing import Literal

import fsspec
from config import FLOCI_URL
from docling.document_converter import DocumentConverter
from docstore import MongoDocstore
from fastapi import FastAPI
from ingestion import get_doc_from_fpath, get_docs_from_dir, ingest_docs
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
)
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from postprocessor import (
    MergeConsecutiveNodesPostprocessor,
    PrevNextNodePostprocessor,
    dedup_nodes_with_score,
)
from pydantic import BaseModel
from rerank import reciprocal_rerank_fusion, rerank_nodes
from searchstores import BaseSearchStore, ElasticsearchStore, QdrantStore

app = FastAPI()


class RagDir(BaseModel):
    dir_name: str


class RagFiles(BaseModel):
    file_paths: str | list[str]


@app.post("/ingest_docs_from_dir/")
async def ingest_docs_from_dir(dir_name: RagDir):
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    fs = fsspec.filesystem("s3", endpoint_url=FLOCI_URL)
    converter = DocumentConverter()
    docs = get_docs_from_dir(dir_name=dir_name.dir_name, fs=fs, converter=converter)
    ingest_docs(docs, docstore=docstore, searchstores=searchstores)
    return {"status": "Success"}


@app.post("/ingest_doc_from_file/")
async def ingest_doc_from_file(file_path: RagFiles):
    docstore = MongoDocstore()
    searchstores: list[BaseSearchStore] = [ElasticsearchStore(), QdrantStore()]
    fs = fsspec.filesystem("s3", endpoint_url=FLOCI_URL)
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
