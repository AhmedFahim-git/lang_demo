from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from typing import Sequence

import filetype
import fsspec
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docstore import BaseDocstore
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
)
from searchstores import BaseSearchStore


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

    nodes_diff = docstore.get_nodes_diff(nodes=nodes)

    for store in searchstores:
        store.add_delete_nodes(
            nodes_to_add=nodes_diff.nodes_add,
            node_ids_to_delete=nodes_diff.node_ids_delete,
        )
