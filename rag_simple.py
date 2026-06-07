import uuid
from glob import glob

from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings

# from elasticsearch import Elasticsearch, helpers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client import QdrantClient, models

embedding_model = OpenAIEmbeddings(
    model="Qwen0.6B", base_url="http://localhost:8080/v1/", api_key=SecretStr("None")
)

client = QdrantClient(url="http://localhost:6333")

converter = DocumentConverter()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len
)

if client.collection_exists(collection_name="collection"):
    client.delete_collection(collection_name="collection")
client.create_collection(
    collection_name="collection",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
# client = Elasticsearch(
#     "https://localhost:9200",
#     ca_certs="http_ca.crt",
#     basic_auth=("elastic", "AthtidW2mrdT672-eP-7"),
# )
#
# mappings = {
#     "properties": {
#         "page_content": {"type": "text"},
#         "filename": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
#         "chunk_index": {"type": "integer"},
#     }
# }
#
# client.indices.create(index="my_index", mappings=mappings)
# res = []


def get_docs():
    for file in glob("pdfs/*"):
        parsed_doc = converter.convert(file)
        mkdn = parsed_doc.document.export_to_markdown()
        texts = splitter.split_text(mkdn)
        print(file)
        client.upload_points(
            collection_name="collection",
            points=[
                models.PointStruct(
                    id=uuid.uuid4(),
                    vector=embedding_model.embed_query(chunk),
                    payload={
                        "page_content": chunk,
                        "filename": file.split("/")[-1],
                        "chunk_index": i,
                    },
                )
                for i, chunk in enumerate(texts)
            ],
        )
        break
        # for i, chunk in enumerate(texts):
        # yield {
        #     "_index": "my_index",
        #     "page_content": chunk,
        #     "filename": file.split("/")[-1],
        #     "chunk_index": i,
        # }
        # res.append(
        #     {
        #         "page_content": chunk,
        #         "filename": file.split("/")[-1],
        #         "chunk_index": i,
        #     }
        # )

        # break


get_docs()

hits = client.query_points(
    collection_name="collection",
    query=embedding_model.embed_query("What is the result?"),
    limit=3,
).points

for hit in hits:
    print(hit.payload, "score:", hit.score)

# helpers.bulk(client, actions=get_docs())

# helpers.bulk(client=client, actions=res)


# results = client.search(
#     index="my_index",
#     query={"match": {"page_content": "Italian income municipality"}},
#     size=5,
# )
#
# for res in results["hits"]["hits"]:
#     print(res["_score"], res["_source"])
