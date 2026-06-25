import os

from llama_index.embeddings.openai_like import OpenAILikeEmbedding

EMBED_MODEL = OpenAILikeEmbedding(
    model_name="Qwen0.6B",
    api_base=f"http://{os.environ['LLAMA_CPP_HOSTNAME']}:8080/v1/",
)
EMBED_DIM = 1024

MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@{os.environ['MONGO_SERVICE_HOSTNAME']}:27017/?authSource=admin"  # "mongodb://root:example@localhost:27017/?authSource=admin"
# MONGO_URI = "mongodb://root:example@localhost:27017/?authSource=admin"
ES_URL = os.environ["ES_URL"]
ES_INDEX_NAME = "es_index"
RERANK_BASE_URL = f"http://{os.environ['LLAMA_CPP_HOSTNAME']}:8081/v1"
QDRANT_URL = f"http://{os.environ['QDRANT_HOSTNAME']}:6333"
QDRANT_COLLECTION_NAME = "collection_name"
FLOCI_URL = os.environ["AWS_ENDPOINT_URL"]
