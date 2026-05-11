# Trying out Langchain, WebSearch, Code Execution in Secure Kata Containers, RAG, MCP

# WIP


# Starting llama cpp server

`llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q8_0 --jinja -ngl 99 -fa auto -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 40960 -n 32768 --no-context-shift`

# FastAPI dev server

`fastapi dev dir_func.py`

# Curl test command

`curl -F "files=@pyproject.toml" -F "files=@uv.lock" -F "code='print(345)'" -F "pkl_file=@graph_websearch.py" http://127.0.0.1:8000/uploadfile/`


# Gateway API

Install helm charts, minkube tunnel, make gateway api and httproute manifests, update /etc/hosts
