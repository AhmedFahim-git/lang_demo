FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends firejail \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./run_code_func.py /code/run_code_func.py

EXPOSE 8000

CMD ["fastapi", "run", "run_code_func.py", "--port", "8000"]
