FROM python:3.12-slim-bookworm

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir docling --extra-index-url https://download.pytorch.org/whl/cpu

RUN docling-tools models download

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./*.py /code/

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]
