FROM python:3.12-slim-bookworm


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./bucket_manage.py /code/bucket_manage.py

EXPOSE 8000

CMD ["fastapi", "run", "bucket_manage.py", "--port", "8000"]
