# Use the official Python image from the Docker Hub as the base image.  
FROM python:3.14-slim

WORKDIR /app


# Installation of dependencies for python runtime sandbox.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir /.local
RUN mkdir /app/user

# Change ownership of the /app directory to the non-root user 1000.
RUN chown -R 1000:1000 /app
RUN chown -R 1000:1000 /.local
USER 1000

# Expose the port that the Uvicorn server will run on.
# This must match the port in the CMD instruction below.
EXPOSE 8888

# The command to run when the container starts.
# This starts the Uvicorn server, making our API available.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--log-level", "trace"]
# CMD ["sh", "-c", "ipython kernel & exec uvicorn main:app --host 0.0.0.0 --port 8888 --log-level trace"]
