# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/kubernetes-sigs/agent-sandbox/blob/main/examples/python-runtime-sandbox/main.py

# Note that files are stored in /app/user. Functions for downloading and uploading give the filename directly, but in code the file path needs to be /app/user/filename

import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import urllib.parse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BUCKET_MANAGE_BASE_URL = f"http://{os.environ['BUCKET_MANAGE_SERVICE_HOSTNAME']}:{os.environ['BUCKET_MANAGE_SERVICE_PORT']}"

JUPYTER_TEST_CODE = """
from jupyter_client import BlockingKernelClient, find_connection_file

conn_file = find_connection_file()

client = BlockingKernelClient(connection_file=conn_file)

client.load_connection_file()

client.execute_interactive({}, allow_stdin=False, timeout=15)
"""


class ExecuteRequest(BaseModel):
    """Request model for the /execute endpoint."""

    command: str


class ExecuteResponse(BaseModel):
    """Response model for the /execute endpoint."""

    stdout: str
    stderr: str
    exit_code: int


async def ensure_bucket() -> None:
    async with httpx.AsyncClient() as async_client:
        assert (await async_client.get(BUCKET_MANAGE_BASE_URL)).json()[
            "status"
        ] == "success"


async def write_file_from_sandbox_to_bucket(
    filepath: str, user_id: int, session_id: int
) -> None:
    async with httpx.AsyncClient() as async_client:
        post_url = (
            await async_client.get(
                BUCKET_MANAGE_BASE_URL + "/upload_file_link_user_session",
                params={
                    "file_name": filepath.split(os.path.sep)[-1],
                    "userid": user_id,
                    "sessionid": session_id,
                },
            )
        ).json()["url"]
        with open(filepath, "rb") as f:
            await async_client.post(
                post_url["url"], data=post_url["fields"], files={"file": f}
            )


async def get_download_file_url(filepath: str, user_id: int, session_id: int) -> str:
    await ensure_bucket()
    await write_file_from_sandbox_to_bucket(
        filepath=filepath, user_id=user_id, session_id=session_id
    )
    async with httpx.AsyncClient() as async_client:
        return (
            await async_client.get(
                BUCKET_MANAGE_BASE_URL + "/download_file_link_user_session",
                params={
                    "file_name": filepath.split(os.path.sep)[-1],
                    "userid": user_id,
                    "sessionid": session_id,
                },
            )
        ).json()["url"]


def get_base_dir() -> str:
    """Reads SANDBOX_BASE_DIR, falling back to /app/user when it's unset or blank.

    Making the base directory configurable lets the sandbox run with a
    read-only root filesystem: the runtime code stays wherever the image put
    it, while commands and file operations are confined to a writable volume
    (e.g. an emptyDir) mounted at SANDBOX_BASE_DIR.
    """
    return os.environ.get("SANDBOX_BASE_DIR", "").strip() or "/app/user"


def get_safe_path(file_path: str) -> str:
    """Sanitizes the file path to ensure it stays within the base directory."""
    base_dir = os.path.realpath(get_base_dir())
    # Remove leading slashes to ensure path is relative
    clean_path = file_path.lstrip("/")
    full_path = os.path.realpath(os.path.join(base_dir, clean_path))

    if os.path.commonpath([base_dir, full_path]) != base_dir:
        raise ValueError(
            "Access denied: Path must be within the sandbox base directory"
        )

    return full_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = subprocess.Popen(["ipython", "kernel"])
    yield
    p.kill()


app = FastAPI(
    title="Agentic Sandbox Runtime",
    description="An API server for executing commands and managing files in a secure sandbox.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_exec_timeout_seconds() -> float:
    """Reads SANDBOX_EXEC_TIMEOUT_SECONDS, falling back to the 300s default
    (with a warning) if it's unset or not a finite number greater than 0, so
    a misconfigured value doesn't fail every /execute request.
    """
    raw = os.environ.get("SANDBOX_EXEC_TIMEOUT_SECONDS", "300")
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")

    if not (math.isfinite(value) and value > 0):
        logging.warning(
            "Ignoring invalid SANDBOX_EXEC_TIMEOUT_SECONDS=%r; using default of 300 seconds",
            raw,
        )
        return 300.0
    return value


def _run_command(args: list, timeout: float) -> subprocess.CompletedProcess:
    """Runs args as the leader of a new process group so that on timeout the
    entire process tree can be killed, not just the direct child (matching
    the pattern used in examples/firecracker-sandbox/main.py).
    """
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=get_base_dir(),
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


@app.get("/", summary="Health Check")
async def health_check():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "ok", "message": "Sandbox Runtime is active."}


@app.post("/execute", summary="Execute a shell command", response_model=ExecuteResponse)
async def execute_command(request: ExecuteRequest):
    """
    Executes a shell command inside the sandbox and returns its output.
    Uses shlex.split for security to prevent shell injection.
    """
    # command = f'python -c {json.dumps(JUPYTER_TEST_CODE.format(request.command))}'
    try:
        # return ExecuteResponse(stdout="some stdout", stderr="some stderr", exit_code=0)
        # Split the command string into a list to safely pass to subprocess
        # args = shlex.split(command)
        args = ["python", "-c", JUPYTER_TEST_CODE.format(json.dumps(request.command))]

        # Execute the command, always from the base directory. Run it in a
        # worker thread so a long-running or hung command doesn't block the
        # event loop (and with it, the health check and file endpoints), and
        # enforce a timeout so a runaway command can't wedge the sandbox
        # forever.
        process = await asyncio.to_thread(
            _run_command,
            args,
            _get_exec_timeout_seconds(),
        )
        return ExecuteResponse(
            stdout=process.stdout, stderr=process.stderr, exit_code=process.returncode
        )
    except Exception as e:
        return ExecuteResponse(
            stdout="", stderr=f"Failed to execute command: {e!s}", exit_code=1
        )


@app.post("/upload", summary="Upload a file to the sandbox")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file and saves it to the base directory in the sandbox.
    """
    try:
        assert file.filename
        # decode_string = file.decode()
        # pos = decode_string.find("::http")
        # filename, file_url = decode_string[:pos], decode_string[pos + 2 :]
        logging.info(
            f"--- UPLOAD_FILE CALLED: Attempting to save '{file.filename}' ---"
        )
        # print(f"file to save: {file.filename}")

        try:
            file_path = get_safe_path(file.filename)
        except ValueError:
            return JSONResponse(status_code=403, content={"message": "Access denied"})

        # The filename may carry a relative destination path (e.g.
        # "data/input.csv"); create the intermediate directories so such
        # uploads don't fail. file_path is already confined to the base
        # directory by get_safe_path, so its parents are too.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        file_url = (await file.read()).decode()
        # logging.info(f"The file url is: {file_url}")
        # print(f"The file url is: {file_url}")
        # await ensure_bucket()
        # with httpx.Client() as sync_client:
        #     res = sync_client.get(file_url)
        #     print(res.status_code)
        #     res_content = res.content
        # print("got content")
        # with open(file_path, "wb") as f:
        #     f.write(res_content)

        async with (
            httpx.AsyncClient() as async_client,
            async_client.stream("GET", file_url) as r,
        ):
            with open(file_path, "wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"File '{file.filename}' uploaded successfully."},
        )
    except Exception as e:
        logging.exception("An error occurred during file upload.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"File upload failed: {e!s}"},
        )


@app.get(
    "/download/{encoded_file_path:path}", summary="Download a file from the sandbox"
)
async def download_file(encoded_file_path: str):
    """
    Downloads a specified file from the base directory in the sandbox.
    """
    decoded_path = urllib.parse.unquote(encoded_file_path)
    filename, user_id_str, session_id_str = decoded_path.split("::")
    user_id, session_id = int(user_id_str), int(session_id_str)
    try:
        full_path = get_safe_path(filename)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN, content={"message": "Access denied"}
        )

    if os.path.isfile(full_path):
        file_url = await get_download_file_url(
            filepath=full_path, user_id=user_id, session_id=session_id
        )
        return file_url.encode("utf-8")
        # return JSONResponse(
        #     status_code=status.HTTP_200_OK,
        #     content=file_url.encode("utf-8"),
        # )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"message": "File not found"}
    )


@app.get("/list/{encoded_file_path:path}", summary="List files in a directory")
async def list_files(encoded_file_path: str):
    """
    Lists the contents of a directory under the base directory in the sandbox.
    """
    decoded_path = urllib.parse.unquote(encoded_file_path)
    try:
        full_path = get_safe_path(decoded_path)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN, content={"message": "Access denied"}
        )

    if not os.path.isdir(full_path):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Path is not a directory"},
        )

    try:
        entries = []
        with os.scandir(full_path) as it:
            for entry in it:
                stats = entry.stat()
                entries.append(
                    {
                        "name": entry.name,
                        "size": stats.st_size,
                        "type": "directory" if entry.is_dir() else "file",
                        "mod_time": stats.st_mtime,
                    }
                )
        return JSONResponse(status_code=status.HTTP_200_OK, content=entries)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"List files failed: {e!s}"},
        )


@app.get(
    "/exists/{encoded_file_path:path}", summary="Check if the relative path exists"
)
async def exists(encoded_file_path: str):
    """
    Checks if a specified file or directory exists under the base directory in the sandbox.
    """
    decoded_path = urllib.parse.unquote(encoded_file_path)
    try:
        full_path = get_safe_path(decoded_path)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN, content={"message": "Access denied"}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"path": decoded_path, "exists": os.path.exists(full_path)},
    )
