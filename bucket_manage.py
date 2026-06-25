import os
import shutil
import tempfile
import zipfile
from typing import Annotated

import boto3
import botocore
from fastapi import BackgroundTasks, FastAPI, Form, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()


# TODO: bucket name need to be extracted into config
def ensure_bucket(s3_client, bucket_name="my-bucket"):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:  # Bucket does not exist
            s3_client.create_bucket(Bucket=bucket_name)
            return True
        else:
            return False
    return True


@app.get("/")
async def check_bucket():
    s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
    created_bool = ensure_bucket(s3_client=s3)
    assert created_bool
    return {"status": "success"}


@app.get("/upload_file_link/")
async def get_post_link(file: str, userid: str, sessionid: str):
    s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
    url = s3.generate_presigned_post(
        "my-bucket",
        f"users/{userid}/{sessionid}/file_dir/{file}",
        Conditions=[
            ["content-length-range", 0, 5 * 1024 * 1024]  # 0–5 MB
        ],
    )
    print(url)
    return {"status": "success", "url": url}


@app.get("/download_file_link/")
async def get_presigned_url(file: str, userid: str, sessionid: str):
    s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": "my-bucket",
            "Key": f"users/{userid}/{sessionid}/file_dir/{file}",
        },
    )
    return {"status": "success", "url": url}


@app.post("/uploadfile/")
async def upload_file(
    files: list[UploadFile],
    userid: Annotated[str, Form()],
    sessionid: Annotated[str, Form()],
):
    files_up = []
    try:
        s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
        created_bool = ensure_bucket(s3_client=s3)
        assert created_bool
        for file in files:
            s3.upload_fileobj(
                file.file,
                "my-bucket",
                f"users/{userid}/{sessionid}/file_dir/{str(file.filename)}",
            )
            files_up.append(str(file.filename))
        return {"status": "success", "files_uploaded": files_up}
    except Exception as e:
        return {"status": "failed", "files_uploaded": files_up, "error": str(e)}


@app.get("/downloadfile/")
async def download_file(
    background_tasks: BackgroundTasks, files: list[str], userid: str, sessionid: str
):
    with tempfile.TemporaryDirectory(delete=False) as my_temp_dir:
        s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
        for file in files:
            s3.download_file(
                "my-bucket",
                f"users/{userid}/{sessionid}/file_dir/{file}",
                os.path.join(my_temp_dir, "file_dir", file),
            )
        with zipfile.ZipFile(os.path.join(my_temp_dir, "my_zip.zip"), "w") as archive:
            for f in [os.path.join(my_temp_dir, "file_dir", file) for file in files]:
                archive.write(f)
        background_tasks.add_task(lambda: shutil.rmtree(my_temp_dir))
    return FileResponse(os.path.join(my_temp_dir, "my_zip.zip"))
