import os
import shutil
import tempfile
import traceback
import zipfile
from typing import Annotated

import boto3
import botocore
from fastapi import BackgroundTasks, FastAPI, Form, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

# TODO: In terms of authentication, it might be good idea to not expose the base get upload or download url. Instead make sub methods that extend that function


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


@app.get("/", response_class=JSONResponse)
async def check_bucket():
    try:
        s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
        created_bool = ensure_bucket(s3_client=s3)
        assert created_bool
        return JSONResponse(
            content={"status": "success"}, status_code=status.HTTP_201_CREATED
        )
        # return {"status": "success"}
    except Exception:
        return JSONResponse(
            content={"status": "failed", "exception": traceback.format_exc()},
            status_code=status.HTTP_409_CONFLICT,
        )


# TODO: Convention is for dir_name to start with / not end with /
@app.get("/upload_file_link", response_class=JSONResponse)
async def get_post_link(path_in_bucket: str = ""):
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    # if path_in_bucket:
    #     path_in_bucket = f"{path_in_bucket.strip('/')}/{file_name}"
    # else:
    #     path_in_bucket = file_name
    # TODO: Check if file exists
    url = s3.generate_presigned_post(
        "my-bucket",
        path_in_bucket,
        Conditions=[
            ["content-length-range", 0, 50 * 1024 * 1024]  # 0–50 MB
        ],
    )
    # print(url)
    return JSONResponse(content={"status": "success", "url": url})


# TODO: This would require some auth
@app.get("/upload_file_link_user_session", response_class=JSONResponse)
async def get_post_link_user_session(file_name: str, userid: int, sessionid: int):
    return await get_post_link(
        path_in_bucket=f"users/{userid}/{sessionid}/file_dir/{file_name}"
    )
    # s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    # url = s3.generate_presigned_post(
    #     "my-bucket",
    #     f"users/{userid}/{sessionid}/file_dir/{file}",
    #     Conditions=[
    #         ["content-length-range", 0, 5 * 1024 * 1024]  # 0–5 MB
    #     ],
    # )
    # print(url)
    # return {"status": "success", "url": url}


# TODO: This would require some auth
@app.get("/download_file_link", response_class=JSONResponse)
async def get_presigned_url(path_in_bucket: str = ""):
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    # if path_in_bucket:
    #     path_in_bucket = f"{path_in_bucket.strip('/')}/{file_name}"
    # else:
    #     path_in_bucket = file_name
    # TODO: Check if file exists
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": "my-bucket", "Key": path_in_bucket}
    )
    return JSONResponse({"status": "success", "url": url})


@app.get("/download_file_link_user_session", response_class=JSONResponse)
async def get_presigned_url_user_session(file_name: str, userid: int, sessionid: int):
    return await get_presigned_url(
        path_in_bucket=f"users/{userid}/{sessionid}/file_dir/{file_name}"
    )
    # s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    # url = s3.generate_presigned_url(
    #     "get_object",
    #     Params={
    #         "Bucket": "my-bucket",
    #         "Key": f"users/{userid}/{sessionid}/file_dir/{file}",
    #     },
    # )
    # return {"status": "success", "url": url}


@app.get("/exists", response_class=JSONResponse)
async def check_exists(path_in_bucket: str = ""):
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    status_code, exists = status.HTTP_500_INTERNAL_SERVER_ERROR, False
    try:
        s3.head_object(Bucket="my-bucket", Key=path_in_bucket)
        status_code = status.HTTP_200_OK
        exists = True
    except botocore.exceptions.ClientError as e:
        status_code = int(e.response["Error"]["Code"])
    return JSONResponse(
        content={"path": path_in_bucket, "exists": exists}, status_code=status_code
    )


@app.get("/list_objects", response_class=JSONResponse)
async def list_objects(path_in_bucket: str = ""):
    s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    response = s3.list_objects_v2(
        Bucket="my-bucket", Prefix=f"{path_in_bucket.strip('/')}/"
    )
    items = []
    for obj in response.get("Contents", []):
        items.append(obj["Key"])
    return JSONResponse(content=items, status_code=status.HTTP_200_OK)


@app.post("/uploadfile")
async def upload_file(
    files: list[UploadFile],
    userid: Annotated[str, Form()],
    sessionid: Annotated[str, Form()],
):
    files_up = []
    try:
        s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
        created_bool = ensure_bucket(s3_client=s3)
        assert created_bool
        for file in files:
            s3.upload_fileobj(
                file.file,
                "my-bucket",
                f"users/{userid}/{sessionid}/file_dir/{file.filename!s}",
            )
            files_up.append(str(file.filename))
        return {"status": "success", "files_uploaded": files_up}
    except Exception as e:
        return {"status": "failed", "files_uploaded": files_up, "error": str(e)}


@app.get("/downloadfile")
async def download_file(
    background_tasks: BackgroundTasks, files: list[str], userid: str, sessionid: str
):
    with tempfile.TemporaryDirectory(delete=False) as my_temp_dir:
        s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
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
