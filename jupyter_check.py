import asyncio
import json
import logging
import os

import httpx
from k8s_agent_sandbox import SandboxClient
from k8s_agent_sandbox.models import SandboxLocalTunnelConnectionConfig

logging.basicConfig(level=logging.INFO)
client = SandboxClient(connection_config=SandboxLocalTunnelConnectionConfig())

BUCKET_MANAGE_BASE_URL = "http://my-bucket"


async def ensure_bucket() -> None:
    async with httpx.AsyncClient() as async_client:
        assert (await async_client.get(BUCKET_MANAGE_BASE_URL)).json()[
            "status"
        ] == "success"


async def write_file_to_bucket(
    filepath: str, user_id: int, session_id: int, bucket_filename: str | None = None
) -> None:
    if bucket_filename is None:
        bucket_filename = filepath.split(os.path.sep)[-1]
    async with httpx.AsyncClient() as async_client:
        post_url = (
            await async_client.get(
                BUCKET_MANAGE_BASE_URL + "/upload_file_link_user_session",
                params={
                    "file_name": bucket_filename,
                    "userid": user_id,
                    "sessionid": session_id,
                },
            )
        ).json()["url"]
        print(post_url)
        with open(filepath, "rb") as f:
            await async_client.post(
                post_url["url"].replace(":4566", ""),
                data=post_url["fields"],
                files={"file": f},
            )


async def get_download_file_url(
    filepath: str, user_id: int, session_id: int, bucket_filename: str | None = None
) -> str:
    if bucket_filename is None:
        bucket_filename = filepath.split(os.path.sep)[-1]
    await ensure_bucket()
    await write_file_to_bucket(
        filepath=filepath,
        user_id=user_id,
        session_id=session_id,
        bucket_filename=bucket_filename,
    )
    async with httpx.AsyncClient() as async_client:
        return (
            await async_client.get(
                BUCKET_MANAGE_BASE_URL + "/download_file_link_user_session",
                params={
                    "file_name": bucket_filename,
                    "userid": user_id,
                    "sessionid": session_id,
                },
            )
        ).json()["url"]


async def download_file(file_url: str, file_name: str):
    # print(file_url)
    async with (
        httpx.AsyncClient() as async_client,
        async_client.stream("GET", file_url) as r,
    ):
        with open(file_name, "wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)


# print("Basic check")
# sandbox = client.create_sandbox(warmpool="my-sandboxwarmpool")
# result = sandbox.commands.run('print("hello world")')
# print(result.stdout)
# print(result.stderr)
#
# print(sandbox.claim_name)
# sandbox.terminate()

# print("Check persistence")
# sandbox = client.create_sandbox(
#     warmpool="my-sandboxwarmpool", shutdown_after_seconds=360
# )
# claim_name = sandbox.claim_name
# result = sandbox.commands.run("x=56\nprint(78)")
# print(result)
# print(claim_name)
#
# # claim_name = "sandbox-claim-68a1cfc9"
#
# re_sandbox = client.get_sandbox(claim_name=claim_name)
# res_again = re_sandbox.commands.run("print(x+45)")
# print(res_again)
# re_sandbox.terminate()

print("Check sandbox write, read, exist, list programmatically and use file in code")


async def main():
    # async with AsyncSandboxClient(
    #     connection_config=SandboxLocalTunnelConnectionConfig()
    # ) as async_sandbox_client:
    file_url = await get_download_file_url(
        filepath="database.py",
        user_id=1,
        session_id=1,
        bucket_filename="database_2.py",
    )
    print(file_url)
    sandbox = client.create_sandbox(
        warmpool="my-sandboxwarmpool", shutdown_after_seconds=90
    )
    print(sandbox.files.write(path="database_2.py", content=file_url))
    print(sandbox.files.exists(path="database_2.py"))
    print(sandbox.files.list(path="./"))
    print(
        sandbox.commands.run(
            "with open('user/database_2.py') as f:\n  all_lines=f.read()\nprint(all_lines[:50])"
        )
    )
    res: str = json.loads(
        sandbox.files.read(path="database_2.py::1::1").decode()
    ).strip()
    # print(res)
    # print(type(res))
    cleaned_res = res.replace(":4566", "")
    # print(cleaned_res)
    # print(urlparse(cleaned_res))
    await download_file(file_url=cleaned_res, file_name="database_2.py")

    sandbox.terminate()
    # print(await sandbox.files.read())


if __name__ == "__main__":
    asyncio.run(main())
