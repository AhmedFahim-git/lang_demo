import requests

# BASE_URL = "http://127.0.0.1:8000/"
# CODE_URL = "http://127.0.0.1:8001/"
BASE_URL = "http://my-bucket/"
CODE_URL = "http://code-runner/"


def upload_files(files: list[str], userid: str, sessionid: str):
    post_urls = dict()
    # print(requests.get(BASE_URL).json()["exception"])
    assert requests.get(BASE_URL).json()["status"] == "success"
    for file in files:
        post_urls[file] = requests.get(
            BASE_URL + "upload_file_link/",
            params={"file": file, "userid": userid, "sessionid": sessionid},
        ).json()["url"]
    for k, v in post_urls.items():
        with open(k, "rb") as f:
            # print(v)
            requests.post(
                v["url"].replace(":4566", ""), data=v["fields"], files={"file": (k, f)}
            )


def download_files(files: list[str], userid: str, sessionid: str):
    get_urls = dict()
    assert requests.get(BASE_URL).json()["status"] == "success"
    for file in files:
        get_urls[file] = requests.get(
            BASE_URL + "download_file_link/",
            params={"file": file, "userid": userid, "sessionid": sessionid},
        ).json()["url"]
    print(get_urls)
    for k, v in get_urls.items():
        r = requests.get(v.replace(":4566", ""), stream=True)
        # r.raise_for_status()
        with open(k, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    userid = "user2"
    sessionid = "session6"
    upload_files(files=["pyproject.toml"], userid=userid, sessionid=sessionid)
    code = "from time import time\nwith open('pyproject.toml', 'r') as f:\n  line_count=len(f.readlines())\nwith open('exp.txt', 'w') as f:\n  f.write(str(time())+'__'+str(line_count))\nprint(line_count)\nsecret_num=444\n"
    r = requests.post(
        CODE_URL, json={"code": code, "userid": userid, "sessionid": sessionid}
    )  # .json()
    # print(r.json()["exception"])
    r = r.json()
    download_files(r["filenames"], userid=userid, sessionid=sessionid)
    print(r)
    r = requests.post(
        CODE_URL,
        json={
            "code": "print(secret_num)\nprint(time())\n",
            "userid": userid,
            "sessionid": sessionid,
        },
    ).json()
    download_files(r["filenames"], userid=userid, sessionid=sessionid)
    print(r)
