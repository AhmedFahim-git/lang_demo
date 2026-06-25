import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from glob import glob
from typing import Annotated, Optional

from fastapi import FastAPI, Form, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse


def get_mod_times_dict():
    time_dict = dict()
    for filename in glob("**/*", recursive=True, include_hidden=True):
        time_dict[filename] = os.path.getmtime(filename)
    return time_dict


app = FastAPI()


@app.post("/uploadfile/")
async def writefile(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    code: Annotated[str, Form()],
    pkl_file: Optional[UploadFile] = None,
):
    with tempfile.TemporaryDirectory(delete=False) as my_temp_dir:
        cur_dir = os.getcwd()
        os.mkdir(os.path.join(my_temp_dir, "file_dir"))
        os.mkdir(os.path.join(my_temp_dir, "pkl_dir"))
        os.chdir(my_temp_dir)
        # print(eval(code))
        for file in files:
            with open(os.path.join("file_dir", str(file.filename)), "wb") as f:
                shutil.copyfileobj(file.file, f)
        if pkl_file:
            with open(os.path.join("pkl_dir", str(pkl_file.filename)), "wb") as f:
                shutil.copyfileobj(pkl_file.file, f)
        init_time_dict = get_mod_times_dict()
        os.chdir("file_dir")
        # print(code)
        if pkl_file:
            code = "import dill\n" + 'dill.load_module("../my_session.pkl")\n' + code
        else:
            code = "import dill\n" + code
        code = code + '\ndill.dump_module("../my_session.pkl")'
        # print(code)
        result = subprocess.run(
            ["unshare", "--net", "python", "-c", code], capture_output=True, text=True
        )
        with open("My_File.txt", "w") as f:
            f.write("Yo its me")

        os.chdir("..")
        final_time_dict = get_mod_times_dict()
        modified_files = [
            k
            for k, v in final_time_dict.items()
            if (k not in init_time_dict) or (v != init_time_dict[k])
        ]
        json_send = (
            {
                "filename": "bro",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": str(result.returncode),
            },
        )
        with zipfile.ZipFile("my_zip.zip", "w") as archive:
            for f in modified_files:
                archive.write(f)
            archive.comment = json.dumps(json_send).encode()
        # print(os.getcwd())
        os.chdir(cur_dir)
        background_tasks.add_task(lambda: shutil.rmtree(my_temp_dir))
        # print(os.listdir(my_temp_dir))
        print(os.path.join(my_temp_dir, "my_zip.zip"))
        print(os.path.isfile(os.path.join(my_temp_dir, "my_zip.zip")))
    return FileResponse(
        os.path.join(my_temp_dir, "my_zip.zip"),
        # headers={
        #     "filename": "bro",
        #     "stdout": result.stdout,
        #     "stderr": result.stderr,
        #     "returncode": str(result.returncode),
        # },
    )

    # return {
    #     "filename": "bro",
    #     "stdout": result.stdout,
    #     "stderr": result.stderr,
    #     "returncode": result.returncode,
    # }
