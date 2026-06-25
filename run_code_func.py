import os
import subprocess
import tempfile
from glob import glob

import fsspec
from fastapi import FastAPI
from pydantic import BaseModel


def get_mod_times_dict():
    time_dict = dict()
    for filename in glob("**/*", recursive=True, include_hidden=True):
        if os.path.isfile(filename):
            time_dict[filename] = os.path.getmtime(filename)
    return time_dict


app = FastAPI()


class CodeInfo(BaseModel):
    code: str
    userid: str
    sessionid: str


@app.post("/")
async def run_code(code_info: CodeInfo):
    fs = fsspec.filesystem("s3", endpoint_url="http://localhost:4566")
    with tempfile.TemporaryDirectory() as my_temp_dir:
        cur_dir = os.getcwd()
        remote_dir = f"s3://my-bucket/users/{code_info.userid}/{code_info.sessionid}/"  # TODO: refactor out bucket name
        # print("exists", fs.exists(remote_dir + "*/*"))
        # remote_dir = "s3://my-bucket/users/uservasd/session343/"  # TODO: refactor out bucket name
        # if fs.isdir(remote_dir):
        #     fs.get(remote_dir, my_temp_dir, recursive=True)
        # print("fs remote_dir", fs.isdir(remote_dir))
        try:
            fs.get(remote_dir, my_temp_dir, recursive=True)
        except FileNotFoundError:
            print("Shit file not found")
        os.makedirs(os.path.join(my_temp_dir, "file_dir"), exist_ok=True)
        os.makedirs(os.path.join(my_temp_dir, "pkl_dir"), exist_ok=True)
        os.chdir(my_temp_dir)
        os.chdir("file_dir")
        init_time_dict = get_mod_times_dict()
        if os.path.exists("../pkl_dir/my_session.pkl"):
            code = (
                "import dill\n"
                + 'dill.load_module("../pkl_dir/my_session.pkl")\n'
                + code_info.code
            )
        else:
            cleanup_func = """def __file_handle_cleanup():
    import io
    to_delete = []
    for name, obj in globals().copy().items():
        if isinstance(obj, io.IOBase) and obj.closed:
            to_delete.append(name)

    for name in to_delete:
        del globals()[name]
"""
            code = "import dill\n" + cleanup_func + code_info.code
        code = (
            code
            + "\n__file_handle_cleanup()"
            + '\ndill.dump_module("../pkl_dir/my_session.pkl")'
        )
        print(os.listdir())
        # if os.path.exists("exp.txt"):
        #     with open("exp.txt") as f:
        #         print(f.read())
        result = subprocess.run(
            [
                "firejail",
                "--net=none",
                f"--whitelist={os.path.dirname(os.getcwd())}",
                "--quiet",
                "python",
                "-c",
                code,
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        # with open("exp.txt") as f:
        #     print(f.read())

        fs.put(
            os.path.join(my_temp_dir, "file_dir"),
            remote_dir,
            recursive=True,
            auto_mkdir=True,
        )
        fs.put(
            os.path.join(my_temp_dir, "pkl_dir"),
            remote_dir,
            recursive=True,
            auto_mkdir=True,
        )
        final_time_dict = get_mod_times_dict()
        print(init_time_dict)
        print(final_time_dict)
        modified_files = [
            k
            for k, v in final_time_dict.items()
            if (k not in init_time_dict) or (v != init_time_dict[k])
        ]
        os.chdir("..")
        json_send = {
            "filenames": modified_files,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": str(result.returncode),
        }

        os.chdir(cur_dir)
        return json_send
