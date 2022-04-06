import os
import shutil

with open("_toc.yml") as f:
    current_dir = None
    for line in f.readlines():
        if line.startswith("- caption"):
            current_dir = line.strip()[11:].lower().replace(" ", "_")
            if not os.path.exists(f"notebooks/{current_dir}"):
                os.makedirs(f"notebooks/{current_dir}")
        elif line.strip().startswith("- file"):
            filename = line[20:].strip()
            print(filename)
            base_path = "/home/patel_zeel/explain-ml-book/notebooks"
            try:
                os.system(f"mv {base_path}/{filename}.ipynb {base_path}/{current_dir}/")
            except Exception as e:
                print(e)
