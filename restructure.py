import os
import glob
import shutil

dir = "dataset"

def get_subdirs(directory):
    return [os.path.join(directory, sdir) for sdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, sdir))]

def count_subdirs(directory):
    return len(get_subdirs(directory))

for subdir in glob.glob(f"{dir}/*"):
    if os.path.isdir(subdir):
        if count_subdirs(subdir) != 7:
            subsubdirs = get_subdirs(subdir)
            for subsubdir in subsubdirs:
                new_name = f"{os.path.basename(subdir)}_{os.path.basename(subsubdir)}"
                new_path = os.path.join(dir, new_name)
                shutil.move(subsubdir, new_path)
            shutil.rmtree(subdir)  # Remove the original directory and its contents
