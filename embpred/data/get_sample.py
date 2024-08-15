import os
import json 
import glob


mapping = {
        "t1": 0,
        "t2": 1,
        "t3": 2,
        "t4": 3,
        "t5": 4,
        "t6": 5,
        "t7": 6, 
        "t8": 7,
        "tPN": 8,
        "tPNf": 9
    }

labels = "data/raw/output.json"
data_dir = "data/raw/DatasetNew/Subj_10707_Emb1/F0"
json_key = "Subj_10707_Emb1"
outfile = "data/interim/sample_emb.csv"

with open("data/raw/output.json", "r") as f:
    labels = json.load(f)[json_key]["timepoint_labels"]

print(labels)

ims = sorted(glob.glob(os.path.join(data_dir, "*.jpeg")), key=lambda x: int(x.split("RUN")[-1].split(".")[0]))
data = []

with open(outfile, "w") as out:
    for i, im in enumerate(ims):
        try:
            label = labels[str(i)]
        except KeyError:
            label =  labels[str(i-1)]
        
        if label in mapping:
            out.write(f"{im},{mapping[label]} \n")
            




