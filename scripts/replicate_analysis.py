import os
from glob import glob
import pandas as pd

os.chdir("../")
os.makedirs("Analysis", exist_ok=True)
li = list()

for exp in ["Heatmap", "Scaling"]:
    for exp_replicate in sorted(glob(exp + "/Experiments/*")):
        for heatmap_folder in sorted(glob(exp_replicate + "/heatmap_*_plot/")):
            if not os.path.isfile(heatmap_folder + "results.tsv"): continue
            li.append(pd.read_csv(heatmap_folder + "results.tsv", sep='\t').assign(filename=heatmap_folder))

df = pd.concat(li, axis=0, ignore_index=True)
print(df)

for name in set(df["name"]):
    if "dnd" not in name: continue
    for mean in set(df["mean"]):
        for log in set(df["log"]):
            sub = df[(df["mean"] == mean) & (df["name"] == name) & (df["log"] == log)]
            sub.to_csv("Analysis/{0}_mean{1}_log{2}.tsv".format(name, mean, log), index=False, sep="\t")
