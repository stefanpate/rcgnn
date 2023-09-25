# take json and convert into fasta file
import json
from tqdm import tqdm
from dvc.api import params_show

params = params_show()["make_fasta"]

if __name__ == "__main__":
    print("Load proteins from json")
    with open(params["JSON_PATH"], "r") as f:
        data = json.load(f)

    print("Converting json to fasta")

    count = 0
    with open(params["FASTA_PATH"], "w") as f:
        for k, v in tqdm(data.items()):
            for e in v:
                uniprot, seq = list(e.keys())[0], list(e.values())[0]
                f.write(f">{uniprot} | {k}\n{seq}\n")
                count += 1

    print(
        f"A total of {count} sequences from {len(data)} EC numbers were saved to fasta"
    )
