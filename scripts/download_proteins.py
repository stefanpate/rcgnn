import requests
import json
import logging
import urllib
from tqdm import tqdm
from dvc.api import params_show

params = params_show()["download_proteins"]

get_primary_accession = lambda r: r["primaryAccession"]
get_sequence = lambda r: r["sequence"]["value"]


if __name__ == "__main__":
    print("Reading EC values...")
    with open(params["EC_PATH"], "r") as f:
        ec = f.readlines()

    print("Accessing protein sequences from uniprot rest api...")
    results = {}
    for e in tqdm(ec[:]):
        url = urllib.parse.urljoin(
            params["BASE_URL"], f"search?&query=ec:{e}&fields=sequence,ec"
        )
        response = requests.get(url)
        if response.status_code != 200 or "results" not in response.json():
            logging.warning(f"Error with request for {e}")
            continue

        resp = response.json()["results"]
        result = []
        for r in resp:
            try:
                accession, seq = get_primary_accession(r), get_sequence(r)
                result.append({accession: seq})
            except Exception as e:
                continue
        results[e.strip()] = result

    print("Writing to file.")
    with open(params["SEQUENCE_PATH"], "w") as f:
        json.dump(results, f)
