import json
import pandas as pd
import os
from argparse import ArgumentParser
from collections import defaultdict


gs_res_dir = "../artifacts/model_evals/mf/tmp"
parser = ArgumentParser()
parser.add_argument("-g", "--gs-name", type=str)
args = parser.parse_args()
gs_name = args.gs_name

gs_res = defaultdict(list)

for fn in os.listdir(gs_res_dir):
    if gs_name in fn:

        with open(fn, 'r') as f:
            res = json.load(f)
        
        for k,v in res.items():
            gs_res[k].append(v)

'''
todo
- make sure you have everything working through model fit called by batch_fit
- putinto one df and save ala mf_gs
- check at batch_fit that you are not repeating gs_names - keep a list in a file somewhere
'''