import json
import pandas as pd
import os
from argparse import ArgumentParser
from collections import defaultdict


gs_res_tmp = "../artifacts/model_evals/mf/tmp"
gs_res_dir = "../artifacts/model_evals/mf"
parser = ArgumentParser()
parser.add_argument("gs_name", type=str, help="Unique grid search name to aggregate")
args = parser.parse_args()
gs_name = args.gs_name

gs_res = defaultdict(list)

# Get jsons matching gs name
for fn in os.listdir(gs_res_tmp):
    if gs_name in fn:

        with open(f"{gs_res_tmp}/{fn}", 'r') as f:
            res = json.load(f)
        
        for k,v in res.items():
            gs_res[k].append(v)

# Aggregate into df

df = pd.DataFrame(gs_res)

# Average and std of losses and fit times
birthday_list = ['train_loss', 'val_loss', 'fit_time', 'hp_idx', 'split_idx']
agg_dict = {
    'train_loss':['mean', 'std'],
    'val_loss':['mean', 'std'],
    'fit_time':'mean',
    **{k:'first' for k in gs_res.keys() if k not in birthday_list}
} 
gs_df = df.groupby("hp_idx").agg(agg_dict)
gs_df.columns = ['mean_train_loss', "std_train_loss",
                "mean_val_loss", "std_val_loss",
                "mean_fit_time"] + [k for k in gs_res.keys() if k not in birthday_list]
gs_df.reset_index(inplace=True)
gs_df.sort_values(by=['mean_val_loss'], inplace=True, ascending=True)
gs_df.to_csv(f"{gs_res_dir}/{gs_name}_gs_res.csv", sep='\t', index=False)