import csv
import pandas as pd
from src.CLEAN.utils import *
import os

rem = []
n_missing = 0
done = os.listdir('../data/clean_data/esm_data')
with open('../data/clean_data/split100.csv', 'r') as f:
    r = csv.reader(f, delimiter='\t')
    for i, row in enumerate(r):
        if i == 0:
            header = row
        elif row[0] + '.pt' not in done:
            rem.append(row)
            n_missing += 1

        if n_missing % 100 ==0:
            print(f"{n_missing}", end='\r')


df = pd.DataFrame(rem, columns=header)
df.to_csv('../data/clean_data/split100_rem.csv', sep='\t', index=False)

csv_to_fasta('../data/clean_data/split100_rem.csv', '../data/clean_data/split100_rem.fasta')
