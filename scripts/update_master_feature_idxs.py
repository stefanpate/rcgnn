import pandas as pd

datasets = ['swissprot', 'price', 'new', 'halogenase']
fn = 'master_ec_idxs'
save_to = "../data/"

ec_idxs = set()
for ds in datasets:
    print('\n', ds)
    path = f"../data/{ds}/{ds}.csv"
    df = pd.read_csv(path, delimiter='\t')
    df.set_index('Entry', inplace=True)
    for i, elt in enumerate(df.loc[:, "EC number"]):
        for ec in elt.split(';'):
            ec_idxs.add(ec)

        print(f"{i}", end='\r')

master_df = pd.DataFrame(data = {"EC number": list(ec_idxs)})
master_df.to_csv(save_to + fn + ".csv", sep='\t')