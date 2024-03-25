from src.collaborative_filtering import cf
import time
import pandas as pd

master_ec_path = '../data/master_ec_idxs.csv'
master_ec_df = pd.read_csv(master_ec_path, delimiter='\t')
master_ec_idxs = {k: i for i, k in enumerate(master_ec_df.loc[:, 'EC number'])}

X_name, Y_name = 'swissprot', 'swissprot'
embed_type = 'clean'
kfold = 5
gs_dict = {'ks':[1, 3, 5]}
cf_model = cf(X_name, Y_name, embed_type, master_ec_idxs) # Init
cf_model.fit() # Fit
res = cf_model.kfold_knn_opt(kfold, gs_dict) # k-fold knn opt