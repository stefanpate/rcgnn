from src.similarity import reaction_mcs_similarity, extract_operator_patts
import pandas as pd
import numpy as np

def rcmcs_similarity_matrix(rxns:dict[str, dict], rules:pd.DataFrame, matrix_idx_to_rxn_id: dict[int, str], dt: np.dtype = np.float32, norm='max'):
    '''
    Computes reaction center MCS 
    similarity matrix for set of reactions

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    rules:pd.DataFrame
        Minimal rules indexed by rule name, e.g., 'rule0123', w/ 'SMARTS' col
    matrix_idx_to_rxn_id:dict
        Maps reaction's similarity matrix / embed matrix index to its reaction index from rxns
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    '''
    fields = ['smarts', 'rcs', 'min_rules']
    for i in range(len(matrix_idx_to_rxn_id) - 1):
        id_i = matrix_idx_to_rxn_id[i]
        smarts_i, rcs_i, rules_i = [rxns[id_i][f] for f in fields]
        patts = [extract_operator_patts(rules.loc[rule, 'SMARTS'], side=0) for rule in rules_i]
        print(f"Rxn # {i} : {matrix_idx_to_rxn_id[i]}", end='\r')
        for j in range(i + 1, len(matrix_idx_to_rxn_id)):
            id_j = matrix_idx_to_rxn_id[j]
            smarts_j, rcs_j, rules_j = [rxns[id_j][f] for f in fields]

            if tuple(rules_i) != tuple(rules_j):
                rules_j = rules_j[::-1]
            
                if tuple(rules_i) != tuple(rules_j):
                    continue
                else:
                    rcs_j = rcs_j[::-1]
                    smarts_j = ">>".join(smarts_j.split(">>")[::-1])
            
            score = reaction_mcs_similarity([smarts_i, smarts_j], (rcs_i, rcs_j), patts)
            print(id_i, id_j, score)

if __name__ == '__main__':
    from src.config import filepaths
    from src.utils import construct_sparse_adj_mat, load_json
    dataset = 'sprhea'
    toc = 'v3_folded_pt_ns'

    adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset, toc)
    krs = load_json(filepaths['data_sprhea'] / f"{toc}.json")
    rules = pd.read_csv(filepaths['data'] / "minimal1224_all_uniprot.tsv", sep='\t', )
    rules.set_index('Name', inplace=True)

    rid1 = str(3166)
    rid2 = str(7189)
    rxn1 = krs[rid1]
    rxn2 = krs[rid2]
    reactions = {rid1: rxn1, rid2: rxn2}
    idx2id = {0: rid1, 1: rid2}
    rcmcs_similarity_matrix(
        rxns=reactions,
        rules=rules,
        matrix_idx_to_rxn_id=idx2id,
    )