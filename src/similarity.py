'''
Libary of similarity functions, clustering support functions etc.
'''
import re
from itertools import chain
from rdkit import Chem
from rdkit.Chem import rdFMCS, Mol, AllChem
from typing import Iterable, Dict
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from Bio import Align

def embedding_similarity_matrix(X: np.ndarray, X2: np.ndarray = None, dt: np.dtype = np.float32):
    '''
    Multiplies X X.T or X X2.T and saves result
    '''
    if X2 is not None:
        S = np.matmul(X, X2.T)
    else:
        S = np.matmul(X, X.T)

    S = 1 / (1 + np.exp(-S))  
    return S.astype(dt)

def tanimoto_similarity_matrix(rxns:dict[str, dict], matrix_idx_to_rxn_id: dict[int, str], dt: np.dtype = np.float32):
    '''
    Computes aligned-substrates-tanimoto-similarity 
    similarity matrix for set of reactions

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    matrix_idx_to_rxn_id:dict
        Maps reaction's similarity matrix / embed matrix index to its reaction index from rxns
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    '''
    fields = ['smarts', 'min_rules']
    S = np.eye(N=len(matrix_idx_to_rxn_id)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
    for i in range(len(matrix_idx_to_rxn_id) - 1):
        id_i = matrix_idx_to_rxn_id[i]
        smarts_i, rules_i = [rxns[id_i][f] for f in fields]
        print(f"Rxn # {i} : {matrix_idx_to_rxn_id[i]}", end='\r')
        for j in range(i + 1, len(matrix_idx_to_rxn_id)):
            id_j = matrix_idx_to_rxn_id[j]
            smarts_j, rules_j = [rxns[id_j][f] for f in fields]

            if tuple(rules_i) != tuple(rules_j):
                rules_j = rules_j[::-1]
            
                if tuple(rules_i) != tuple(rules_j):
                    continue
                else:
                    smarts_j = ">>".join(smarts_j.split(">>")[::-1])

            S_idxs.append((i, j))
            to_do.append(([smarts_i, smarts_j],))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(_wrap_rxn_tani, to_do), total=len(to_do)))
    
    if S_idxs:
        i, j = [np.array(elt) for elt in zip(*S_idxs)]
        S[i, j] = res
        S[j, i] = res

    return S.astype(dt)

def bag_of_tanimoto_similarity_matrix(rxns:dict[str, dict], matrix_idx_to_rxn_id: dict[int, str], dt: np.dtype = np.float32):
    '''
    Computes similarity matrix using bag of tanimoto similarity: tanimoto similarity on vectors gotten
    by taking the abs diff of sum of mfps on each side of reaction.

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    matrix_idx_to_rxn_id:dict
        Maps reaction's similarity matrix / embed matrix index to its reaction index from rxns
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    '''
    S = np.eye(N=len(matrix_idx_to_rxn_id)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
    for i in range(len(matrix_idx_to_rxn_id) - 1):
        id_i = matrix_idx_to_rxn_id[i]
        smarts_i = rxns[id_i]['smarts']
        print(f"Rxn # {i} : {matrix_idx_to_rxn_id[i]}", end='\r')
        for j in range(i + 1, len(matrix_idx_to_rxn_id)):
            id_j = matrix_idx_to_rxn_id[j]
            smarts_j = rxns[id_j]['smarts']

            S_idxs.append((i, j))
            to_do.append(([smarts_i, smarts_j],))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(_wrap_bag_of_tani, to_do), total=len(to_do)))
    
    if S_idxs:
        i, j = [np.array(elt) for elt in zip(*S_idxs)]
        S[i, j] = res
        S[j, i] = res

    return S.astype(dt)

def mcs_similarity_matrix(rxns:dict[str, dict], matrix_idx_to_rxn_id: dict[int, str], dt: np.dtype = np.float32):
    '''
    Computes regular MCS 
    similarity matrix for set of reactions

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    matrix_idx_to_rxn_id:dict
        Maps reaction's similarity matrix / embed matrix index to its reaction index from rxns
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    '''

    fields = ['smarts', 'min_rules']
    S = np.eye(N=len(matrix_idx_to_rxn_id)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
    for i in range(len(matrix_idx_to_rxn_id) - 1):
        id_i = matrix_idx_to_rxn_id[i]
        smarts_i, rules_i = [rxns[id_i][f] for f in fields]
        print(f"Rxn # {i} : {matrix_idx_to_rxn_id[i]}", end='\r')
        for j in range(i + 1, len(matrix_idx_to_rxn_id)):
            id_j = matrix_idx_to_rxn_id[j]
            smarts_j, rules_j = [rxns[id_j][f] for f in fields]

            if tuple(rules_i) != tuple(rules_j):
                rules_j = rules_j[::-1]
            
                if tuple(rules_i) != tuple(rules_j):
                    continue
                else:
                    smarts_j = ">>".join(smarts_j.split(">>")[::-1])

            S_idxs.append((i, j))
            to_do.append(([smarts_i, smarts_j],))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(_wrap_rxn_mcs, to_do), total=len(to_do)))
    
    if S_idxs:
        i, j = [np.array(elt) for elt in zip(*S_idxs)]
        S[i, j] = res
        S[j, i] = res

    return S.astype(dt)

def rcmcs_similarity_matrix(rxns:dict[str, dict], rules:pd.DataFrame, matrix_idx_to_rxn_id: dict[int, str], dt: np.dtype = np.float32):
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
    S = np.eye(N=len(matrix_idx_to_rxn_id)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
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

            S_idxs.append((i, j))
            to_do.append(([smarts_i, smarts_j], (rcs_i, rcs_j), patts))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(_wrap_rxn_mcs, to_do), total=len(to_do)))
    
    if S_idxs:
        i, j = [np.array(elt) for elt in zip(*S_idxs)]
        S[i, j] = res
        S[j, i] = res

    return S.astype(dt)

def merge_cd_hit_clusters(
        pairs:Iterable[tuple],
        Drxn,
        sim_i_to_rxn_id:Dict,
        cd_hit_clusters:Dict,
        distance_cutoff
    ):
    rxn2i = {v: k for k, v in sim_i_to_rxn_id.items()}
    upid2cluster = {}
    for cluster, upids in cd_hit_clusters.items():
        for upid in upids:
            upid2cluster[upid] = cluster
    
    for i in range(len(pairs) - 1):
        rxn_idx_1 = rxn2i[pairs[i][1]]

        if pairs[i][0] not in upid2cluster:
            continue

        cluster1 = upid2cluster[pairs[i][0]]
        
        for j in range(i + 1, len(pairs)):
            rxn_idx_2 = rxn2i[pairs[j][1]]

            if pairs[j][0] not in upid2cluster:
                continue

            cluster2 = upid2cluster[pairs[j][0]]

            if Drxn[rxn_idx_1, rxn_idx_2] < distance_cutoff and cluster1 != cluster2:
                cd_hit_clusters[cluster1] = cd_hit_clusters[cluster1] + cd_hit_clusters[cluster2]
                for upid in cd_hit_clusters[cluster2]:
                    upid2cluster[upid] = cluster1
                cd_hit_clusters.pop(cluster2)
    
    id2cluster = {}
    for upid, rxnid in pairs:
        if upid in upid2cluster:
            id2cluster[(upid, rxnid)] = upid2cluster[upid]

    return id2cluster
  
def combo_similarity_matrix(pairs, Sseq, Srxn, idx2seq, idx2rxn):
    seq2idx = {v: k for k, v in idx2seq.items()}
    rxn2idx = {v: k for k, v in idx2rxn.items()}
    sim_i_to_id = {i : id for i, id in enumerate(pairs)}
    S = np.eye(N=len(pairs)) # Similarity matrix
    
    for i in range(len(sim_i_to_id) - 1):
        seq1, rxn1 = sim_i_to_id[i]
        seq_idx1, rxn_idx1 = seq2idx[seq1], rxn2idx[rxn1]
        for j in range(i + 1, len(sim_i_to_id)):
            seq2, rxn2 = sim_i_to_id[j]
            seq_idx2, rxn_idx2 = seq2idx[seq2], rxn2idx[rxn2]
            sim = max(Sseq[seq_idx1, seq_idx2], Srxn[rxn_idx1, rxn_idx2])
            S[i, j] = sim
            S[j, i] = sim

    return S, sim_i_to_id
            
def global_sequence_identity(seq1:str, seq2:str, aligner:Align.PairwiseAligner):
    '''
    Aligns two sequences and calculates global sequence
    identity = # aligned residues / min(len(seq1), len(seq2))

    Args
    ------
    seq:str
        Amino acid sequence
    aligner:Bio.Align.PairwiseAligner
        Pairwise aligner object
    '''
    alignment = aligner.align(seq1, seq2)[0]
    t_segments, q_segments = alignment.aligned

    talign = ''
    qalign = ''
    for i in range(t_segments.shape[0]):
        talign += alignment.target[t_segments[i, 0] : t_segments[i, 1]]
        qalign += alignment.query[q_segments[i, 0] : q_segments[i, 1]]

    ct = 0
    for t, q in zip(talign, qalign):
        if t == q:
            ct += 1
    
    return ct / min(len(alignment.target), len(alignment.query))

def homology_similarity_matrix(sequences:Dict[str, str], aligner:Align.PairwiseAligner):
    '''
    With multiprocessing, Computes reaction center MCS 
    similarity matrix for set of reactions

    Args
    ----
    sequences:Dict[str, str]
        {id: amino acid sequence}
    aligner:Bio.Align.PairwiseAligner
        Pairwise aligner object
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    sim_i_to_rxn_idx:dict
        Maps sequences's similarity matrix index to its sequence id
    '''
    sim_i_to_id = {i : id for i, id in enumerate(sequences.keys())}
    S = np.eye(N=len(sim_i_to_id)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
    for i in range(len(sim_i_to_id) - 1):
        seq1 = sequences[sim_i_to_id[i]]
        print(f"Sequence # {i} : {sim_i_to_id[i]}", end='\r')
        for j in range(i + 1, len(sim_i_to_id)):
            seq2 = sequences[sim_i_to_id[j]]
            S_idxs.append((i, j))
            to_do.append((seq1, seq2, aligner))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(wrap_gsi, to_do), total=len(to_do)))
    
    i, j = [np.array(elt) for  elt in zip(*S_idxs)]
    S[i, j] = res
    S[j, i] = res

    return S, sim_i_to_id

def molecule_mcs_similarity(molecules: Iterable[Mol], reaction_centers: Iterable[tuple[int]] = None, patt:str = None, norm: str='max', return_match_patt: bool = False):
    '''
    Calculates MCS similarity score for a pair of molecules. If reaction_centers and patt
    are provided, reaction center MCS score will be provided, otherwise a straight MCS score is provided.

    Args
    ----
    molecules: Iterable[Mol]
        Molecules to compare
    reaction_centers: Iterable[tuple[int]]
        Molecules' reaction centers in same order as molecules
    patt:str
        Reaction center substructure pattern in SMARTS
    norm:str
        'min' normalizes by # atoms in smallest molecule, 'max' the largest
    return_match_patt: bool
    
    Returns
    -------
    score:float
        Reaction center max common substructure score [0, 1]
    mcs_patt:str
        SMARTS match pattern if return_match_patt == True
    '''
    if reaction_centers is None and patt is None:
        mode = 'mcs'
    elif reaction_centers is not None and patt is not None:
        mode = 'rcmcs'
    else:
        raise ValueError("Either provide both reaction_centers and patt for RCMCS or provide neither for MCS")

    if mode == 'rcmcs' and len(molecules) != len(reaction_centers):
        raise ValueError("Mismatch in number of molecules and reaction centers provided")

    if mode == 'rcmcs':
        rc_scalar = 100

        def replace(match):
            atomic_number = int(match.group(1))
            return f"[{atomic_number * rc_scalar}#{atomic_number}"
        
        atomic_sub_patt = r'\[#(\d+)'
        patt = re.sub(atomic_sub_patt, replace, patt) # Mark reaction center patt w/ isotope number

        # Mark reaction center vs other atoms in substrates w/ isotope number
        for mol, rc in zip(molecules, reaction_centers):
            for atom in mol.GetAtoms():
                if atom.GetIdx() in rc:
                    atom.SetIsotope(atom.GetAtomicNum() * rc_scalar) # Rxn ctr atom
                else:
                    atom.SetIsotope(atom.GetAtomicNum()) # Non rxn ctr atom

        cleared, patt = _mcs_precheck(molecules, reaction_centers, patt) # Prevents FindMCS default behavior of non-rc-mcs

        if not cleared and return_match_patt:
            return 0.0, ''
        elif not cleared:
            return 0.0

    # Find MCS
    res = rdFMCS.FindMCS(
        molecules,
        seedSmarts=patt if mode == 'rcmcs' else '',
        atomCompare=rdFMCS.AtomCompare.CompareIsotopes if mode == 'rcmcs' else rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=True,
        timeout=10
    )

    # Compute score
    if res.canceled and return_match_patt:
        return 0.0, ''
    elif res.canceled:
        return 0.0
    elif norm == 'min':
        score = res.numAtoms / min(m.GetNumAtoms() for m in molecules)
    elif norm == 'max':
        score = res.numAtoms / max(m.GetNumAtoms() for m in molecules)

    if return_match_patt:
        match_patt = res.smartsString
        return score, match_patt
    else:
        return score

def reaction_mcs_similarity(
        reactions: Iterable[str], reaction_centers: tuple[Iterable[Iterable[tuple[int]]]] = None, patts: Iterable[Iterable[str]] = None,
        norm: str = 'max', analyze_sides: str = 'both'
    ):
    '''
    Calculates atom-weighted MCS similarity score for a pair of reactions with aligned substrates. If reaction_centers and patts
    are provided, reaction center MCS score will be provided, otherwise a straight atom-weighted MCS score is provided
    
    Args
    -------
    reactions: Iterable[str]
        Reaction smarts 'rsmi1.rsmi2>psmi1.psmi2'
    reaction_centers: tuple[Iterable[Iterable[tuple[int]]]]
        Innermost tuples have reaction center atom indices for a reactant / product.
        Next level up is a side of a reaction, followed by a reaction, and a top-level
        tuple containing the two reactions' iterables.
        ( [[ (0,1), (2, 3) ], [ (0,1), (2, 3) ]], # For rxn 1
            [[ (0,1), (2, 3) ], [ (0,1), (2, 3) ]]) # For rxn 2
    patts: Iterable[Iterable[str]]
        SMARTS patterns of reaction center fragments organized
            as one Iterable per side, one SMARTS string per reactant / product
    norm: str
        Normalize by 'max' or 'min' number of atoms of compare molecules
    analyze_sides: str
        'left' or 'both'
    '''
    if reaction_centers is not None and patts is not None:
        mode = 'rcmcs'
    elif reaction_centers is None and patts is None:
        mode = 'mcs'
    else:
        raise ValueError("Either provide both reaction_centers and patts for RCMCS or provide neither for MCS")

    if analyze_sides == 'left':
        molecules = [[Chem.MolFromSmiles(smi) for smi in rxn.split('>>')[0].split('.')] for rxn in reactions]
        if mode == 'rcmcs':
            reaction_centers = [(elt for elt in rc[0]) for rc in reaction_centers] # Collapse rxn centers from each side of reaction
            patts = (elt for elt in patts[0])
    elif analyze_sides == 'both':
        molecules = [[Chem.MolFromSmiles(smi) for smi in fractionate(rxn)] for rxn in reactions] # List of mols for each rxn
        if mode == 'rcmcs':
            reaction_centers = [chain(*rc) for rc in reaction_centers] # Collapse rxn centers from each side of reaction
            patts = chain(*patts) # Collapse patts from each side of reaction
    
    molecules = zip(*molecules) # Transpose to list of molecule pairs

    if mode == 'rcmcs':
        reaction_centers = zip(*reaction_centers) # Transpose to list of rc pairs

    cum_atoms = 0
    cum_score = 0
    if mode == 'rcmcs':
        iterargs = zip(molecules, reaction_centers, patts)
    elif mode == 'mcs':
        iterargs = ((elt, ) for elt in molecules)
    for args in iterargs:
        score = molecule_mcs_similarity(*args, norm=norm)

        if norm == 'max':
            n_atoms = max([m.GetNumAtoms() for m in args[0]])
        elif norm == 'min':
            n_atoms = min([m.GetNumAtoms() for m in args[0]])
        
        cum_score += score * n_atoms
        cum_atoms += n_atoms

    return cum_score / cum_atoms

def reaction_tanimoto_similarity(reactions: Iterable[str], norm: str = 'max', analyze_sides: str = 'both'):
    '''
    Calculates atom-weighted tanimoto similarity score for a pair of reactions with aligned substrates.
    
    Args
    -------
    reactions: Iterable[str]
        Reaction smarts 'rsmi1.rsmi2>psmi1.psmi2'
    analyze_sides: str
        'left' or 'both'
    '''
    if analyze_sides == 'left':
        molecules = [[Chem.MolFromSmiles(smi) for smi in rxn.split('>>')[0].split('.')] for rxn in reactions]
    elif analyze_sides == 'both':
        molecules = [[Chem.MolFromSmiles(smi) for smi in fractionate(rxn)] for rxn in reactions] # List of mols for each rxn
    
    molecules = zip(*molecules) # Transpose to list of molecule pairs

    cum_atoms = 0
    cum_score = 0
    for mols in molecules:
        mfps = [morgan_fingerprint(mol) for mol in mols]
        score = tanimoto_similarity(*mfps)

        if norm == 'max':
            n_atoms = max([m.GetNumAtoms() for m in mols])
        elif norm == 'min':
            n_atoms = min([m.GetNumAtoms() for m in mols])
        
        cum_score += score * n_atoms
        cum_atoms += n_atoms

    return cum_score / cum_atoms

def bag_of_tanimoto_similarity(reactions: Iterable[str]) -> float:
    '''
    Computes tanimoto similarity between abs(rct_mfp_sum - pdt_mfp_sum)
    '''
    rxn_vecs = []
    for rxn in reactions:
        mols = [[Chem.MolFromSmiles(smi) for smi in side.split(".")] for side in rxn.split(">>")]
        mfps = [[morgan_fingerprint(mol) for mol in side] for side in mols]
        rxn_vecs.append(abs(sum(mfps[0]) - sum(mfps[1])))

    return tanimoto_similarity(*rxn_vecs)


def extract_operator_patts(rxn_smarts:str, side:int):
    '''
    Returns list of smarts patts, one for each 
    molecule on a given side of reaction

    Args
    ----
    rxn_smarts:str
        reaction smarts 'rsmi1.rsmi2>psmi1.psmi2'
    side:int
        0 = left side, 1 = right
    '''

    # side smarts pattern
    side_smarts = rxn_smarts.split('>>')[side]
    side_smarts = re.sub(r':[0-9]+]', ']', side_smarts)

    # identify each fragment
    smarts_list = []
    temp_fragment = []

    # append complete fragments only
    for fragment in side_smarts.split('.'):
        temp_fragment += [fragment]
        if '.'.join(temp_fragment).count('(') == '.'.join(temp_fragment).count(')'):
            smarts_list.append('.'.join(temp_fragment))
            temp_fragment = []

            # remove component grouping for substructure matching
            if '.' in smarts_list[-1]:
                smarts_list[-1] = smarts_list[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

    return smarts_list

def fractionate(rxn_smarts: str) -> tuple[str]:
    '''
    Returns one tuple of SMILES strings from left to right
    given a SMARTS reaction
    '''
    sides = rxn_smarts.split('>>')
    return tuple(chain(*[side.split('.') for side in sides]))

def tanimoto_similarity(bit_vec_1: np.ndarray, bit_vec_2: np.ndarray, dtype=np.float32):
    dot = np.dot(bit_vec_1, bit_vec_2)
    return dtype(dot / (bit_vec_1.sum() + bit_vec_2.sum() - dot))

def morgan_fingerprint(mol: Mol, radius: int = 2, length: int = 2**10, use_features: bool = False, use_chirality: bool = False):
    vec = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=length,
        useFeatures=use_features,
        useChirality=use_chirality,
    )
    return np.array(vec, dtype=np.float32)

def _mcs_precheck(molecules: Iterable[Mol], reaction_centers: Iterable[tuple[int]], patt:str):
    '''
    Modifies single-atom patts and pre-checks ring info
    to avoid giving FindMCS a non-common-substructure which
    results in non-reaction-center-inclusive MCSes
    '''
    if patt.count('#') == 1:
        patt = _handle_single_atom_patt(molecules, reaction_centers, patt)
    
    cleared = _check_ring_info(molecules, reaction_centers)

    return cleared, patt

def _handle_single_atom_patt(molecules: Iterable[Mol], reaction_centers: Iterable[tuple[int]], patt:str):
    '''    
    Pre-pends wildcard atom and bond to single-atom
    patt if mols share a neighbor w/ common isotope,
    ring membership, & bond type between
    '''
    for rc in reaction_centers:
        if len(rc) > 1:
            raise ValueError("Reaction centers must be single atoms")

    setlist = []
    for mol, rc in zip(molecules, reaction_centers):
        aidx = rc[0] # Get single atom idx
        neighbor_set = set()
        for neighbor in mol.GetAtomWithIdx(aidx).GetNeighbors():
            nidx = neighbor.GetIdx()
            nisotope = neighbor.GetIsotope()
            in_ring = neighbor.IsInRing()
            bond_type = mol.GetBondBetweenAtoms(aidx, nidx).GetBondType()
            neighbor_set.add((nisotope, in_ring, bond_type))
        setlist.append(neighbor_set)
    
    if len(set.intersection(*setlist)) > 0:
        patt = '*~' + patt
    
    return patt

def _check_ring_info(molecules: Iterable[Mol], reaction_centers: Iterable[tuple[int]]):
    ''''
    Rejects any mol pair where corresponding
    reaction center atoms have distinct ring membership
    '''
    rc_lens = [len(rc) for rc in reaction_centers]
    if len(set(rc_lens)) > 1:
        raise ValueError("Reaction centers must have same number of atoms")
    else:
        rc_len = rc_lens[0]

    for i in range(rc_len):
        ring_status = set()
        for mol, rc in zip(molecules, reaction_centers):
            ring_status.add(mol.GetAtomWithIdx(rc[i]).IsInRing())
        
        if len(ring_status) != 1:
            return False
        
    return True

def wrap_gsi(args):
    return global_sequence_identity(*args)

def _wrap_rxn_mcs(args):
    return reaction_mcs_similarity(*args)

def _wrap_rxn_tani(args):
    return reaction_tanimoto_similarity(*args)

def _wrap_bag_of_tani(args):
    return bag_of_tanimoto_similarity(*args)

if __name__ == '__main__':
    reaction1 = "CC=O.O>>CC.O.O"
    reaction2 = "CC=O.O>>CC.O.O"
    print(bag_of_tanimoto_similarity([reaction1, reaction2]))