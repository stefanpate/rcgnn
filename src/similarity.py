'''
Libary of similarity functions, clustering support functions etc.
'''
import re
from itertools import chain
from rdkit import Chem
from rdkit.Chem import rdFMCS
from typing import Iterable, Dict
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from Bio import Align

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

def wrap_gsi(args):
    return global_sequence_identity(*args)

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

def calc_molecule_rcmcs(
        mol_rc1,
        mol_rc2,
        patt,
        norm='max'
    ):
    '''
    Args
    ----
    mol_rc1:Tuple[Mol, Tuple[int]]
        1st molecule and tuple of its reaction center atom indices
    mol_rc2:Tuple[Mol, Tuple[int]]
        2nd molecule and tuple of its reaction center atom indices
    patt:str
        Reaction center substructure pattern in SMARTS
    norm:str - Normalization to get an index out of
        prcmcs. 'min' normalizes by # atoms in smaller
        of the two substrates, 'max' by that of the larger
    
    Returns
    -------
    rcmcs:float
        Reaction center max common substructure score [0, 1]
    '''
    rc_scalar = 100

    def _replace(match):
        atomic_number = int(match.group(1))
        return f"[{atomic_number * rc_scalar}#{atomic_number}"
    
    atomic_sub_patt = r'\[#(\d+)'
    pairs = (mol_rc1, mol_rc2)

    patt = re.sub(atomic_sub_patt, _replace, patt) # Mark reaction center patt w/ isotope number

    # Mark reaction center vs other atoms in substrates w/ isotope number
    for pair in pairs:
        for atom in pair[0].GetAtoms():
            if atom.GetIdx() in pair[1]:
                atom.SetIsotope(atom.GetAtomicNum() * rc_scalar) # Rxn ctr atom
            else:
                atom.SetIsotope(atom.GetAtomicNum()) # Non rxn ctr atom

    cleared, patt = mcs_precheck(mol_rc1, mol_rc2, patt) # Prevents FindMCS default behavior of non-rc-mcs

    if not cleared:
        return 0.0

    # Get the mcs that contains the reaction center pattern
    molecules = [elt[0] for elt in pairs]

    res = rdFMCS.FindMCS(
        molecules,
        seedSmarts=patt,
        atomCompare=rdFMCS.AtomCompare.CompareIsotopes,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=True,
        timeout=10
    )

    # Compute prc mcs index
    if res.canceled:
        return 0
    elif norm == 'min':
        return res.numAtoms / min(m.GetNumAtoms() for m in molecules)
    elif norm == 'max':
        return res.numAtoms / max(m.GetNumAtoms() for m in molecules)

def calc_rxn_rcmcs(
        rxn_rc1:Iterable,
        rxn_rc2:Iterable,
        norm:str='max'
    ):
    '''
    Calculates atom-weighted reaction rcmcs score of aligned reactions

    Args
    -------
    rxn_rc:Iterable of len = 3
        rxn_rc[0]:str - reaction smarts 'rsmi1.rsmi2>psmi1.psmi2'
        rxn_rc[1]:Iterable[Iterable[Iterable[int]]] - innermost iterables have reaction
            center atom indices for a reactant / product. Each side of reaction has separate
            iterable of aidx iterables
        rxn_rc[2]:Iterable[Iterable[str]] - SMARTS patterns of reaction center fragments organized
            the same way as rxn_rc[1] except here, one SMARTS string per reactant / product
    '''
    patts1 = tuple(chain(*rxn_rc1[2]))
    patts2 = tuple(chain(*rxn_rc2[2]))
    if patts1 != patts2: # Reaction centers are distinct
        return 0.0

    smiles = [fractionate(rxn_rc1[0]), fractionate(rxn_rc2[0])]
    rc_idxs = [chain(*rxn_rc1[1]), chain(*rxn_rc2[1])]
    molecules= [[Chem.MolFromSmiles(smi) for smi in elt] for elt in smiles]
    mol_rcs1, mol_rcs2 = [list(zip(molecules[i], rc_idxs[i])) for i in range(2)]
    
    n_atoms = 0
    rcmcs = 0
    for mol_rc1, mol_rc2, patt in zip(mol_rcs1, mol_rcs2, patts1):
        rcmcs_i = calc_molecule_rcmcs(mol_rc1, mol_rc2, patt, norm=norm)

        if norm == 'max':
            atoms_i = max(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
        elif norm == 'min':
            atoms_i = min(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
        
        rcmcs += rcmcs_i * atoms_i
        n_atoms += atoms_i

    return rcmcs / n_atoms

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

def fractionate(rxn_smarts):
    '''
    Returns molecule smiles in list of two
    lists, one sublist per each side of rxn
    '''
    sides = rxn_smarts.split('>>')
    return tuple(chain(*[side.split('.') for side in sides]))

def rcmcs_similarity_matrix(rxns:dict, rules:pd.DataFrame, norm='max'):
    '''
    Computes reaction center MCS similarity matrix for set of reactions

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    rules:pd.DataFrame
        Minimal rules indexed by rule name, e.g., 'rule0123', w/ 'SMARTS' col
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    sim_i_to_rxn_idx:dict
        Maps reaction's similarity matrix index to its reaction index from rxns
    '''
    sim_i_to_rxn_idx = {i : idx for i, idx in enumerate(rxns.keys())}
    fields = ['smarts', 'rcs', 'min_rules']
    S = np.eye(N=len(sim_i_to_rxn_idx)) # Similarity matrix

    for i in range(len(sim_i_to_rxn_idx) - 1):
        print(f"Processing {i} : {sim_i_to_rxn_idx[i]}")
        rowi = [rxns[sim_i_to_rxn_idx[i]][f] for f in fields]
        for j in range(i + 1, len(sim_i_to_rxn_idx)):
            rowj = [rxns[sim_i_to_rxn_idx[j]][f] for f in fields]
            
            if tuple(rowi[2]) != tuple(rowj[2]): # Distinct minimal rules:
                rev_rules = rowj[2][::-1]

                if tuple(rowi[2]) != tuple(rev_rules): # Rules are still distinct
                    continue
                else: # Directions now match
                    rowj[2] = rev_rules
                    rowj[1] = rowj[1][::-1]
                    rowj[0] = ">>".join(rowj[0].split('>>')[::-1])

            # Convert rules to patts
            patts = [extract_operator_patts(rules.loc[rowi[2][k], 'SMARTS'], 0) for k in range(2)]
            rxn_rci = rowi[:2] + [patts]
            rxn_rcj = rowj[:2] + [patts]
            
            rcmcs = calc_rxn_rcmcs(rxn_rci, rxn_rcj, norm=norm)
            S[i, j] = rcmcs
            S[j, i] = rcmcs

    return S, sim_i_to_rxn_idx

def rcmcs_similarity_matrix_mp(rxns:dict, rules:pd.DataFrame, norm='max'):
    '''
    With multiprocessing, Computes reaction center MCS 
    similarity matrix for set of reactions

    Args
    ----
    rxns:dict
        Reactions dict. Must contains 'smarts', 'rcs', 'min_rules' keys
        in each reaction_idx indexed sub-dict
    rules:pd.DataFrame
        Minimal rules indexed by rule name, e.g., 'rule0123', w/ 'SMARTS' col
    
    Returns
    -------
    S:np.ndarray
        nxn similarity matrix
    sim_i_to_rxn_idx:dict
        Maps reaction's similarity matrix index to its reaction index from rxns
    '''
    sim_i_to_rxn_idx = {i : idx for i, idx in enumerate(rxns.keys())}
    fields = ['smarts', 'rcs', 'min_rules']
    S = np.eye(N=len(sim_i_to_rxn_idx)) # Similarity matrix

    to_do = []
    S_idxs = []
    print("Preparing reaction pairs\n")
    for i in range(len(sim_i_to_rxn_idx) - 1):
        print(f"Rxn # {i} : {sim_i_to_rxn_idx[i]}", end='\r')
        rowi = [rxns[sim_i_to_rxn_idx[i]][f] for f in fields]
        for j in range(i + 1, len(sim_i_to_rxn_idx)):
            rowj = [rxns[sim_i_to_rxn_idx[j]][f] for f in fields]
            
            if tuple(rowi[2]) != tuple(rowj[2]): # Distinct minimal rules:
                rev_rules = rowj[2][::-1]

                if tuple(rowi[2]) != tuple(rev_rules):
                    continue
                else: # Directions now match
                    rowj[2] = rev_rules
                    rowj[1] = rowj[1][::-1]
                    rowj[0] = ">>".join(rowj[0].split('>>')[::-1])

            # Convert rule name -> patts
            patts = [extract_operator_patts(rules.loc[rowi[2][k], 'SMARTS'], 0) for k in range(2)]
            rxn_rci = rowi[:2] + [patts]
            rxn_rcj = rowj[:2] + [patts]

            S_idxs.append((i, j))
            to_do.append((rxn_rci, rxn_rcj, norm))

    print("\nProcessing pairs\n")    
    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(wrap_rcmcs, to_do), total=len(to_do)))
    
    i, j = [np.array(elt) for  elt in zip(*S_idxs)]
    S[i, j] = res
    S[j, i] = res

    return S, sim_i_to_rxn_idx

def mcs_precheck(mol_rc1, mol_rc2, patt):
    '''
    Modifies single-atom patts and pre-checks ring info
    to avoid giving FindMCS a non-common-substructure which
    results in non-reaction-center-inclusive MCSes
    '''
    if patt.count('#') == 1:
        patt = handle_single_atom_patt(mol_rc1, mol_rc2, patt)
    
    cleared = check_ring_infor(mol_rc1, mol_rc2)

    return cleared, patt

def handle_single_atom_patt(mol_rc1, mol_rc2, patt):
    '''
    Pre-pends wildcard atom and bond to single-atom
    patt if mols share a neighbor w/ common isotope,
    ring membership, & bond type between
    '''
    couples = [set(), set()]
    for i, mol_rc in enumerate([mol_rc1, mol_rc2]):
        mol = mol_rc[0]
        rc_idx = mol_rc[1][0]
        for neighbor in mol.GetAtomWithIdx(rc_idx).GetNeighbors():
            nidx = neighbor.GetIdx()
            nisotope = neighbor.GetIsotope()
            in_ring = neighbor.IsInRing()
            bond_type = mol.GetBondBetweenAtoms(rc_idx, nidx).GetBondType()
            couples[i].add((nisotope, in_ring, bond_type))

    if len(couples[0] & couples[1]) > 0:
        patt = '*~' + patt
    
    return patt

def check_ring_infor(mol_rc1, mol_rc2):
    ''''
    Rejects any mol pair where corresponding
    reaction center atoms have distinct ring membership
    '''
    mol1, mol2 = mol_rc1[0], mol_rc2[0]
    for aidx1, aidx2 in zip(mol_rc1[1], mol_rc2[1]):
        a1_in_ring = mol1.GetAtomWithIdx(aidx1).IsInRing()
        a2_in_ring = mol2.GetAtomWithIdx(aidx2).IsInRing()
        
        if a1_in_ring ^ a2_in_ring:
            return False
        
    return True

def wrap_rcmcs(args):
        return calc_rxn_rcmcs(*args)