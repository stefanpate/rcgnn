import hydra
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from logging import getLogger
from collections import defaultdict

log = getLogger(__name__)

def format_operator_output(reactants: list[Chem.Mol], output: list[Chem.Mol], am_to_reactant_idx: dict[int, int]) -> tuple[str, str, tuple[tuple[tuple[int]]]]:
    '''
    Args
    ----
    reactants:Iterable[Chem.Mol]
        Reactants. Note: must be ordered as they match
        lhs operator templates
    output:Iterable[Chem.Mol]
        Output from operator.RunReactants(reactants) that
        matches the actual products
    permuted_idxs:list[int]
        Original indices of reactants in reaction,
        permuted to match the operator
    am_to_reactant_idx:dict[int, int]
        Mapping of atom map numbers to reactant indices
        (i.e. which reactant the atom map number belongs to)
    Returns
    -------
    :tuple[str, str, tuple[tuple[tuple[int]]]]
        Operator aligned reaction without atom mapping
        Operator aligned reaction WITH atom mapping
        Reaction center indices
    '''
    aligned_no_am = '.'.join([Chem.MolToSmiles(m) for m in reactants]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])

    am = 1
    lhs_rc = [[] for _ in reactants]
    rhs_am_rc = []
    for prod in output:
        for atom in prod.GetAtoms():
            atom.SetAtomMapNum(am)
            props = atom.GetPropsAsDict()
            rct_atom_idx = props.get('react_atom_idx')
            rct_idx = props.get('reactant_idx')
            
            if rct_idx is not None:
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
            else:
                old_am = props.get('old_mapno')
                rct_idx = am_to_reactant_idx[old_am]
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
                lhs_rc[rct_idx].append(rct_atom_idx)
                rhs_am_rc.append(atom.GetAtomMapNum())
                
            am += 1

    # Reaction.RunReactants() outputs do not have atoms ordered according to their
    # canonical SMILES ordering. Must go back and forth betweeen SMILES and mol, 
    # then label according to atom map numbers
    rhs_am_smiles = [Chem.MolToSmiles(m, ignoreAtomMapNumbers=True) for m in output]
    rhs_am_recap = [Chem.MolFromSmiles(smi) for smi in rhs_am_smiles]
    rhs_rc = []
    for mol in rhs_am_recap:
        inner_rc = []
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in rhs_am_rc:
                inner_rc.append(atom.GetIdx())
        rhs_rc.append(tuple(inner_rc))
    
    rhs_rc = tuple([tuple(sorted(r)) for r in rhs_rc])
    lhs_rc = tuple([tuple(sorted(r)) for r in lhs_rc])

    aligned_with_am = '.'.join([Chem.MolToSmiles(m, ignoreAtomMapNumbers=True) for m in reactants]) + '>>' + '.'.join(rhs_am_smiles)
    return aligned_no_am, aligned_with_am, (lhs_rc, rhs_rc)

def apply_rule(reactants: list[Chem.Mol], rule_smarts: str) -> list[tuple[str, str, tuple[tuple[tuple[int]]]]]:
    '''
    Apply a rule to reactants and return the resulting reactions as SMILES strings.

    Args
    ----
    reactants: list[Chem.Mol]
        List of reactant molecules as RDKit Mol objects.
    rule_smarts: str
        SMARTS string representing the reaction rule.

    Returns
    -------
    list[tuple[str, str, tuple[tuple[tuple[int]]]]]
        List of tuples, each containing:
        - Reaction SMILES without atom mapping.
        - Reaction SMILES with atom mapping.
        - Tuple of reaction center indices for reactants and products.
    
    '''

    op = AllChem.ReactionFromSmarts(rule_smarts)
    # Mark reactant atoms for atom mapping
    for i, m in enumerate(reactants):
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx', i)

    # Preserve mapping of op am numbers to op reactant indices, i.e., which reactant template
    # each atom map number belongs to (will lose this after running operator)
    am_to_reactant_idx ={}
    for ri in range(op.GetNumReactantTemplates()):
        rt = op.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                am_to_reactant_idx[atom.GetAtomMapNum()] = ri

    ps = op.RunReactants(reactants)
    res = []
    for p in ps:
        res.append(format_operator_output(reactants, p, am_to_reactant_idx))

    return res

@hydra.main(version_base=None, config_path="../configs", config_name="")
def main(cfg):
    # Load reactions
    with open(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}.json") as f:
        sprhea = json.load(f)

    next_rid = max([int(rid) for rid in sprhea.keys()]) + 1
    rxn2rid = {entry['smarts']: rid for rid, entry in sprhea.items()}

    # Load rules
    rules = pd.read_csv(cfg.rules, sep="\t")

    # Load toc
    toc = pd.read_csv(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}.tsv", sep="\t")
    rid2pids = defaultdict(set)
    for _, row in toc.iterrows():
        for rid in row['Label']:
            rid2pids[rid].add(row['Entry'])
    
    unobs_rxns = {}
    negative_pairs = set()
    for rid, entry in tqdm(sprhea.items(), total=len(sprhea), desc="Processing reactions"):
        rxn = entry['smarts']
        reactants = [Chem.MolFromSmiles(smi) for smi in rxn.split(">>")[0].split(".")]

        if not entry['rule']:
            log.warning(f"No rule for reaction {rxn}, skipping")
            continue

        rule_name = entry['rule'][0]

        rule_smarts = rules.loc[rules['Name'] == rule_name, 'SMARTS'].values[0]
        try:
            res = apply_rule(reactants, rule_smarts)
        except Exception as e:
            log.error(f"Error applying rule {rule_name} to reaction {rxn}: {e}")
            continue

        assert any([r[0] == rxn for r in res]), f"Original reaction {rxn} not in operator results"

        this_pids = rid2pids[rid]
        for elt in res:
            gen_rxn, gen_rxn_am, gen_rc = elt

            if gen_rxn == rxn:
                continue
            elif gen_rxn in rxn2rid:
                other_rid = rxn2rid[gen_rxn]
                other_pids = rid2pids[other_rid]

                for pid in other_pids - this_pids:
                    negative_pairs.add((pid, rid))
                
                for pid in this_pids - other_pids:
                    negative_pairs.add((pid, other_rid))
            else:
                new_rid = str(next_rid)
                next_rid += 1
                unobs_rxns[new_rid] = {
                    'smarts': gen_rxn,
                    'smarts_am': gen_rxn_am,
                    'rule': [rule_name],
                    'rc': gen_rc,
                }

                for pid in this_pids:
                    negative_pairs.add((pid, new_rid))