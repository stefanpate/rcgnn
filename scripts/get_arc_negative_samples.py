import hydra
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from logging import getLogger
from collections import defaultdict
from copy import deepcopy

log = getLogger(__name__)
_ = rdBase.BlockLogs()

def format_operator_output(rcts_mol: list[Chem.Mol], output: list[Chem.Mol], am_to_reactant_idx: dict[int, int]) -> tuple[str, str, tuple[tuple[tuple[int]]]]:
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
    aligned_no_am = '.'.join([Chem.MolToSmiles(m) for m in rcts_mol]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])

    if len(output) != len(aligned_no_am.split('>>')[1].split('.')):
        raise ValueError("Output length does not match number of products in reaction SMILES")

    am = 1
    lhs_rc = [[] for _ in rcts_mol]
    rhs_am_rc = []
    for prod in output:
        reactants = deepcopy(rcts_mol)
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

def apply_rule(rcts_smi: list[str], rule_smarts: str) -> list[tuple[str, str, tuple[tuple[tuple[int]]]]]:
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
    rcts_mol = [Chem.MolFromSmiles(smi) for smi in rcts_smi]
    op = AllChem.ReactionFromSmarts(rule_smarts)
    # Mark reactant atoms for atom mapping
    for i, m in enumerate(rcts_mol):
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
    try:
        ps = op.RunReactants(rcts_mol)
    except Exception as e:
        return []
    
    res = []
    for p in ps:
        try:
            res.append(format_operator_output(rcts_mol, p, am_to_reactant_idx))
        except Exception as e:
            continue

    return res

@hydra.main(version_base=None, config_path="../configs", config_name="get_arc_negative_samples")
def main(cfg):
    # Load reactions
    with open(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}.json") as f:
        sprhea = json.load(f)

    next_rid = max([int(rid) for rid in sprhea.keys()]) + 1
    rxn2rid = {entry['smarts']: rid for rid, entry in sprhea.items()}

    # Load rules
    rules = pd.read_csv(cfg.rules, sep="\t")

    # Load toc
    toc = pd.read_csv(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}.csv", sep="\t")
    rid2pids = defaultdict(set)
    for _, row in toc.iterrows():
        for rid in row['Label'].split(";"):
            rid2pids[rid].add(row['Entry'])
    
    unobs_rxns = {}
    unobs_smarts = set()
    negative_pairs = defaultdict(set)
    unrecapitulated = set()
    for rid, entry in tqdm(sprhea.items(), total=len(sprhea), desc="Processing reactions"):
        rxn = entry['smarts']
        reactants = [smi for smi in rxn.split(">>")[0].split(".")]

        if not entry['min_rules']:
            log.warning(f"No rule for reaction {rid}, skipping")
            continue

        rule_name = entry['min_rules'][0]
        rule_smarts = rules.loc[rules['Name'] == rule_name, 'SMARTS'].values[0]
        res = apply_rule(reactants, rule_smarts)

        if not any([r[0] == rxn for r in res]):
            unrecapitulated.add(rid)
            log.warning(f"Original reaction {rid} not in operator results")
            continue

        this_pids = rid2pids[rid]
        for elt in res:
            gen_rxn, gen_rxn_am, gen_rc = elt
            gen_rxn_reversed = ">>".join(gen_rxn.split('>>')[::-1])
            gen_rxn_am_reversed = ">>".join(gen_rxn_am.split('>>')[::-1])
            gen_rc_reversed = (gen_rc[1], gen_rc[0])

            if gen_rxn == rxn:
                continue
            elif gen_rxn in rxn2rid:
                other_rid = rxn2rid[gen_rxn]
                other_pids = rid2pids[other_rid]

                for pid in other_pids - this_pids:
                    negative_pairs[pid].add(rid)
                
                for pid in this_pids - other_pids:
                    negative_pairs[pid].add(other_rid)
            elif gen_rxn_reversed in rxn2rid:
                other_rid = rxn2rid[gen_rxn_reversed]
                other_pids = rid2pids[other_rid]

                for pid in other_pids - this_pids:
                    negative_pairs[pid].add(rid)
                
                for pid in this_pids - other_pids:
                    negative_pairs[pid].add(other_rid)
            elif gen_rxn not in unobs_smarts and gen_rxn_reversed not in unobs_smarts:
                unobs_smarts.add(gen_rxn)
                new_rid = str(next_rid)
                next_rid += 1
                unobs_rxns[new_rid] = {
                    'smarts': gen_rxn,
                    'am_smarts': gen_rxn_am,
                    'min_rules': entry['min_rules'],
                    'rcs': gen_rc,
                }

                for pid in this_pids:
                    negative_pairs[pid].add(new_rid)

    for pid, rids in negative_pairs.items():
        for rid in rids:
            if rid in unobs_rxns:
                continue
            
            assert not pid in rid2pids[rid]
    
    log.info(f"Generated {len(unobs_rxns)} unobserved reactions")
    log.info(f"Generated {sum(len(v) for v in negative_pairs.values())} negative reaction-toc entry pairs")
    log.info(f"Failed to recapitulate {len(unrecapitulated)} reactions")
    
    negative_pairs = pd.DataFrame([
        {'Entry': pid, 'Label': ';'.join(sorted(rids))}
        for pid, rids in negative_pairs.items()
    ])
    
    negative_pairs.to_csv(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}_arc_negative_samples.csv", sep="\t", index=False)
    
    with open(Path(cfg.filepaths.data) / cfg.dataset / f"{cfg.toc}_arc_unobserved_reactions.json", 'w') as f:
        json.dump(unobs_rxns, f)

if __name__ == "__main__":
    main()
