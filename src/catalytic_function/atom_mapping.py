from rdkit import Chem
from rdkit.Chem import AllChem, CanonSmiles
from itertools import product, chain, permutations
import re
from copy import deepcopy

def am_label_reactants(reactants, lhs_rc):
    '''
    Set AtomMapNum of reaction center atoms 1:n
    and non-rc atoms as n+1:total_atoms
    '''

    # Init am idx for rc and non rc atoms
    rc_am_idx = 1
    am_idx = sum(len(elt) for elt in lhs_rc) + 1
    for i, rct in enumerate(reactants):
        for a in rct.GetAtoms():
            idx = a.GetIdx()
            if idx in lhs_rc[i]:
                a.SetAtomMapNum(rc_am_idx)
                rc_am_idx += 1
            else:
                a.SetAtomMapNum(am_idx)
                am_idx += 1

    return reactants

def remove_am_products(products):
    for i, pdt in enumerate(products):
        for a in pdt.GetAtoms():
            a.SetAtomMapNum(0)

    return products

def fill_in_output_rc_am(output_smiles, rule):
    matches = get_patts_from_operator(rule, 1, True)
    am_patt = r':\d'
    output_rc_am = [[int(elt.strip(":")) for elt in re.findall(am_patt, match)] for match in matches]
    rhs_patts = get_patts_from_operator(rule, 1)
    rhs_patts = [Chem.MolFromSmarts(elt) for elt in rhs_patts]
    output_mols = tuple([Chem.MolFromSmiles(elt) for elt in output_smiles])
    substruct_matches = [output_mols[i].GetSubstructMatches(rhs_patts[i]) for i in range(len(rhs_patts))]

    for om, potential_matches, rct_am  in zip(output_mols, substruct_matches, output_rc_am):
        for ss_match in potential_matches:
            amless = [om.GetAtomWithIdx(i).GetAtomMapNum() == 0 for i in ss_match]
            if all(amless): # All potential rc atoms have no am numbers
                for idx, am in zip(ss_match, rct_am):
                    om.GetAtomWithIdx(idx).SetAtomMapNum(am)

                break # out of this product

    return [Chem.MolToSmiles(elt) for elt in output_mols]


def atom_map_rxn(rxn_smarts, rule, lhs_rc, rhs_rc, matched_idxs, max_products=10000):
    reactants, products = [elt.split('.') for elt in rxn_smarts.split(">>")]
    operator = Chem.rdChemReactions.ReactionFromSmarts(rule) # Make reaction object from smarts string
    reactants_mol = [Chem.MolFromSmiles(elt) for elt in reactants] # Convert reactant smiles to mol obj
    lhs_patts = get_patts_from_operator(rule, 0)
    lhs_patts = [Chem.MolFromSmarts(elt) for elt in lhs_patts]
    

    # For every permutation of that subset of reactants
    for idx_perm in matched_idxs:
        perm = tuple([reactants_mol[idx] for idx in idx_perm]) # Re-order reactants based on allowable idx perms
        rc_perm = tuple([lhs_rc[idx] for idx in idx_perm])
        outputs = operator.RunReactants(perm, maxProducts=max_products) # Apply rule to that permutation of reactants
        
        substruct_matches = [perm[i].GetSubstructMatches(lhs_patts[i]) for i in range(len(lhs_patts))]
        all_putative_rc_atoms = [set(chain(*elt)) for elt in substruct_matches] # ith element has set of all putative rc atoms of ith reactant

        # Protect all but rc currently considered in each reactant
        for j, reactant_rc in enumerate(rc_perm):
            all_but = all_putative_rc_atoms[j] - set(reactant_rc) # To protect: "all but current rc"
            for protect_idx in all_but:
                perm[j].GetAtomWithIdx(protect_idx).SetProp('_protected', '1')

        perm = am_label_reactants(perm, lhs_rc)

        outputs = operator.RunReactants(perm, maxProducts=max_products) # Run operator with protected atoms

        found_match, output_smiles = compare_operator_outputs_w_products(outputs, products)
        if found_match:
            output_smiles = fill_in_output_rc_am(output_smiles, rule)
            output_mols = [Chem.MolFromSmiles(elt) for elt in output_smiles]
            am_idx = 1
            for rct_idx in range(len(output_mols)):
                for atom_idx in rhs_rc[rct_idx]:
                    atom = output_mols[rct_idx].GetAtomWithIdx(atom_idx)
                    atom.SetAtomMapNum(am_idx)
                    am_idx += 1

            reactant_smiles = [Chem.MolToSmiles(elt) for elt in perm]
            output_smiles = [Chem.MolToSmiles(elt) for elt in output_mols]
            am_smarts = ".".join(reactant_smiles) + ">>" + ".".join(output_smiles)
            return am_smarts


def postsanitize_smiles(smiles_list):
    """Postsanitize smiles after running SMARTS.
    :returns tautomer list of list of smiles"""

    sanitized_list = []
    tautomer_smarts = '[#7H1X3&a:1]:[#6&a:2]:[#7H0X2&a:3]>>[#7H0X2:1]:[#6:2]:[#7H1X3:3]'

    for s in smiles_list:

        temp_mol = Chem.MolFromSmiles(s, sanitize=False)
        aromatic_bonds = [i.GetIdx() for i in temp_mol.GetBonds() if i.GetBondType() == Chem.rdchem.BondType.AROMATIC]

        for i in temp_mol.GetBonds():
            if i.GetBondType() == Chem.rdchem.BondType.UNSPECIFIED:
                i.SetBondType(Chem.rdchem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(temp_mol)
            Chem.rdmolops.RemoveStereochemistry(temp_mol)
            temp_smiles = Chem.MolToSmiles(temp_mol)

        except Exception as msg:
            if 'Can\'t kekulize mol' in str(msg):
                pyrrole_indices = [i[0] for i in temp_mol.GetSubstructMatches(Chem.MolFromSmarts('n'))]

                # indices to sanitize
                for s_i in pyrrole_indices:
                    temp_mol = Chem.MolFromSmiles(s, sanitize=False)
                    if temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 0:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(1)
                    elif temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 1:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(0)
                    try:
                        Chem.SanitizeMol(temp_mol)

                        processed_pyrrole_indices = [i[0] for i in
                                                     temp_mol.GetSubstructMatches(Chem.MolFromSmarts('n'))]
                        processed_aromatic_bonds = [i.GetIdx() for i in
                                                    temp_mol.GetBonds() if i.GetBondType() == Chem.rdchem.BondType.AROMATIC]
                        if processed_pyrrole_indices != pyrrole_indices or aromatic_bonds != processed_aromatic_bonds:
                            continue

                        Chem.rdmolops.RemoveStereochemistry(temp_mol)
                        temp_smiles = Chem.MolToSmiles(temp_mol)
                        break
                    except:
                        continue
                if 'temp_smiles' not in vars():
                    Chem.rdmolops.RemoveStereochemistry(temp_mol)
                    temp_smiles = Chem.MolToSmiles(temp_mol)
                    sanitized_list.append([temp_smiles])
                    continue
            else:
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                temp_smiles = Chem.MolToSmiles(temp_mol)
                sanitized_list.append([temp_smiles])
                continue
        rxn = AllChem.ReactionFromSmarts(tautomer_smarts)

        try:
            tautomer_mols = rxn.RunReactants((Chem.MolFromSmiles(temp_smiles), ))
        except:
            try:
                tautomer_mols = rxn.RunReactants((Chem.MolFromSmiles(temp_smiles, sanitize=False),))
            except:
                continue

        tautomer_smiles = [Chem.MolToSmiles(m[0]) for m in tautomer_mols]
        sanitized_list.append(list(set(tautomer_smiles + [temp_smiles])))

    return list(product(*sanitized_list))

def get_patts_from_operator(smarts:str, side:int, include_am=False):
    '''
    Get substructure patts from half of reation at idx side

    '''

    # lhs smarts pattern
    lhs_smarts = smarts.split('>>')[side]
    if not include_am:
        lhs_smarts = re.sub(r':[0-9]+]', ']', lhs_smarts)

    # identify each fragment
    smarts_list = []
    temp_fragment = []

    # append complete fragments only
    for fragment in lhs_smarts.split('.'):
        temp_fragment += [fragment]
        if '.'.join(temp_fragment).count('(') == '.'.join(temp_fragment).count(')'):
            smarts_list.append('.'.join(temp_fragment))
            temp_fragment = []

            # remove component grouping for substructure matching
            if '.' in smarts_list[-1]:
                smarts_list[-1] = smarts_list[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

    return smarts_list

def compare_operator_outputs_w_products(outputs, products):
    '''
    Args
    outputs: tuple(tuple(Mol))
        Outputs of operator applied to reactants w/ atom mapping numbers
    products: str
        Correct smiles of reaction products w/o atom mapping numbers
        
    Returns
        - found_match: boolean. Whether found operator output matching
        products smarts
        - perm:str | None. The matching tuple of smiles in the order of products
    '''
    products = tuple(products)
    outputs_am = outputs
    outputs = deepcopy(outputs) # Defensive copy
    outputs = [remove_am_products(elt) for elt in outputs] # Remove atom map nums to compare with products
    
    # Convert to SMILES
    for output, output_am in zip(outputs, outputs_am):
        try:
            output = [CanonSmiles(Chem.MolToSmiles(elt)) for elt in output] # Convert pred products to canonical smiles
        except:
            output = [Chem.MolToSmiles(elt) for elt in output]

        try:
            output_am = [CanonSmiles(Chem.MolToSmiles(elt)) for elt in output_am] # Convert pred products to canonical smiles
        except:
            output_am = [Chem.MolToSmiles(elt) for elt in output_am]    
        
        smiles_idx = list(range(len(output)))
        for perm_idx in permutations(smiles_idx):
        # for perm in permutations(output):
            perm = tuple([output[idx] for idx in perm_idx])
            perm_am = tuple([output_am[idx] for idx in perm_idx])
            # Compare predicted to actual products. If mapped, return True
            if perm == products: 
                return True, perm_am
        
        # Last, try fixing kekulization issues
        postsan_output = postsanitize_smiles(output)
        postsan_output_am = postsanitize_smiles(output_am)
        for ps_output, ps_output_am in zip(postsan_output, postsan_output_am): # Iterate over sets of outputs w/ diff tautomers
            smiles_idx = list(range(len(ps_output)))
            for perm_idx in permutations(smiles_idx):
                perm = tuple([ps_output[idx] for idx in perm_idx])
                perm = tuple([ps_output_am[idx] for idx in perm_idx])
            # for perm in permutations(elt):
                if perm == products:
                    return True, perm_am
            
    return False, None

def match_template(rxn, rule_reactants_template, rule_products_template, smi2paired_cof, smi2unpaired_cof):
    '''
    Returns the permuted indices corresponding to
    a match between reactant and rule templates
    '''
    reactants_smi, products_smi = [elt.split('.') for elt in rxn.split(">>")]
    rule_reactants_template = tuple(rule_reactants_template.split(';'))
    rule_products_template = tuple(rule_products_template.split(';'))
    matched_idxs = [] # Return empty if no matches found
    # First check the cardinality of reactants, products matches
    if (len(rule_reactants_template) == len(reactants_smi)) & (len(rule_products_template) == len(products_smi)):

        reactants_template = ['Any' for elt in reactants_smi]
        products_template = ['Any' for elt in products_smi]

        # Search for unpaired cofactors first
        for i, r in enumerate(reactants_smi):
            if r in smi2unpaired_cof:
                reactants_template[i] = smi2unpaired_cof[r]

        for i, p in enumerate(products_smi):
            if p in smi2unpaired_cof:
                products_template[i] = smi2unpaired_cof[p]

        # Search for paired cofactors
        # Only overwriting should be PPi/Pi as phosphate donor/acceptor
        for i, r in enumerate(reactants_smi):
            for j, p in enumerate(products_smi):
                if (r, p) in smi2paired_cof:
                    reactants_template[i] = smi2paired_cof[(r, p)][0]
                    products_template[j] = smi2paired_cof[(r, p)][1]
                elif (p, r) in smi2paired_cof:
                    reactants_template[i] = smi2paired_cof[(p, r)][1]
                    products_template[j] = smi2paired_cof[(p, r)][0]

        reactants_idx_template = [(elt, i) for i, elt in enumerate(reactants_template)]

        # First try to products templates
        product_template_match = False
        for perm in permutations(products_template):
            if perm == rule_products_template:
                product_template_match = True

        # If product templates match
        # find permutations of reactant template that match
        # rule template and keep the indices of those good permutations
        # Else return empty list
        if product_template_match:
            for perm in permutations(reactants_idx_template):
                this_template, this_idx = list(zip(*perm))
                if this_template == rule_reactants_template:
                    matched_idxs.append(this_idx)

    return matched_idxs

if __name__ == '__main__':
    from catalytic_function.utils import load_known_rxns, load_json, save_json
    import pandas as pd
    from tqdm import tqdm

    folded_rxn_path = "../data/sprhea/known_rxns_240310_v2_folded_protein_transcript.json"
    paired_cof_path = "../artifacts/atom_mapping/smi2paired_cof_min.json"
    unpaired_cof_path = "../artifacts/atom_mapping/smi2unpaired_cof_min.json"
    rules_path = "../data/sprhea/minimal1224_all_uniprot.tsv"

    krs = load_known_rxns(folded_rxn_path)
    min_ops = pd.read_csv(rules_path, sep='\t')
    min_ops.set_index("Name", inplace=True)
    smi2paired_cof = load_json(paired_cof_path)
    smi2unpaired_cof = load_json(unpaired_cof_path)
    smi2paired_cof = {tuple(k.split(',')):v[0].split(',') for k,v in smi2paired_cof.items()}

    bad_rxns = []
    am_rxns = {}
    for k in tqdm(krs):
        smarts = krs[k]['smarts']
        rcs = krs[k]['rcs']
        min_rules = krs[k]['min_rules']
        rule = min_ops.loc[min_rules[0], 'SMARTS']
        rule_reactants_template = min_ops.loc[min_rules[0], "Reactants"]
        rule_products_template = min_ops.loc[min_rules[0], "Products"]
        lhs_rc, rhs_rc = rcs
        
        # Match cofactor template
        matched_idxs = match_template(
            rxn=smarts,
            rule_reactants_template=rule_reactants_template,
            rule_products_template=rule_products_template,
            smi2paired_cof=smi2paired_cof,
            smi2unpaired_cof=smi2unpaired_cof
        )

        am_smarts = atom_map_rxn(
            rxn_smarts=smarts,
            rule=rule,
            lhs_rc=lhs_rc,
            rhs_rc=rhs_rc,
            matched_idxs=matched_idxs

        )
        
        if am_smarts:
            am_rxns[k] = am_smarts
        else:
            bad_rxns.append(k)


    save_json(am_rxns, "../artifacts/atom_mapping/atom_mapped_sprhea.json")
    df = pd.DataFrame(data={'Bad rxns':bad_rxns})
    df.to_csv("../artifacts/atom_mapping/sprhea_not_atom_mapped.csv", sep='\t', index=False)
    print(f"{len(am_rxns)} / {len(krs)} atom mapped")

        