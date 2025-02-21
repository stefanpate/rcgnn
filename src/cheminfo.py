from rdkit import Chem
from rdkit.Chem import Draw, Mol, rdmolfiles
from rdkit.Chem.rdChemReactions import ChemicalReaction
from itertools import combinations
from typing import Iterable

def draw_molecule(molecule: str | Chem.Mol, size: tuple = (200, 200), highlight_atoms: tuple = tuple(), draw_options: dict = {}) -> str:
    '''
    Draw molecule to svg string

    Args
    ----
    mol:str | Chem.Mol
        Molecule
    size:tuple
        (width, height)
    highlight_atoms:tuple
        Atom indices to highlight
    draw_options:dict
        Key-value pairs to set fields of 
        rdkit.Chem.Draw.drawOptions object
    '''
    if type(molecule) is str:
        mol = Chem.MolFromSmiles(molecule)

        # Catch failed MolFromSmiles
        if mol is None: 
            mol = Chem.MolFromSmiles(molecule, sanitize=False)
    elif type(molecule) is Chem.Mol:
        mol = molecule

    drawer = Draw.MolDraw2DSVG(*size)
    _draw_options = drawer.drawOptions()
    for k, v in draw_options.items():
        if not hasattr(_draw_options, k):
            raise ValueError(f"Select from {dir(_draw_options)}")
        elif callable(getattr(_draw_options, k)):
            getattr(_draw_options, k)(v)
        else:
            setattr(_draw_options, k, v)

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()

    return img

def draw_reaction(rxn: str | ChemicalReaction, sub_img_size: tuple = (200, 200), use_svg: bool = True, use_smiles: bool = True):
    '''
    Draw reaction.

    Args
    ----
    rxn:str | ChemicalReaction
    sub_img_size:tuple
        Substrate img size
    use_svg:bool
    use_smiles:bool
    '''
    if type(rxn) is str:
        rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=use_smiles)

    return Draw.ReactionToImage(rxn, useSVG=use_svg, subImgSize=sub_img_size)

def get_r_hop_from_rc(smiles: str, reaction_center: tuple[int], radius: int):
    '''
    Get molecular fragment with structure within r hops of rc

    Args
    ----
    smiles:str
        SMILES string
    reaction_center:tuple[int]
        Atom indices
    radius:int
        Number of hops
    
    Returns
    -------
    fragment_smiles:str
        Fragment SMILES
    fragment_rc:tuple[int]
        Fragment reaction center
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

    reaction_center = [int(elt) for elt in reaction_center] # rdkit wants normal pyhton ints

    # Mark atoms to work out new reaction center indices later
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    # Fragment by atoms
    if (mol.GetNumAtoms() == 1) or (radius == 0 and len(reaction_center) == 1):
        fragment_smiles = rdmolfiles.MolFragmentToSmiles(
            mol=mol,
            atomsToUse=reaction_center,
        )
        fragment_mol = Chem.MolFromSmiles(fragment_smiles)
        if fragment_mol is None:
            fragment_mol = Chem.MolFromSmiles(fragment_smiles, sanitize=False)

    # Fragment by bonds
    else:
        bidxs = []
        if radius == 0:
            for a, b in combinations(reaction_center, 2):
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond:
                    bidxs.append(bond.GetIdx())

        else:
            for aidx in reaction_center:
                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol=mol,
                    radius=radius,
                    rootedAtAtom=aidx
                )
                bidxs += list(env)

            # Beyond full molecule
            if not bidxs:
                bidxs = [bond.GetIdx() for bond in mol.GetBonds()]


        bidxs = list(set(bidxs))

        fragment_mol = Chem.PathToSubmol(
            mol=mol,
            path=bidxs
        )

    fragment_rc = []
    for atom in fragment_mol.GetAtoms():
        if atom.GetAtomMapNum() in reaction_center:
            fragment_rc.append(atom.GetIdx())
        atom.SetAtomMapNum(0)

    fragment_smiles = Chem.MolToSmiles(fragment_mol)
    
    return fragment_smiles, fragment_rc

if __name__ == "__main__":
    # smi1 = 'CC(C)=CCCC(C)=CCCC(C)=CCOP(=O)(O)OP(=O)(O)O'
    # rc1 = (6, 8, 12, 11, 13, 14, 15)
    # r = 1
    # fragment_smiles, fragment_rc = get_r_hop_from_rc(smi1, rc1, r)

    from rdkit.Chem import rdChemReactions
    
    rxn = rdChemReactions.ReactionFromSmarts('[C:1]-[O:2].[C:3](=[O:4])[OH]>>[C:1]-[O:2]-[C:3]=[O:4]')
    reactants = [Chem.MolFromSmiles(x) for x in ('CCO','OC=O')]
    for i,m in enumerate(reactants):
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx',i)
    
    ps = rxn.RunReactants(reactants)
    p0 = ps[0][0]
    for atom in p0.GetAtoms():
        print(atom.GetIdx(),atom.GetPropsAsDict())
    
    atomMapToReactantMap={}
    for ri in range(rxn.GetNumReactantTemplates()):
        rt = rxn.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                atomMapToReactantMap[atom.GetAtomMapNum()] = ri
    
    print()
