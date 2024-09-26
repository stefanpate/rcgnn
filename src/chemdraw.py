from rdkit import Chem
from rdkit.Chem import Draw, Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction

def draw_molecule(mol: str | Mol, size: tuple = (200, 200), use_svg: bool = True):
    '''
    Draw molecule.

    Args
    ----
    mol:str | Mol
        Molecule
    size:tuple
        (width, height)
    use_svg:bool
    '''
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)

    if use_svg:
        drawer = Draw.MolDraw2DSVG(*size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
    else:
        img = Draw.MolToImage(mol, size=size)

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