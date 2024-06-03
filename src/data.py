'''
Data handling objects
'''
from __future__ import annotations
from dataclasses import dataclass
from chemprop.data.datapoints import _DatapointMixin
from chemprop.data.datasets import ReactionDataset
from chemprop.data.molgraph import MolGraph
from chemprop.utils import make_mol
from typing import List, Tuple
from rdkit.Chem.rdchem import Mol
from itertools import chain
from chemprop.featurizers import Featurizer

RxnRC = Tuple[List[Mol], List[Mol], List[list]]

@dataclass
class _RCDatapointMixin:
    reactants: List[Mol]
    products: List[Mol]
    rcs: List[list] # Reaction centers

    @classmethod
    def from_smi(
        cls,
        rxn: dict,
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs
    ) -> _RCDatapointMixin:
        rcs = list(chain(*rxn['rcs']))
        lhs, rhs = rxn['smarts'].split('>>')
        reactants = [make_mol(elt, keep_h, add_h) for elt in lhs.split('.')]
        products = [make_mol(elt, keep_h, add_h) for elt in rhs.split('.')]
        
        return cls(reactants, products, rcs, *args, **kwargs)

@dataclass
class RxnRCDatapoint(_DatapointMixin, _RCDatapointMixin):
    pass        

@dataclass
class RxnRCDataset(ReactionDataset):
    data: List[RxnRCDatapoint]
    featurizer: Featurizer[RxnRC, MolGraph]
    
    @property
    def mols(self):
        return [(d.reactants, d.products, d.rcs) for d in self.data]