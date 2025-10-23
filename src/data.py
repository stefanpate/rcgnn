'''
Data handling objects
'''
from __future__ import annotations
from dataclasses import dataclass
from chemprop.data.datapoints import _DatapointMixin
from chemprop.data.datasets import ReactionDataset
from chemprop.data.samplers import SeededSampler
from chemprop.data.molgraph import MolGraph
from chemprop.utils import make_mol
from typing import List, Tuple
from rdkit.Chem.rdchem import Mol
from itertools import chain
from chemprop.featurizers import Featurizer
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import torch
import numpy as np
from typing import Iterable

RxnRC = Tuple[List[Mol], List[Mol], List[list]]
MFPDatum = namedtuple("MFPDatum", field_names=["rxn_embed", "y", "x_d", "weight", "gt_mask", "lt_mask"])

@dataclass
class _RCDatapointMixin:
    '''Adds reaction center field to datapoint'''
    reactants: List[Mol]
    products: List[Mol]
    rcs: List[list] # Reaction centers

    @classmethod
    def from_smi(
        cls,
        smarts: str,
        reaction_center: list[list[int]],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs
    ) -> _RCDatapointMixin:
        rcs = list(chain(*reaction_center))
        lhs, rhs = smarts.split('>>')
        reactants = [make_mol(elt, keep_h, add_h) for elt in lhs.split('.')]
        products = [make_mol(elt, keep_h, add_h) for elt in rhs.split('.')]
        
        return cls(reactants, products, rcs, *args, **kwargs)
    
@dataclass
class _PretrainedDatapointMixin:
    rxn_idx: int

    @classmethod
    def from_smi( # Bit of a misnomer but keeping for consistency
        cls,
        rxn_idx: int,
        *args,
        **kwargs
    ) -> _PretrainedDatapointMixin:
        return cls(rxn_idx, *args, **kwargs)
    
@dataclass
class PretrainedDatapoint(_DatapointMixin, _PretrainedDatapointMixin):
    pass

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
    
class MFPDataset(Dataset):
    def __init__(
            self,
            data: List[RxnRCDatapoint],
            featurizer
            ) -> None:
        super().__init__()
        self.data = data
        self.featurizer = featurizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> MFPDatum:
        rxn_embed = self.featurizer(self.data[index].reactants, self.data[index].products)
        return MFPDatum(
            rxn_embed=rxn_embed,
            y=self.data[index].y,
            x_d=self.data[index].x_d,
            weight=self.data[index].weight,
            gt_mask=self.data[index].gt_mask,
            lt_mask=self.data[index].lt_mask,
            )

class PretrainedFPDataset(Dataset):
    def __init__(
            self,
            data: List[PretrainedDatapoint],
            featurizer
            ) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> MFPDatum:
        rxn_embed = self.featurizer(self.data[index].rxn_idx)
        return MFPDatum(
            rxn_embed=rxn_embed,
            y=self.data[index].y,
            x_d=self.data[index].x_d,
            weight=self.data[index].weight,
            gt_mask=self.data[index].gt_mask,
            lt_mask=self.data[index].lt_mask,
            )


def collate_mfps(batch: Iterable[MFPDatum]):
    rxn_embeds, ys, x_ds, weights, gt_masks, lt_masks = zip(*batch)

    return (
        None if rxn_embeds[0] is None else torch.from_numpy(np.array(rxn_embeds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )

def mfp_build_dataloader(dataset, batch_size=64, shuffle=False, seed=None, collate_fcn=collate_mfps):
    if shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    dataloader =  DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        collate_fn=collate_fcn
    )

    return dataloader