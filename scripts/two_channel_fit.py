'''
Fit "two channel" (reaction gnn + esm) model
'''

from chemprop import models, nn
from chemprop.data import build_dataloader
from src.utils import load_known_rxns, construct_sparse_adj_mat, ensure_dirs
from src.featurizer import RCVNReactionMolGraphFeaturizer, MultiHotAtomFeaturizer, MultiHotBondFeaturizer
from src.nn import LastAggregation
from src.data import RxnRCDatapoint, RxnRCDataset
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from sklearn.model_selection import KFold
from argparse import ArgumentParser


# CLI parsing
parser = ArgumentParser()
parser.add_argument("-d", "--dataset-name", type=str)
parser.add_argument("-e", "--seed", type=int)
parser.add_argument("-n", "--n-splits", type=int)
parser.add_argument("-s", "--split-idx", type=int)
parser.add_argument("-p", "--hp-idx", type=int)
parser.add_argument("-g", "--gs-name", type=str)
parser.add_argument("-m", "--save-gs-models", action="store_true")