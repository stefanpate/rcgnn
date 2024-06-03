from typing import Sequence, Union, List, Tuple
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Atom, HybridizationType, Mol, Bond, BondType
from chemprop.featurizers.base import VectorFeaturizer, GraphFeaturizer
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from dataclasses import InitVar, dataclass
from itertools import chain

class MultiHotAtomFeaturizer(VectorFeaturizer[Atom]):

    """A :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms.

    .. seealso::
        The class provides three default parameterization schemes:

        * :meth:`MultiHotAtomFeaturizer.v1`
        * :meth:`MultiHotAtomFeaturizer.v2`
        * :meth:`MultiHotAtomFeaturizer.organic`

    The generated atom features are ordered as follows:
    * atomic number
    * degree
    * formal charge
    * chiral tag
    * number of hydrogens
    * hybridization
    * aromaticity
    * mass

    .. important::
        Each feature, except for aromaticity and mass, includes a pad for unknown values.

    Parameters
    ----------
    atomic_nums : Sequence[int]
        the choices for atom type denoted by atomic number. Ex: ``[4, 5, 6]`` for C, N and O.
    degrees : Sequence[int]
        the choices for number of bonds an atom is engaged in.
    formal_charges : Sequence[int]
        the choices for integer electronic charge assigned to an atom.
    chiral_tags : Sequence[int]
        the choices for an atom's chiral tag. See :class:`rdkit.Chem.rdchem.ChiralType` for possible integer values.
    num_Hs : Sequence[int]
        the choices for number of bonded hydrogen atoms.
    hybridizations : Sequence[int]
        the choices for an atom’s hybridization type. See :class:`rdkit.Chem.rdchem.HybridizationType` for possible integer values.
    """

    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
        atom_map_nums: Union[Sequence[int]] = None,
        chiral_tags: Union[Sequence[int], None] = None,
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = None if chiral_tags is None else {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}
        self.atom_map_nums = None if atom_map_nums is None else {j: i for i, j in enumerate(atom_map_nums)}

        self._subfeats = [val for val in vars(self).values() if val]
        subfeat_sizes = [1 + len(elt) for elt in self._subfeats]
        subfeat_sizes.extend([1, 1])
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if a is None:
            return x
        
        feat_dict = {
            'atomic_nums': lambda a : a.GetAtomicNum(),
            'degrees': lambda a: a.GetTotalDegree(),
            'formal_charges': lambda a: a.GetFormalCharge(),
            'num_Hs': lambda a: int(a.GetTotalNumHs()),
            'hybridizations': lambda a: a.GetHybridization(),
            'atom_map_nums': lambda a : a.GetAtomMapNum(),
            'chiral_tags': lambda a: int(a.GetChiralTag()),
        }

        feats = [feat_dict[k](a) for k in feat_dict if vars(self)[k]]

        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x

    @classmethod
    def v1(cls, max_atomic_num: int = 100):
        """The original implementation used in Chemprop V1 [1]_, [2]_.

        Parameters
        ----------
        max_atomic_num : int, default=100
            Include a bit for all atomic numbers in the interval :math:`[1, \mathtt{max_atomic_num}]`

        References
        -----------
        .. [1] Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.; Guzman-Perez, A.; Hopper, T.;
        Kelley, B.; Mathea, M.; Palmer, A. "Analyzing Learned Molecular Representations for Property Prediction."
        J. Chem. Inf. Model. 2019, 59 (8), 3370–3388. https://doi.org/10.1021/acs.jcim.9b00237
        .. [2] Heid, E.; Greenman, K.P.; Chung, Y.; Li, S.C.; Graff, D.E.; Vermeire, F.H.; Wu, H.; Green, W.H.; McGill,
        C.J. "Chemprop: A machine learning package for chemical property prediction." J. Chem. Inf. Model. 2024,
        64 (1), 9–17. https://doi.org/10.1021/acs.jcim.3c01250
        """

        return cls(
            atomic_nums=list(range(1, max_atomic_num + 1)),
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    @classmethod
    def v2(cls):
        """An implementation that includes an atom type bit for all elements in the first four rows of the periodic table plus iodine."""

        return cls(
            atomic_nums=list(range(1, 37)) + [53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )
    
    @classmethod
    def no_stereo(cls):
        """"""

        return cls(
            atomic_nums=list(range(1, 37)) + [53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    @classmethod
    def aam(cls):
        """"""

        return cls(
            atomic_nums=list(range(1, 37)) + [53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
            atom_map_nums=list(range(1, 25 +1)) # TODO Get largest reaction center size
        )

    @classmethod
    def organic(cls):
        r"""A specific parameterization intended for use with organic or drug-like molecules.

        This parameterization features:
            1. includes an atomic number bit only for H, B, C, N, O, F, Si, P, S, Cl, Br, and I atoms
            2. a hybridization bit for :math:`s, sp, sp^2` and :math:`sp^3` hybridizations.
        """

        return cls(
            atomic_nums=[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
            ],
        )

class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):
    """A :class:`MultiHotBondFeaturizer` feauturizes bonds based on the following attributes:

    * ``null``-ity (i.e., is the bond ``None``?)
    * bond type
    * conjugated?
    * in ring?
    * stereochemistry

    The feature vectors produced by this featurizer have the following (general) signature:

    +---------------------+-----------------+--------------+
    | slice [start, stop) | subfeature      | unknown pad? |
    +=====================+=================+==============+
    | 0-1                 | null?           | N            |
    +---------------------+-----------------+--------------+
    | 1-5                 | bond type       | N            |
    +---------------------+-----------------+--------------+
    | 5-6                 | conjugated?     | N            |
    +---------------------+-----------------+--------------+
    | 6-8                 | in ring?        | N            |
    +---------------------+-----------------+--------------+
    | 7-14                | stereochemistry | Y            |
    +---------------------+-----------------+--------------+

    **NOTE**: the above signature only applies for the default arguments, as the bond type and
    sterochemistry slices can increase in size depending on the input arguments.

    Parameters
    ----------
    bond_types : Sequence[BondType] | None, default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        the known bond types
    stereos : Sequence[int] | None, default=[0, 1, 2, 3, 4, 5]
        the known bond stereochemistries. See [1]_ for more details

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values
    """

    def __init__(
        self, bond_types: Sequence[BondType] | None = None
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]

    def __len__(self):
        return 1 + len(self.bond_types) + 2

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond)
                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

                E[i : i + 2] = x_e

                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

                i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)
    
@dataclass
class RCVNReactionMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[dict]):
    """A :class:`RCVNReactionMolGraphFeaturizer` attaches reaction center atoms to virtual node

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    def __post_init__(self):
        super().__post_init__()

    def __call__(
        self,
        rxn: Tuple[List[Mol], List[Mol], List[list]],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        
        reactants, products, rcs = rxn
        n_atoms_mol = [mol.GetNumAtoms() for mol in reactants + products]
        cumsum_atoms = [sum(n_atoms_mol[:i]) for i in range(len(n_atoms_mol))]
        n_atoms = sum(n_atoms_mol)
        n_bonds = sum([mol.GetNumBonds() for mol in reactants + products])
        n_virtual_edges = sum(len(elt) for elt in rcs)

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:

            # Featurize atom nodes
            V = []
            for mol in reactants + products:
                for a in mol.GetAtoms():
                    V.append(self.atom_featurizer(a))
            V = np.array(V, dtype=np.single)
        
        E = np.empty((2 * (n_bonds + n_virtual_edges), self.bond_fdim))
        edge_index = [[], []]

        edge_i = 0
        for mol_i, mol in enumerate(reactants + products):

            # Add bond edges
            for mol_u in range(n_atoms_mol[mol_i]):
                for mol_v in range(mol_u + 1, n_atoms_mol[mol_i]):
                    bond = mol.GetBondBetweenAtoms(mol_u, mol_v)
                    
                    if bond is None:
                        continue

                    u = cumsum_atoms[mol_i] + mol_u
                    v = cumsum_atoms[mol_i] + mol_v
                    x_e = self.bond_featurizer(bond)
                    E[edge_i : edge_i + 2] = x_e
                    edge_index[0].extend([u, v])
                    edge_index[1].extend([v, u])
                    edge_i += 2

        # Add virtual node to node feat matrix
        V = np.vstack((V, np.zeros(shape=(1, self.atom_fdim))))
        
        # Add edges between RC atoms and VN
        v = V.shape[0] - 1 # VN located at last node index
        for mol_i in range(len(reactants + products)):
            for mol_u in rcs[mol_i]:
                u = cumsum_atoms[mol_i] + mol_u
                x_e = np.zeros(shape=(self.bond_fdim,))
                E[edge_i : edge_i + 2] = x_e
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])
                edge_i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)