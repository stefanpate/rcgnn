from abc import ABC, abstractmethod
from typing import List

class FingerprintGenerator(ABC):
    """
    Base class for FingerprintGenerator
    """

    @abstractmethod
    def convert(self, rxn_smiles: str) -> List[float]:
        """
        Convert rxn_smiles to fingerprint
        """

    @abstractmethod
    def convert_batch(self, rxn_smiles_batch: List[str]) -> List:
        """
        Convert batch of rxn_smiles to fingerprints
        """