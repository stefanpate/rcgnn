import torch
from transformers import BertModel
from .tokenization import (
    SmilesTokenizer
)
from .core import (
    FingerprintGenerator
)
from typing import List
from . import TRANSFORMERS_DIR

def get_default_model_and_tokenizer(model='bert_ft', force_no_cuda=False):

    model_path = TRANSFORMERS_DIR / model
    tokenizer_vocab_path = model_path / "vocab.txt"
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")
    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(
        tokenizer_vocab_path
    )
    return model, tokenizer

class RXNBERTFingerprintGenerator(FingerprintGenerator):
    """
    Generate RXNBERT fingerprints from reaction SMILES
    """

    def __init__(self, model: BertModel, tokenizer: SmilesTokenizer, force_no_cuda=False):
        super(RXNBERTFingerprintGenerator).__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    def convert(self, rxn_smiles: str):
        """
        Convert rxn_smiles to fingerprint

        Args:
            rxn_smiles (str): precursors>>products
        """
        bert_inputs = self.tokenizer.encode_plus(rxn_smiles,
                                                max_length=self.model.config.max_position_embeddings,
                                                padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.model(
                **bert_inputs
            )


        embeddings = output['last_hidden_state'].squeeze()[0].cpu().numpy().tolist()
        return embeddings

    def convert_batch(self, rxn_smiles_list: List[str]):
        bert_inputs = self.tokenizer.batch_encode_plus(rxn_smiles_list,
                                                       max_length=self.model.config.max_position_embeddings,
                                                       padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model(
                **bert_inputs
            )


        # [CLS] token embeddings in position 0
        embeddings = output['last_hidden_state'][:, 0, :].cpu().numpy().tolist()
        return embeddings
