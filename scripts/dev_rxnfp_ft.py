from src.rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
)
rxn = "CC=O>>CCO"

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
fp = rxnfp_generator.convert(rxn)

print(rxn)
print(fp)