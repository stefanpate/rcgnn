'''
Matrix factorization
'''
from catalytic_function.utils import load_design_matrix
import numpy as np
import torch

data_dir = "/projects/p30041/spn1560/hiec/data"
scratch_dir = "/scratch/spn1560"

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors, scl_embeds=False):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)

        if scl_embeds:
            self.user_factors.weight = self.scale_embed(self.user_factors.weight)
            self.item_factors.weight = self.scale_embed(self.item_factors.weight)

    def logits(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(dim=1, keepdim=True)

    def forward(self, X):
        user, item = X[:,0].reshape(-1,), X[:,1].reshape(-1,)
        return torch.sigmoid(self.logits(user, item))
    
    def scale_embed(self, embedding):
        np_embed = embedding.detach().numpy()
        np_embed /= np.linalg.norm(np_embed, axis=1).reshape(-1,1)
        return torch.nn.Parameter(torch.FloatTensor(np_embed))
    
class BiasedMatrixFactorization(MatrixFactorization):
    def __init__(self, n_users, n_items, n_factors, scl_embeds=False):
        super().__init__(n_users, n_items, n_factors, scl_embeds)
        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=False)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=False)

        if scl_embeds:
            self.user_factors.weight = self.scale_embed(self.user_factors.weight)
            self.item_factors.weight = self.scale_embed(self.item_factors.weight)
            torch.nn.init.normal_(self.user_biases.weight, mean=0, std=3e-1)
            torch.nn.init.normal_(self.item_biases.weight, mean=0, std=3e-1)

    def forward(self, X):
        user, item = X[:,0].reshape(-1,1), X[:,1].reshape(-1,1)
        dot_prods = super.logits(user, item)
        dot_prods += (self.user_biases(user) + self.item_biases(item))
        return torch.sigmoid(dot_prods)
    
class LinearMatrixFactorization(MatrixFactorization):
    def __init__(self, n_users, n_items, n_factors, scl_embeds=False):
        super().__init__(n_users, n_items, n_factors, scl_embeds)

    def forward(self, X):
        user, item = X[:,0].reshape(-1,1), X[:,1].reshape(-1,1)
        return super.logits(user, item)
    
class PretrainedMatrixFactorization(MatrixFactorization):
    def __init__(
            self,
            user_embeds: np.ndarray | None = None,
            item_embeds: np.ndarray | None = None,
            n_users: int | None = None,
            n_items: int | None = None,
            scl_embeds=False
    ):
        # Decide usage mode
        if (user_embeds is None and n_users is None) or (item_embeds is None and n_items is None):
            raise ValueError("Must provide pretrained embeddings or number of rows/cols for both users and items")
        elif user_embeds is None and item_embeds is None:
            raise ValueError("Must provided pretrained embeddings for either users or items")
        elif user_embeds is not None and item_embeds is None: # Pretrained user embeds
            self.pretrained = 'users'
            n_users, n_factors = user_embeds.shape
        elif user_embeds is None and item_embeds is not None: # Pretrained item embeds
            self.pretrained = 'items'
            n_items, n_factors = item_embeds.shape
        elif user_embeds is not None and item_embeds is not None: # Both pretrained
            if user_embeds.shape[1] != item_embeds.shape[1]:
                raise ValueError("Dimension of user and item embeddings must match")
            self.pretrained = 'both'
            n_users, n_factors = user_embeds.shape
            n_items = item_embeds.shape[0]

        super().__init__(n_users, n_items, n_factors, scl_embeds=False)

        # Overwrite embeddings from super w/ untrainable pretrained embeddings
        if self.pretrained == 'users':
            user_embeds = torch.from_numpy(user_embeds)
            self.user_factors = torch.nn.Embedding.from_pretrained(user_embeds, freeze=True)
        elif self.pretrained == 'items':
            item_embeds = torch.from_numpy(item_embeds)
            self.item_factors = torch.nn.Embedding.from_pretrained(item_embeds, freeze=True)
        elif self.pretrained == 'both':
            user_embeds = torch.from_numpy(user_embeds)
            item_embeds = torch.from_numpy(item_embeds)
            self.user_factors = torch.nn.Embedding.from_pretrained(user_embeds, freeze=True)
            self.item_factors = torch.nn.Embedding.from_pretrained(item_embeds, freeze=True)

        if scl_embeds:
            self.user_factors.weight = self.scale_embed(self.user_factors.weight)
            self.item_factors.weight = self.scale_embed(self.item_factors.weight)

def load_pretrained_embeds(ds_name, special_hps_for_gs, scratch_dir=scratch_dir):
    special_hps_for_model = {}
    for key in ['user_embeds', 'item_embeds']:
        if special_hps_for_gs[key] is not None:
            pt_embeds = np.load(f"{scratch_dir}/{ds_name}_{special_hps_for_gs[key]}_X.npy")
            special_hps_for_model[f"module__{key}"] = pt_embeds

    return special_hps_for_model
