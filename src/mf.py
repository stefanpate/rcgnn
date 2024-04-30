'''
Matrix factorization
'''
import numpy as np
import torch

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

def negative_sample_bipartite(n_samples, n_rows, n_cols, obs_pairs, seed):
    rng = np.random.default_rng(seed=seed)
    
    # Sample subset of unobserved pairs
    unobs_pairs = []
    while len(unobs_pairs) < n_samples:
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_cols)

        if (i, j) not in obs_pairs:
            unobs_pairs.append((i, j))

    return unobs_pairs
