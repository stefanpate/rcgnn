import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)

    def forward(self, user, item):
        return torch.matmul(self.user_factors(user), torch.t(self.item_factors(item)))
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from scipy.sparse import rand as sprand
    import numpy as np
    import time

    # Make up some random explicit feedback ratings
    # and convert to a numpy array
    n_users = 1_00
    n_items = 1_00
    seed = 1234
    n_epochs = 1000
    bs = 10 # Batch size

    rng = np.random.default_rng(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ratings = sprand(n_users, n_items, density=0.01, format="csr")
    ratings.data = rng.integers(1, 5, size=ratings.nnz).astype(np.float64)
    nz_idxs = torch.t(torch.LongTensor(np.array(ratings.nonzero()))).to(device)
    ratings = torch.FloatTensor(ratings.toarray()).to(device)

    model = MatrixFactorization(n_users, n_items, n_factors=20).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)  # learning rate

    train_tic = time.perf_counter()
    loss_curve = []
    for ep in range(n_epochs):
    
        dl = DataLoader(nz_idxs, batch_size=bs, shuffle=True)
        epoch_loss = []
        for i, batch in enumerate(dl):
            # Set gradients to zero
            optimizer.zero_grad()
            batch = batch.to(device)
            row, col = batch[:,0], batch[:,1]

            target = ratings[*torch.meshgrid(row, col, indexing='xy')]
            
            if bs == 1:
                target = np.array([target]).reshape(1,1)

            # Predict and calculate loss
            prediction = model(row, col)
            loss = loss_func(prediction, target)

            # Backpropagate
            loss.backward()

            # Update the parameters
            optimizer.step()

            epoch_loss.append(loss.item())

        epoch_loss = np.array(epoch_loss)
        loss_curve.append((epoch_loss.mean(), epoch_loss.std()))
        print(loss_curve[-1])

    train_toc = time.perf_counter()
    print(f"Trained {n_epochs} epochs w/ batch size {bs} in {train_toc - train_tic} secons")