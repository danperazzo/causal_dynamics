import warnings

import numpy as np
warnings.filterwarnings("ignore")


from kausal import Graph
from kausal.observables import MLPFeatures
import torch


class Kausal:
    """
    Kausal baseline.

    Reference:
        [1] https://www.nature.com/articles/s42005-025-02426-1
    """

    def __init__(self, lr: float = 1e-3, epochs: int = 1, hidden_channels: list = [8, 16], bootstrap_nums: int = 30,
                 bootstrap_ratio: float = 0.9):
        """Initialize regressor"""
        super(Kausal, self).__init__()

        self.lr = lr
        self.epochs = epochs
        self.bootstrap_nums = bootstrap_nums
        self.bootstrap_ratio = bootstrap_ratio


        ## Initialize the Graph object
        self.graph_model = Graph(
            marginal_observable = MLPFeatures(in_channels=1, hidden_channels=hidden_channels, out_channels=1),
            joint_observable = MLPFeatures(in_channels=2, hidden_channels=hidden_channels, out_channels=1)
        )

    def run(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Kausal expects 2D input (T, D), got shape {X.shape}")

        # Kausal's Graph API passes X[i] to pairwise inference, where each effect/cause
        # is expected to have shape (channels, timesteps).
        X_t = torch.tensor(X.T[:, None, :], dtype=torch.float32)
        n_train = max(1, int(X_t.shape[-1] * 0.8))
        time_shift = 1

        """Estimate lagged adjacency graph"""
        self.graph_model.infer(
                    X=X_t,
                    time_shift=time_shift,
                    fit_kwargs={
                        'n_train': n_train,
                        'epochs': self.epochs,
                        'lr': self.lr,
                        'batch_size': n_train,
                    },
                    bootstrap_kwargs={
                        'bootstrap_ratio': self.bootstrap_ratio,
                        'bootstrap_nums': self.bootstrap_nums,
                    }
                )
        self.adj_matrix = self.graph_model.get_adjacency(p_crit=0.05).numpy()