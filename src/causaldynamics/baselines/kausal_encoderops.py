import warnings

import numpy as np
import torch
from kausal import Graph
from kausal.observables import MLPFeaturesEncoderops

warnings.filterwarnings("ignore")

class KausalEncoderops:
    """Kausal baseline using encoder-ops observables."""

    def __init__(
        self,
        lr: float = 0.00006,
        epochs: int = 10,
        hidden_channels: list[int] = [64, 64, 32],
        bootstrap_nums: int = 20,
        bootstrap_ratio: float = 0.98,
        activation: str = "tanh",
        batch_size: int = 1024,
        whitening_reg: float = 0.055,
    ):
        self.lr = lr
        self.epochs = epochs
        self.bootstrap_nums = bootstrap_nums
        self.bootstrap_ratio = bootstrap_ratio
        self.batch_size = batch_size
        self.whitening_reg = whitening_reg

        # Encoder-ops feature maps for marginal and joint observables.
        self.graph_model = Graph(
            marginal_observable=MLPFeaturesEncoderops(
                in_channels=1,
                hidden_channels=hidden_channels,
                out_channels=1,
                activation=activation,
            ),
            joint_observable=MLPFeaturesEncoderops(
                in_channels=2,
                hidden_channels=hidden_channels,
                out_channels=1,
                activation=activation,
            ),
        )

    def run(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"KausalEncoderops expects 2D input (T, D), got shape {X.shape}"
            )

        # Graph API expects shape (variables, channels, timesteps).
        X_t = torch.tensor(X.T[:, None, :], dtype=torch.float32)
        time_shift = min(100, max(1, X_t.shape[-1] - 1))
        n_train = int(X_t.shape[-1] * 0.8)

        self.graph_model.infer(
            X=X_t,
            time_shift=time_shift,
            fit_kwargs={
                "n_train": n_train,
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "whitening_reg": self.whitening_reg,
            },
            bootstrap_kwargs={
                "bootstrap_ratio": self.bootstrap_ratio,
                "bootstrap_nums": self.bootstrap_nums,
            },
        )
        self.adj_matrix = self.graph_model.get_adjacency(p_crit=0.05).numpy()
