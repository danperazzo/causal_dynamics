from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


class GC_xLSTM:
    """GC-xLSTM baseline wrapper.

    This baseline bridges the neighboring `GC-xLSTM` repository into the
    CausalDynamics baseline interface used by `eval.py`.
    """

    def __init__(
        self,
        context: int = 10,
        embedding_dim: int = 32,
        num_blocks: int = 1,
        lr: float = 1e-4,
        max_iter: int = 100,
        lam: float = 1.0,
        lam_alpha: float | None = None,
        lam_ridge: float = 0.0,
        check_every: int = 25,
        sequence_stride: int = 1,
    ):
        self.context = context
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.lr = lr
        self.max_iter = max_iter
        self.lam = lam
        self.lam_alpha = lam if lam_alpha is None else lam_alpha
        self.lam_ridge = lam_ridge
        self.check_every = check_every
        self.sequence_stride = sequence_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj_matrix: np.ndarray | None = None
        self._available: bool | None = None
        self._import_error: Exception | None = None

    def _add_gc_xlstm_to_path(self) -> None:
        """Add GC-xLSTM roots to `sys.path` for common workspace layouts."""
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "GC-xLSTM",           # e.g. CausalDynamics/GC-xLSTM
            repo_root.parent / "GC-xLSTM",    # e.g. ../GC-xLSTM
            Path.home() / "GC-xLSTM",         # fallback
        ]

        for gc_root in candidates:
            gc_pkg = gc_root / "GC-xLSTM"
            for path in (gc_root, gc_pkg):
                path_str = str(path)
                if path.exists() and path_str not in sys.path:
                    sys.path.insert(0, path_str)

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph with GC-xLSTM."""
        self._add_gc_xlstm_to_path()

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"Expected X to have shape (T, D), got {X_np.shape}")

        if self._available is None:
            try:
                from models.clstm import componentXLSTM, train_model_ista
                from xlstm import mLSTMBlockConfig, xLSTMBlockStackConfig

                self._componentXLSTM = componentXLSTM
                self._train_model_ista = train_model_ista
                self._mLSTMBlockConfig = mLSTMBlockConfig
                self._xLSTMBlockStackConfig = xLSTMBlockStackConfig
                self._available = True
            except Exception as exc:
                self._available = False
                self._import_error = exc

        if not self._available:
            raise ImportError(
                "GC-xLSTM baseline is unavailable. Ensure ../GC-xLSTM exists and "
                "install its dependencies into the active environment. "
                f"Original import error: {self._import_error!r}"
            )

        # componentXLSTM expects batched input of shape (N, T, D).
        X_t = torch.tensor(X_np[None, ...], dtype=torch.float32, device=self.device)

        config = self._xLSTMBlockStackConfig(
            mlstm_block=self._mLSTMBlockConfig(),
            slstm_block=None,
            context_length=self.context,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
            use_lags=False,
            slstm_at=[],
        )

        self.estimator = self._componentXLSTM(
            X_t.shape[-1], hidden=self.embedding_dim, config=config
        ).to(self.device)

        self._train_model_ista(
            self.estimator,
            X_t,
            context=self.context,
            lr=self.lr,
            max_iter=self.max_iter,
            lam=self.lam,
            lam_alpha=self.lam_alpha,
            lam_ridge=self.lam_ridge,
            check_every=self.check_every,
            true_GC=None,
            sequence_stride=self.sequence_stride,
            verbose=int(verbosity),
        )

        # Keep orientation consistent with the existing NGC_LSTM baseline.
        self.adj_matrix = (
            self.estimator.GC(threshold=True).T.detach().cpu().numpy().astype(int)
        )
