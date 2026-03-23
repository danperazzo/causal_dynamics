import sys

sys.path.append("src")

import contextlib
import copy
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from jsonargparse import ArgumentParser
from tqdm import tqdm

from causaldynamics.baselines import KausalEncoderops
from causaldynamics.score import score

warnings.filterwarnings("ignore")

try:
    import wandb
except ImportError as exc:
    raise ImportError(
        "wandb is required for this script. Install it with: pip install wandb"
    ) from exc


DEFAULT_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "Joint_SHD", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
        "epochs": {"values": [1,3, 5, 10]},
        "bootstrap_nums": {"values": [20]},
        "whitening_reg": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-1},
        "bootstrap_ratio": {"distribution": "uniform", "min": 0.6, "max": 1.0},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "hidden_profile": {
            "values": [
                "8-16-8",
                "16-32-16",
                "8-8-8",
                "16-16-16",
                "32-32-16",
                "64-64-32",
                "128-128-64",
            ]
        },
        "batch_size": {"values": [ 128, 256, 512, 1024]},
    },
}


def _parse_hidden_profile(profile: str) -> list[int]:
    return [int(part) for part in profile.split("-") if part]


def _load_dataset(path: Path):
    ds = xr.open_dataset(path)
    try:
        timeseries = ds["time_series"].to_numpy()

        if timeseries.ndim == 4:
            timeseries = timeseries[..., 0]

        # (N, D, T) -> (N, T, D)
        timeseries = timeseries.transpose(1, 0, 2)
        timeseries = np.nan_to_num(timeseries)
        timeseries = (timeseries - timeseries.mean(axis=(0, 1), keepdims=True)) / (
            timeseries.std(axis=(0, 1), keepdims=True) + 1e-8
        )

        if "adjacency_matrix_summary" in ds:
            adj = ds["adjacency_matrix_summary"].to_numpy()
        else:
            adj = ds["adjacency_matrix"].to_numpy()
    finally:
        ds.close()

    if np.all(adj == 1) or np.all(adj == 0):
        return None, None

    return timeseries, adj


def _evaluate_kausal_encoderops(
    *,
    data_dir: str,
    model_kwargs: dict,
    max_systems: int | None,
    max_trajectories: int | None,
    show_progress: bool,
):
    data_path = Path(data_dir) / "data"
    systems = sorted(data_path.glob("*.nc"))
    if max_systems is not None:
        systems = systems[: max(0, max_systems)]

    per_system_rows = []

    systems_iterator = systems
    if not show_progress:
        systems_iterator = tqdm(systems, desc="Processing systems", leave=True)

    for system_path in systems_iterator:
        timeseries, adj_matrix = _load_dataset(system_path)
        if timeseries is None:
            continue

        model = KausalEncoderops(**model_kwargs)
        preds = []
        runtimes = []

        iterator = timeseries
        if max_trajectories is not None:
            iterator = timeseries[: max(0, max_trajectories)]

        if show_progress:
            iterator = tqdm(iterator, desc=f"{system_path.stem}", leave=False)

        for x in iterator:
            start = time.perf_counter()
            try:
                model.run(X=x)
                preds.append(copy.deepcopy(model.adj_matrix))
            except Exception:
                preds.append(np.zeros_like(adj_matrix))
            finally:
                runtimes.append(time.perf_counter() - start)

        preds_array = np.array(preds)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                score_df = score(preds_array, adj_matrix, name="kausal_encoderops")
        except Exception:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                score_df = score(
                    np.zeros((preds_array.shape[0], *adj_matrix.shape), dtype=adj_matrix.dtype),
                    adj_matrix,
                    name="kausal_encoderops",
                )
        metrics = score_df["kausal_encoderops"]

        per_system_rows.append(
            {
                "system": system_path.stem,
                "joint_auroc": float(metrics["Joint AUROC"]),
                "joint_auprc": float(metrics["Joint AUPRC"]),
                "joint_shd": float(metrics["Joint SHD"]),
                "avg_runtime_sec": float(np.mean(runtimes)) if runtimes else float("nan"),
            }
        )

    if not per_system_rows:
        raise RuntimeError(
            "No evaluable systems found. Check --data_dir and dataset contents."
        )

    detail_df = pd.DataFrame(per_system_rows)
    summary = {
        "num_systems": int(detail_df["system"].nunique()),
        "mean_joint_auroc": float(detail_df["joint_auroc"].mean()),
        "mean_joint_auprc": float(detail_df["joint_auprc"].mean()),
        "mean_joint_shd": float(detail_df["joint_shd"].mean()),
        "mean_runtime_sec": float(detail_df["avg_runtime_sec"].mean()),
    }

    return summary, detail_df


def _build_model_kwargs(config: dict) -> dict:
    return {
        "lr": float(config["lr"]),
        "epochs": int(config["epochs"]),
        "bootstrap_nums": int(config["bootstrap_nums"]),
        "bootstrap_ratio": float(config["bootstrap_ratio"]),
        "activation": str(config["activation"]),
        "hidden_channels": _parse_hidden_profile(str(config["hidden_profile"])),
        "batch_size":int(config["batch_size"]),
        "whitening_reg": float(config["whitening_reg"]),
    }


def run_sweep_agent(
    *,
    data_dir: str,
    project: str,
    entity: str | None,
    sweep_id: str,
    count: int,
    max_systems: int | None,
    max_trajectories: int | None,
    show_progress: bool,
):
    def _train():
        run = wandb.init(project=project, entity=entity)
        config = dict(run.config)

        model_kwargs = _build_model_kwargs(config)
        summary, detail_df = _evaluate_kausal_encoderops(
            data_dir=data_dir,
            model_kwargs=model_kwargs,
            max_systems=max_systems,
            max_trajectories=max_trajectories,
            show_progress=show_progress,
        )

        joint_shd_for_log = summary["mean_joint_shd"]
        if not np.isfinite(joint_shd_for_log):
            joint_shd_for_log = 1e12

        log_payload = {
            **summary,
            "Joint_SHD": float(joint_shd_for_log),
            "hidden_channels": "-".join(map(str, model_kwargs["hidden_channels"])),
            "method": "kausal_encoderops",
        }
        wandb.log(log_payload)

        # Store per-system metrics in a table for this run.
        table = wandb.Table(dataframe=detail_df)
        wandb.log({"per_system_metrics": table})

    wandb.agent(sweep_id=sweep_id, function=_train, count=count, project=project, entity=entity)


def main(
    *,
    data_dir: str,
    project: str = "causaldynamics-kausal-encoderops",
    entity: str | None = None,
    sweep_id: str | None = None,
    count: int = 20,
    max_systems: int | None = None,
    max_trajectories: int | None = None,
    show_progress: bool = False,
):
    if sweep_id is None:
        sweep_config = copy.deepcopy(DEFAULT_SWEEP_CONFIG)
        sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
        print(f"Created sweep: {sweep_id}")

    run_sweep_agent(
        data_dir=data_dir,
        project=project,
        entity=entity,
        sweep_id=sweep_id,
        count=count,
        max_systems=max_systems,
        max_trajectories=max_trajectories,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/simple/noise=0.00_confounder=False",
        help="Data directory containing the 'data/' folder with .nc files.",
    )
    parser.add_argument("--project", default="causaldynamics-kausal-encoderops")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--sweep_id",
        default=None,
        help="Existing W&B sweep ID. If omitted, a new sweep is created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of sweep runs for this agent.",
    )
    parser.add_argument(
        "--max_systems",
        type=int,
        default=None,
        help="Optional cap on the number of systems for quicker experiments.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Optional cap on trajectories per system for quicker experiments.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show per-system trajectory progress bars.",
    )

    args = parser.parse_args()
    main(**vars(args))
