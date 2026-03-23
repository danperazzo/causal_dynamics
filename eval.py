import sys
sys.path.append("src")

import copy
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
from tqdm import tqdm

from causaldynamics.baselines import (
    DYNOTEARS,
    FPCMCI,
    GC_xLSTM,
    Kausal,
    KausalEncoderops,
    NGC_LSTM,
    TSCI,
    CUTSPlus,
    PCMCIPlus,
    VARLiNGAM,
    RCD,
    GIN,
    GRASP, 
    TCDF
)
from causaldynamics.creator import logger
from causaldynamics.score import score

warnings.filterwarnings("ignore")

from jsonargparse import ArgumentParser

CAUSAL_MODELS = [
    "pcmciplus",
    "fpcmci",
    "varlingam",
    "dynotears",
    "kausal",
    "kausal_encoderops",
    "ngc_lstm",
    "gc_xlstm",
    "tsci",
    "cutsplus",
    "rcd",
    "grasp",
    "tcdf"
]


def _binarize_prediction(pred_adj: np.ndarray) -> np.ndarray:
    """Convert model predictions to a binary adjacency matrix."""
    pred_adj = np.asarray(pred_adj)
    if pred_adj.ndim == 3:
        pred_adj = pred_adj.mean(axis=0)
    return (pred_adj >= 0.5).astype(int)


def _save_graph_comparison(
    *,
    gt_adj: np.ndarray,
    pred_adj: np.ndarray,
    variable_names: list[str],
    save_path: Path,
    title: str,
):
    """Save ground-truth and predicted graphs side by side."""
    gt_adj = np.asarray(gt_adj)
    pred_adj = _binarize_prediction(pred_adj)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, matrix, subtitle in [
        (axes[0], gt_adj, "Ground Truth"),
        (axes[1], pred_adj, "Predicted"),
    ]:
        _draw_directed_graph(ax=ax, matrix=matrix, variable_names=variable_names)
        ax.set_title(subtitle)

    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _extract_variable_names(input_ds: xr.Dataset, adj_var_name: str, n_vars: int) -> list[str]:
    """Extract variable names from adjacency matrix coordinates when available."""
    adj_da = input_ds[adj_var_name]
    names = None

    if len(adj_da.dims) >= 2:
        out_dim = adj_da.dims[-1]
        if out_dim in adj_da.coords:
            values = adj_da.coords[out_dim].values
            names = [str(v) for v in values]

    if not names or len(names) != n_vars:
        names = [f"x{i}" for i in range(n_vars)]

    return names


def _draw_directed_graph(ax, matrix: np.ndarray, variable_names: list[str]):
    """Draw a directed graph with node labels from an adjacency matrix."""
    n_vars = matrix.shape[0]
    theta = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    radius = 1.0
    positions = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    node_radius = 0.11 if n_vars <= 10 else 0.08

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    # Draw edges first so nodes appear on top.
    for i in range(n_vars):
        for j in range(n_vars):
            if matrix[i, j] == 0:
                continue

            if i == j:
                x_pos, y_pos = positions[i]
                # Self-loop as a prominent curved arrow near the node.
                loop = FancyArrowPatch(
                    posA=(x_pos + node_radius * 0.9, y_pos + node_radius * 1.6),
                    posB=(x_pos - node_radius * 0.4, y_pos + node_radius * 1.45),
                    connectionstyle="arc3,rad=2.4",
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2.0,
                    color="tab:red",
                    alpha=0.95,
                )
                ax.add_patch(loop)
                continue

            start = positions[i]
            end = positions[j]
            direction = end - start
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue

            unit = direction / norm
            start_adj = start + unit * node_radius
            end_adj = end - unit * node_radius
            edge = FancyArrowPatch(
                posA=tuple(start_adj),
                posB=tuple(end_adj),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="tab:gray",
                alpha=0.9,
            )
            ax.add_patch(edge)

    # Draw nodes and labels.
    for idx, (x_pos, y_pos) in enumerate(positions):
        node = Circle((x_pos, y_pos), node_radius, facecolor="tab:blue", edgecolor="black", alpha=0.9)
        ax.add_patch(node)
        ax.text(
            x_pos,
            y_pos,
            variable_names[idx],
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )


def evaluate(*, data_dir: str, causal_model: str | None = None):
    """
    Evaluate causal discovery methods on time series data.

    This function loads time series data and ground truth adjacency matrices,
    applies a specified causal discovery method, and computes performance metrics.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file,

    Returns
    -------
    None
        Results are saved to disk at `data_dir/eval/{causal_model}/<system_name>.nc` as NetCDF files

    Raises
    ------
    ValueError
        If the specified causal model is not supported

    Notes
    -----
    Example usage:
        `python eval.py --data_dir data/climate/coupled_atmos_ocean`

    The function saves evaluation metrics including AUROC and AUPRC scores for each system.
    """
    # Setting up
    DATA_DIR = Path(data_dir) / "data"
    DYN_SYSTEMS = list(DATA_DIR.glob(f"*.nc"))
    summary_rows = []

    # Initialize causal model
    causal_models = {
        "kausal_encoderops": KausalEncoderops(),
        "kausal": Kausal(),
        "pcmciplus": PCMCIPlus(),
        "fpcmci": FPCMCI(),
        "varlingam": VARLiNGAM(),
        "dynotears": DYNOTEARS(),
        "ngc_lstm": NGC_LSTM(),
        "gc_xlstm": GC_xLSTM(),
        "tsci": TSCI(),
        "cutsplus": CUTSPlus(),
        "rcd": RCD(),
        "grasp": GRASP(),
        "tcdf": TCDF()
    }

    single_model_mode = causal_model is not None

    if causal_model is not None:
        causal_model = causal_model.lower()
        if causal_model not in causal_models:
            available = ", ".join(sorted(causal_models.keys()))
            raise ValueError(
                f"Unsupported causal_model '{causal_model}'. Available: {available}"
            )
        causal_models = {causal_model: causal_models[causal_model]}

    print("Initialized causal models: ", list(causal_models.keys()))

    # Run summary graph inference
    for causal_model, _ in causal_models.items():
        model = causal_models.get(causal_model)
        model_unavailable_error = None
        EVAL_DIR = Path(data_dir) / "eval" / causal_model
        GRAPH_DIR = EVAL_DIR / "graphs"
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating {causal_model} on {len(DYN_SYSTEMS)} systems...")

        for dyn_system in DYN_SYSTEMS:

            ## Load data (timeseries and adjacency matrix)
            input_ds = xr.open_dataset(dyn_system)
            timeseries = input_ds["time_series"].to_numpy()

            ## Subsample one variable if multi-dimensional (ie., partially observed / unresolved systems)
            if timeseries.ndim == 4:
                timeseries = timeseries[..., 0]
            timeseries = timeseries.transpose(1, 0, 2)  # of shape (N, T, D)

            ## Handle missing values
            timeseries = np.nan_to_num(timeseries)

            ## z-standardize for stability
            timeseries = (timeseries - timeseries.mean(axis=(0, 1), keepdims=True)) / (
                timeseries.std(axis=(0, 1), keepdims=True) + 1e-8
            )

            ## Extract adjacency matrix
            ## NOTE: skip if adjacency matrix are all ones (AUROC is unable to process singular truth value)
            if "adjacency_matrix_summary" in input_ds:
                adj_var_name = "adjacency_matrix_summary"
                adj_matrix = input_ds[adj_var_name].to_numpy()
            else:
                adj_var_name = "adjacency_matrix"
                adj_matrix = input_ds[adj_var_name].to_numpy()

            variable_names = _extract_variable_names(
                input_ds=input_ds,
                adj_var_name=adj_var_name,
                n_vars=adj_matrix.shape[0],
            )
        
            if np.all(adj_matrix == 1) or np.all(adj_matrix == 0):
                input_ds.close()
                continue

            ## Infer graph for each trajectory
            ## NOTE: safe run -- assigns zeros for trajectory-level estimated graph if run fails
            est_adj_matrix = []
            system_runtimes = []
            for x in tqdm(timeseries):
                start_time = time.perf_counter()

                try:
                    if model_unavailable_error is not None:
                        est_adj_matrix.append(np.zeros_like(adj_matrix))
                        continue

                    model.run(X=x)
                    est_adj_matrix.append(copy.deepcopy(model.adj_matrix))

                except (ImportError, ModuleNotFoundError) as exc:
                    if model_unavailable_error is None:
                        model_unavailable_error = exc
                        logger.exception(
                            f"Model {causal_model} unavailable due to dependency/import issue: {exc}"
                        )
                    est_adj_matrix.append(np.zeros_like(adj_matrix))

                except Exception as exc:
                    logger.exception(
                        f"Fails for a trajectory in {dyn_system} with model {causal_model}: {exc}"
                    )
                    est_adj_matrix.append(np.zeros_like(adj_matrix))

                finally:
                    elapsed = time.perf_counter() - start_time
                    system_runtimes.append(elapsed)

            ## Compute scores
            ### NOTE: safe eval -- assigns zeros for all estimated graph if evaluation fails
            try:
                est_score_df = score(
                    preds=np.array(est_adj_matrix), labs=adj_matrix, name=causal_model
                )

            except:
                est_score_df = score(
                    preds=np.zeros(
                        (timeseries.shape[0], *adj_matrix.shape), dtype=adj_matrix.dtype
                    ),
                    labs=adj_matrix,
                    name=causal_model,
                )

            ## Save
            est_score = est_score_df[causal_model].values.squeeze()
            eval_ds = xr.Dataset(
                data_vars={
                    "Joint_AUROC": est_score[0],
                    "Individual_AUROC": est_score[1],
                    "Null_AUROC": est_score[2],
                    "Joint_AUPRC": est_score[3],
                    "Individual_AUPRC": est_score[4],
                    "Null_AUPRC": est_score[5],
                    "Joint_SHD": est_score[6]
                },
                attrs={"description": f"Causal discovery performance metrics ({model})"},
            )

            eval_ds.to_netcdf(EVAL_DIR / f"{dyn_system.stem}.nc")
            _save_graph_comparison(
                gt_adj=adj_matrix,
                pred_adj=np.array(est_adj_matrix),
                variable_names=variable_names,
                save_path=GRAPH_DIR / f"{dyn_system.stem}.png",
                title=f"{causal_model} | {dyn_system.stem}",
            )

            summary_rows.append(
                {
                    "method": causal_model,
                    "system": dyn_system.stem,
                    "variables": ",".join(variable_names),
                    "Joint_AUROC": float(est_score[0]),
                    "Joint_AUPRC": float(est_score[3]),
                    "Joint_SHD": float(est_score[6]),
                    "Avg_Runtime_Sec": float(np.mean(system_runtimes)) if system_runtimes else float("nan"),
                }
            )
            input_ds.close()

    if summary_rows:
        detail_df = pd.DataFrame(summary_rows)
        summary_df = (
            detail_df.groupby("method", as_index=False)
            .agg(
                num_systems=("system", "nunique"),
                variables=("variables", lambda s: ";".join(sorted(set(s)))),
                Joint_AUROC=("Joint_AUROC", "mean"),
                Joint_AUPRC=("Joint_AUPRC", "mean"),
                Joint_SHD=("Joint_SHD", "mean"),
                Avg_Runtime_Sec=("Avg_Runtime_Sec", "mean"),
            )
            .sort_values("method")
        )

        print("\nMethod-level average performance across systems:")
        print(summary_df[["method", "num_systems", "Joint_AUROC", "Joint_AUPRC", "Joint_SHD", "Avg_Runtime_Sec"]])

        eval_dir = Path(data_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        summary_csv = eval_dir / "summary_metrics.csv"
        if single_model_mode and summary_csv.exists():
            existing_summary_df = pd.read_csv(summary_csv)
            methods_to_update = set(summary_df["method"].astype(str))
            combined_summary_df = pd.concat(
                [
                    existing_summary_df[
                        ~existing_summary_df["method"].astype(str).isin(methods_to_update)
                    ],
                    summary_df,
                ],
                ignore_index=True,
            ).sort_values("method")
            combined_summary_df.to_csv(summary_csv, index=False)
            print(f"Updated summary CSV with method(s) {sorted(methods_to_update)} at {summary_csv}")
        else:
            summary_df.to_csv(summary_csv, index=False)
            print(f"Saved aggregated summary CSV to {summary_csv}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Directory containing the time series data and adjacency matrices in a netCDF file",
    )
    parser.add_argument(
        "--causal_model",
        default=None,
        help=(
            "Optional single baseline to run (e.g. 'gc_xlstm'). "
            "If omitted, all baselines are evaluated."
        ),
    )
    args = parser.parse_args()
    evaluate(**vars(args))
