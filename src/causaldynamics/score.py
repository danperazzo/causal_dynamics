import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def score(preds, labs, name="Result"):
    """
    Calculates AUROC and AUPRC metrics given preds and labs.
    Accepts either a 2D or a 3D tensor (batch of summary graphs).
    'name' is used for column naming in the output DataFrame.
    """
    print("Scoring...")

    # Some casting concerning input data type:
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labs, list):
        labs = np.array(labs)
    if isinstance(preds, pd.DataFrame):
        preds = preds.values
    if isinstance(labs, pd.DataFrame):
        labs = labs.values

    # Expand dimensions if a single sample is provided.
    if preds.ndim == 2:
        preds = np.expand_dims(preds, 0)

    preds = np.array(preds)

    # Duplicate labels (assuming similar graph for batched timeseries)
    N, T, D = preds.shape
    labs = np.expand_dims(labs, 0)
    labs = np.repeat(labs, N, 0)
    labs = np.array(labs)

    # Individual scoring for each sample.
    auroc_ind = []
    auprc_ind = []
    for x in range(len(labs)):
        auroc_ind.append(
            roc_auc_score(y_true=labs[x].flatten(), y_score=preds[x].flatten())
        )
        auprc_ind.append(
            average_precision_score(
                y_true=labs[x].flatten(), y_score=preds[x].flatten()
            )
        )

    # Mean individual metrics (if any valid samples exist).
    auroc_ind = np.mean(auroc_ind) if len(auroc_ind) > 0 else float("nan")
    auprc_ind = np.mean(auprc_ind) if len(auprc_ind) > 0 else float("nan")

    # Joint calculation: flatten all samples.
    labs = labs.flatten()
    preds = preds.flatten()

    joint_auroc = roc_auc_score(labs, preds)
    joint_auprc = average_precision_score(labs, preds)
    null_model_auroc = roc_auc_score(labs, np.zeros_like(preds))
    null_model_auprc = average_precision_score(labs, np.zeros_like(preds))

    #  Joint SHD: total mismatches over the entire flattened batch
    preds_binary_3d = (np.array(preds).reshape(N, T, D) >= 0.5).astype(int)
    labs_binary_3d = np.array(labs).reshape(N, T, D).astype(int)
    diff_3d = np.abs(preds_binary_3d - labs_binary_3d)
    joint_shd = np.sum(diff_3d)

    # Joint SHD excluding diagonal entries (only meaningful for square adjacency).
    if T == D:
        off_diag_mask = ~np.eye(T, dtype=bool)
        off_diag_mask = np.broadcast_to(off_diag_mask, (N, T, D))
        joint_shd_no_diag = np.sum(diff_3d[off_diag_mask])
    else:
        joint_shd_no_diag = joint_shd

    out = pd.DataFrame(
        [
            joint_auroc,
            auroc_ind,
            null_model_auroc,
            joint_auprc,
            auprc_ind,
            null_model_auprc,
            joint_shd,
            joint_shd_no_diag,
        ],
        columns=[name],
        index=[
            "Joint AUROC",
            "Individual AUROC",
            "Null AUROC",
            "Joint AUPRC",
            "Individual AUPRC",
            "Null AUPRC",
            "Joint SHD",
            "Joint SHD (No Diagonal)",
        ],
    )
    out.index.name = "Metric"
    return out
