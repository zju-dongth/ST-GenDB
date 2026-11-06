"""
Validation-Set Subspace Projection and Zero-Shot Predicate Log-Odds Calibration for ST-GenDB.

Provides frozen validation-split calibration utilities (Section V-A):
  1. SubspaceProjector / fit_multitier_subspace: Regularized Within-Class Covariance
     Normalization (WCCN) and Fisher LDA projection blended with centered cosine structure.
  2. PredicateCalibrator / fit_predicate_calibration: Validation log-odds scaling and
     decision-boundary centering for zero-shot VLM Yes/No probabilities.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


def _canonicalize_columns(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    for col_i in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, col_i])))
        if out[pivot, col_i] < 0.0:
            out[:, col_i] *= -1.0
    return out


@dataclass
class SubspaceProjector:
    """
    Frozen validation-fitted linear subspace projector mapping D-dimensional tier features
    into an L2-normalized out_dim discriminative embedding space (Section V-A).
    """
    mean_vec: np.ndarray
    proj_matrix: np.ndarray
    mix_alpha: float = 0.85
    raw_basis: Optional[np.ndarray] = field(default=None, repr=False)

    def transform_batch(self, mat: np.ndarray) -> np.ndarray:
        """Projects a 2-D feature array of shape (N, D) into (N, out_dim) unit vectors."""
        x = np.asarray(mat, dtype=np.float32)
        centered = x - self.mean_vec.astype(np.float32)
        disc = _l2_normalize_rows(centered @ self.proj_matrix.astype(np.float32))
        alpha = float(np.clip(self.mix_alpha, 0.0, 1.0))
        if alpha >= 1.0 - 1e-6:
            return disc.astype(np.float32)
        if self.raw_basis is not None:
            base = _l2_normalize_rows(centered @ self.raw_basis.astype(np.float32))
        else:
            q_orth, _ = np.linalg.qr(self.proj_matrix.astype(np.float64))
            q_orth = _canonicalize_columns(q_orth).astype(np.float32)
            base = _l2_normalize_rows(centered @ q_orth)
        blended = alpha * disc + (1.0 - alpha) * base
        return _l2_normalize_rows(blended).astype(np.float32)

    def transform(self, vec: np.ndarray) -> np.ndarray:
        """Projects a single 1-D feature vector of shape (D,) into an (out_dim,) unit vector."""
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        return self.transform_batch(v)[0]

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        return self.transform(vec)


def _fit_single_tier_projector(
    x_raw: np.ndarray,
    entity_ids: Sequence[Any],
    out_dim: int,
    reg_lambda: float,
    wccn_alpha: float,
    mix_alpha: float,
) -> SubspaceProjector:
    x = np.asarray(x_raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Calibration embeddings must have shape (N, D) with N >= 2.")
    n_samples, feat_dim = x.shape
    if len(entity_ids) != n_samples:
        raise ValueError("Length of entity_ids must match number of embedding rows.")

    mu = x.mean(axis=0, keepdims=True)
    x_c = x - mu
    _, s_vals, vt = np.linalg.svd(x_c, full_matrices=False)
    pos_rank = int(np.sum(s_vals > 1e-6))
    r_keep = max(1, min(feat_dim, n_samples - 1, max(out_dim * 2, pos_rank)))
    r_keep = min(r_keep, vt.shape[0])
    basis = _canonicalize_columns(vt[:r_keep].T)
    z_mat = x_c @ basis
    k_keep = min(int(out_dim), r_keep)
    pca_basis = basis[:, :k_keep].copy()

    labels = [str(eid) for eid in entity_ids]
    counts = Counter(labels)
    multiview_classes = sorted(cid for cid, cnt in counts.items() if cnt >= 2)
    if len(multiview_classes) < 2:
        return SubspaceProjector(
            mean_vec=mu.squeeze(0).astype(np.float32),
            proj_matrix=pca_basis.astype(np.float32),
            mix_alpha=float(mix_alpha),
            raw_basis=pca_basis.astype(np.float32),
        )

    sw = np.zeros((r_keep, r_keep), dtype=np.float64)
    sb = np.zeros((r_keep, r_keep), dtype=np.float64)
    label_arr = np.array(labels, dtype=object)
    for cid in sorted(counts.keys()):
        zc = z_mat[label_arr == cid]
        mc = zc.mean(axis=0, keepdims=True)
        dc = zc - mc
        sw += dc.T @ dc
        if len(zc) >= 2:
            sb += float(len(zc)) * (mc.T @ mc)

    sw /= float(n_samples)
    sb /= float(n_samples)
    lam = float(np.clip(reg_lambda, 1e-4, 0.99))
    tr_scale = max(1e-6, float(np.trace(sw)) / float(r_keep))
    reg_sw = (1.0 - lam) * sw + lam * tr_scale * np.eye(r_keep, dtype=np.float64)

    wv, wu = np.linalg.eigh(reg_sw)
    whiten_exp = -0.5 * float(np.clip(wccn_alpha, 0.0, 1.0))
    inv_wccn = wu @ np.diag(np.maximum(wv, 1e-8) ** whiten_exp) @ wu.T
    mat_t = inv_wccn @ sb @ inv_wccn
    mat_t = 0.5 * (mat_t + mat_t.T)
    _, bu = np.linalg.eigh(mat_t)
    top_dirs = bu[:, -k_keep:][:, ::-1]
    w_disc = basis @ (inv_wccn @ top_dirs)

    # Align discriminative frame to raw PCA basis via orthogonal Procrustes for coherent blending
    z_disc_val = _l2_normalize_rows(x_c @ w_disc)
    z_pca_val = _l2_normalize_rows(x_c @ pca_basis)
    u_p, _, vt_p = np.linalg.svd(z_disc_val.T @ z_pca_val, full_matrices=False)
    w_aligned = w_disc @ (u_p @ vt_p)
    w_aligned = _canonicalize_columns(w_aligned)

    return SubspaceProjector(
        mean_vec=mu.squeeze(0).astype(np.float32),
        proj_matrix=w_aligned.astype(np.float32),
        mix_alpha=float(mix_alpha),
        raw_basis=pca_basis.astype(np.float32),
    )


def fit_multitier_subspace(
    embeddings_by_tier: Mapping[int, np.ndarray],
    entity_ids: Sequence[Any],
    out_dim: int = 64,
    reg_lambda: float = 0.18,
    wccn_alpha: float = 0.65,
    mix_alpha: float = 0.85,
) -> Dict[int, SubspaceProjector]:
    """Fits regularized WCCN + Fisher subspace projectors across visual token tiers."""
    projectors: Dict[int, SubspaceProjector] = {}
    for tier_key in sorted(embeddings_by_tier.keys()):
        projectors[int(tier_key)] = _fit_single_tier_projector(
            x_raw=embeddings_by_tier[tier_key],
            entity_ids=entity_ids,
            out_dim=out_dim,
            reg_lambda=reg_lambda,
            wccn_alpha=wccn_alpha,
            mix_alpha=mix_alpha,
        )
    return projectors


@dataclass
class PredicateCalibrator:
    """Affine log-odds calibrator for zero-shot VLM predicate probabilities."""
    scale: Dict[str, float] = field(default_factory=dict)
    bias: Dict[str, float] = field(default_factory=dict)

    def calibrate(self, pred: str, raw_prob: float) -> float:
        p = min(1.0 - 1e-5, max(1e-5, float(raw_prob)))
        logit = math.log(p / (1.0 - p))
        a = float(self.scale.get(pred, 1.0))
        b = float(self.bias.get(pred, 0.0))
        z = max(-30.0, min(30.0, a * logit + b))
        return 1.0 / (1.0 + math.exp(-z))

    def calibrate_dict(self, raw_scores: Mapping[str, float]) -> Dict[str, float]:
        return {k: round(self.calibrate(k, v), 5) for k, v in raw_scores.items()}


def fit_predicate_calibration(
    raw_scores_by_pred: Mapping[str, Sequence[float]],
    labels_by_pred: Mapping[str, Sequence[int]],
    target_threshold: float = 0.50,
) -> PredicateCalibrator:
    """Fits validation log-odds scaling and F1-optimal threshold centering per predicate."""
    t_prob = min(1.0 - 1e-5, max(1e-5, float(target_threshold)))
    target_logit = math.log(t_prob / (1.0 - t_prob))
    scales: Dict[str, float] = {}
    biases: Dict[str, float] = {}

    for pred in sorted(raw_scores_by_pred.keys()):
        probs = np.clip(np.asarray(raw_scores_by_pred[pred], dtype=np.float64), 1e-5, 1.0 - 1e-5)
        labels = (np.asarray(labels_by_pred[pred], dtype=np.int32) > 0).astype(np.int32)
        if probs.size != labels.size or int(np.sum(labels == 1)) < 2 or int(np.sum(labels == 0)) < 2:
            scales[pred] = 1.0
            biases[pred] = 0.0
            continue

        logits = np.log(probs / (1.0 - probs))
        uniq = np.unique(logits)
        if uniq.size <= 1:
            scales[pred] = 1.0
            biases[pred] = 0.0
            continue

        pos_l = logits[labels == 1]
        neg_l = logits[labels == 0]
        pooled_std = math.sqrt(0.5 * (float(np.var(pos_l)) + float(np.var(neg_l))) + 0.05 ** 2)
        slope = float(np.clip(max(0.25, float(pos_l.mean() - neg_l.mean())) / max(0.35, 2.0 * pooled_std), 0.75, 3.0))

        mids = 0.5 * (uniq[:-1] + uniq[1:])
        cands = np.quantile(mids, np.linspace(0.02, 0.98, 96)) if mids.size > 96 else mids
        f1s = np.empty(cands.size, dtype=np.float64)
        for idx, thr in enumerate(cands):
            pred_pos = logits >= thr
            tp = int(np.sum(pred_pos & (labels == 1)))
            fp = int(np.sum(pred_pos & (labels == 0)))
            fn = int(np.sum((~pred_pos) & (labels == 1)))
            f1s[idx] = (2.0 * tp) / max(1.0, float(2 * tp + fp + fn))
        best_thr = float(np.median(cands[f1s >= (float(np.max(f1s)) - 1e-6)]))
        scales[pred] = round(slope, 5)
        biases[pred] = round(target_logit - slope * best_thr, 5)

    return PredicateCalibrator(scale=scales, bias=biases)
