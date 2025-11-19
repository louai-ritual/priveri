"""
Verification utilities for matching prover logits against a cached set of
sentinel-block logits.

This module is adapted from the standalone `compare.py` prototype in the
`context/` folder, but kept lightweight and library-style.
"""

from typing import Tuple

import numpy as np


def load_array(path: str, dtype=np.float32) -> np.ndarray:
    """
    Load a vector array from .npy, .npz, or .pt (PyTorch).

    Supports the special .pt format produced by `TrapTokenAugmenter.save_cache`
    (with `augmentation_blocks[*].trap_block.logits`).

    Returns:
        A NumPy array of shape (N, D) in the requested dtype.
    """
    if path is None:
        raise ValueError("No path provided")

    import os

    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, mmap_mode="r")

    elif ext == ".pt":
        try:
            import torch
        except ImportError as e:  # pragma: no cover - depends on torch
            raise ImportError("PyTorch is required to read .pt files") from e
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict) and "augmentation_blocks" in obj:
            blocks = obj.get("augmentation_blocks", [])
            logits_list = []
            for block in blocks:
                tb = block.get("trap_block", {})
                if "logits" in tb:
                    logits_list.append(tb["logits"].flatten().detach().cpu().numpy())
            if not logits_list:
                raise ValueError(
                    f"No logits found in augmentation_blocks of {path}"
                )
            arr = np.array(logits_list)
        else:
            raise ValueError(f"Unsupported object type in {path}: {type(obj)}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def ensure_min_dim(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays: (N, D)")
    D = min(A.shape[1], B.shape[1])
    return A[:, :D], B[:, :D]


def safe_norms(x: np.ndarray, axis=1, keepdims=False) -> np.ndarray:
    s = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    s[s == 0] = 1.0
    return s


def find_best_with_numpy_chunked(
    A: np.ndarray,
    B: np.ndarray,
    metric: str = "euclidean",
    a_chunk: int = 8,
    b_chunk: int = 1024,
) -> Tuple[Tuple[int, int], float]:
    """
    Exact search for the best-matching pair between rows of A and B using NumPy.

    Args:
        A: (Na, D) array (e.g. cache vectors).
        B: (Nb, D) array (e.g. prover vectors).
        metric: 'euclidean', 'l1', or 'cosine'.
        a_chunk: size of outer chunks over A.
        b_chunk: size of inner chunks over B.

    Returns:
        (best_i, best_j), best_score
        where i indexes into A and j into B. For cosine, best_score is similarity
        in [-1,1] (larger is better). For Euclidean/L1, smaller is better.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    Na, D = A.shape
    Nb = B.shape[0]

    best_pair = (None, None)
    best_score = -np.inf if metric == "cosine" else np.inf

    outer = range(0, Na, a_chunk)

    for a_start in outer:
        a_end = min(Na, a_start + a_chunk)
        A_chunk = A[a_start:a_end]

        if metric == "euclidean":
            A_sq = np.sum(A_chunk * A_chunk, axis=1).reshape(-1, 1)
        elif metric == "cosine":
            norms = safe_norms(A_chunk, axis=1, keepdims=True)
            A_chunk_norm = A_chunk / norms

        for b_start in range(0, Nb, b_chunk):
            b_end = min(Nb, b_start + b_chunk)
            B_block = B[b_start:b_end]

            if metric == "euclidean":
                B_sq = np.sum(B_block * B_block, axis=1)
                dot = A_chunk.dot(B_block.T)
                dists = A_sq + B_sq.reshape(1, -1) - 2.0 * dot
                local_flat_idx = np.argmin(dists)
                m_idx, n_idx = divmod(int(local_flat_idx), dists.shape[1])
                local_val = float(dists[m_idx, n_idx])
                if local_val < best_score:
                    best_score = local_val
                    best_pair = (a_start + m_idx, b_start + n_idx)
            elif metric == "cosine":
                B_block_norm = B_block / safe_norms(B_block, axis=1, keepdims=True)
                dot = A_chunk_norm.dot(B_block_norm.T)
                local_flat_idx = np.argmax(dot)
                m_idx, n_idx = divmod(int(local_flat_idx), dot.shape[1])
                local_val = float(dot[m_idx, n_idx])
                if local_val > best_score:
                    best_score = local_val
                    best_pair = (a_start + m_idx, b_start + n_idx)
            elif metric == "l1":
                dists = np.sum(
                    np.abs(A_chunk[:, None, :] - B_block[None, :, :]), axis=2
                )
                local_flat_idx = np.argmin(dists)
                m_idx, n_idx = divmod(int(local_flat_idx), dists.shape[1])
                local_val = float(dists[m_idx, n_idx])
                if local_val < best_score:
                    best_score = local_val
                    best_pair = (a_start + m_idx, b_start + n_idx)
            else:
                raise ValueError("metric should be 'euclidean', 'l1', or 'cosine'")

    if metric == "euclidean" and best_score is not None:
        best_score = float(np.sqrt(max(best_score, 0.0)))

    return best_pair, best_score


def verify_with_cache(
    cache_path: str,
    candidate_path: str,
    metric: str = "cosine",
    threshold: float = 0.99,
) -> Tuple[bool, dict]:
    """
    Verify that at least one candidate vector matches some cached vector.

    The typical usage is:
      - cache_path: .pt file produced by `build_sentinel_cache`, containing many
        flattened sentinel-block logits.
      - candidate_path: .pt/.npy file with one or more flattened sentinel-block
        logits produced by the prover.

    Returns:
        (matched: bool, details: dict)
    """
    A = load_array(cache_path, dtype=np.float32)
    B = load_array(candidate_path, dtype=np.float32)
    A, B = ensure_min_dim(A, B)

    (ia, ib), score = find_best_with_numpy_chunked(
        A, B, metric=metric, a_chunk=8, b_chunk=1024
    )

    if metric == "cosine":
        matched = score >= threshold
    else:
        # For distance metrics, we require the distance to be <= threshold
        matched = score <= threshold

    details = {
        "cache_index": ia,
        "candidate_index": ib,
        "score": score,
        "metric": metric,
        "threshold": threshold,
    }
    return matched, details


