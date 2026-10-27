import numpy as np

# Helpers for missing time points: rows of X (m, n, p) that are NaN in every feature.
# mask (m, n): True at a missing time point.

_imputeTS = None


# imputeTS na_kalman on one series, falling back to na_interpolation if it fails
def kalman_impute(v):
    global _imputeTS
    from rpy2.robjects.vectors import FloatVector
    if _imputeTS is None:
        from rpy2.robjects.packages import importr
        _imputeTS = importr('imputeTS')
    r_vec = FloatVector(v)
    try:
        return np.array(_imputeTS.na_kalman(r_vec), dtype=float)
    except Exception as e:
        print(f"Kalman method failed: {e}")
        print("Falling back to interpolation method...")
        return np.array(_imputeTS.na_interpolation(r_vec), dtype=float)


# Observed time points of each subject run together and end-padded with zeros
# Output:
#   [compacted array (m, max number of observed time points, k), per-step weights
#    (0 on the padding; None if every subject has the same number of observed time points)]
def compact(Z, mask):
    L = (~mask).sum(axis=1)
    m, k = Z.shape[0], Z.shape[2]
    out, w = np.zeros((m, L.max(), k)), np.zeros((m, L.max()))
    for i in range(m):
        out[i, :L[i]] = Z[i, ~mask[i]]
        w[i, :L[i]] = 1
    return out, (None if (L == L.max()).all() else w)


# Inverse of compact: observed time points back in place, NaN at the missing ones
def to_full_grid(Z, mask):
    m, n = mask.shape
    out = np.full((m, n, Z.shape[2]), np.nan)
    for i in range(m):
        obs = ~mask[i]
        out[i, obs] = Z[i, :obs.sum()]
    return out


# Missing time points after each subject's last observed one
def tail_mask(mask):
    tail = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        tail[i, np.flatnonzero(~mask[i])[-1] + 1:] = True
    return tail


# Kalman imputation (per subject and column) of the missing time points up to each subject's
# last observed one; the time points after it are set to 0
def fill_between(Z, mask):
    out = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
        last = np.flatnonzero(~mask[i])[-1] + 1
        Zi = Z[i, :last].astype(float).copy()
        Zi[mask[i, :last]] = np.nan
        if mask[i, :last].any():
            for j in range(Z.shape[2]):
                Zi[:, j] = kalman_impute(Zi[:, j])
        out[i, :last] = Zi
    return out
