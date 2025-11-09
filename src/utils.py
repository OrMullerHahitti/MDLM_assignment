from typing import Any, Dict, List,Tuple
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
rng = np.random.default_rng(42)
def make_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Ensure Sigma is symmetric PSD. If tiny negative eigenvalues appear, add ֿ\epsilon I.
    Args:
        S (np.ndarray): The input matrix to be projected.
        eps (float, optional): A small value to ensure numerical stability. Defaults to 1e-8.
    Returns:
        np.ndarray: The projected positive semi-definite matrix.
    """
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def pretty_print_buffer():
    print("\n" + "-"*32 + "\n" )

def iter_grid(grid: Dict[str, List[Any]]):
    from itertools import product
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))
        
def stratified_kfold_indices(y: np.ndarray, k: int, rng) -> List[np.ndarray]:
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    folds0 = np.array_split(idx0, k)
    folds1 = np.array_split(idx1, k)

    return [np.concatenate([f0, f1]) for f0, f1 in zip(folds0, folds1)]

def manual_cv_accuracy(estimator, X, y, k: int = 10) -> Dict[str, Any]:
    folds = stratified_kfold_indices(y, k, rng)
    accs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=True)

        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], y_pred))

    accs = np.array(accs, dtype=float)
    return {
        "fold_acc": accs,
        "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std(ddof=1)),
        "gen_error": float(1.0 - accs.mean()),
    }

def make_mlp(h: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(h,),
                              activation="relu",
                              solver="adam",
                              alpha=1e-3,
                              max_iter=600,
                              tol=1e-4,
                              random_state=42))
    ])

def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsampling of the dataset.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n (int): Number of samples to draw.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Subsampled feature matrix and target vector.
    """
    n0 = int(round((y==0).mean() * n))
    n1 = n - n0
    idx0 = rng.choice(np.where(y==0)[0], size=n0, replace=False)
    idx1 = rng.choice(np.where(y==1)[0], size=n1, replace=False)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]