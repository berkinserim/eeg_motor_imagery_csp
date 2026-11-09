import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

FS = 160
BAND = (8.0, 30.0)
FILTER_ORDER = 5
CHANNELS = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
TMIN, TMAX = 0.5, 2.5            # seconds after cue onset
IMAGERY_RUNS = [4, 8, 12]
EXECUTION_RUNS = [3, 7, 11]
SEED = 2601
M_VALUES = list(range(1, 11))    # 2m <= 21 channels
K_VALUES = list(range(1, 100, 2))
N_FOLDS = 5


def bandpass(data, fs=FS):
    sos = butter(FILTER_ORDER, BAND, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def subject_kfold(subjects, n_splits=N_FOLDS, seed=SEED):
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = []
    for group in np.array_split(uniq, n_splits):
        va = np.isin(subjects, group)
        folds.append((np.where(~va)[0], np.where(va)[0]))
    return folds


def trial_covariances(X):
    X = X - X.mean(axis=2, keepdims=True)
    C = np.einsum("ict,idt->icd", X, X) / X.shape[2]
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]   # trace normalization


def csp_filters(X, y):
    C = trial_covariances(X)
    R0, R1 = C[y == 0].mean(axis=0), C[y == 1].mean(axis=0)
    evals, evecs = eigh(R0, R0 + R1)
    return evecs[:, np.argsort(evals)[::-1]]


def component_variances(X, W):
    return np.einsum("cf,ict->ift", W, X).var(axis=2)


def features_from_variances(v, m):
    sel = np.concatenate([v[:, :m], v[:, -m:]], axis=1)
    return np.log(sel / sel.sum(axis=1, keepdims=True))


def knn_predict_multi(f_tr, y_tr, f_te, k_values):
    d = (np.sum(f_te**2, axis=1)[:, None] + np.sum(f_tr**2, axis=1)[None, :]
         - 2 * f_te @ f_tr.T)
    nn = y_tr[np.argsort(d, axis=1)[:, :max(k_values)]]
    return {k: (nn[:, :k].mean(axis=1) > 0.5).astype(int) for k in k_values}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def train_lr(F, y, lr=0.1, iters=2000, lam=0.0):
    w, b, n = np.zeros(F.shape[1]), 0.0, len(y)
    for _ in range(iters):
        g = sigmoid(F @ w + b) - y
        w -= lr * (F.T @ g + lam * w) / n
        b -= lr * g.mean()
    return w, b


def predict_lr(F, w, b):
    return (sigmoid(F @ w + b) >= 0.5).astype(int)


def train_lda(F, y, alpha=0.1):
    mu0, mu1 = F[y == 0].mean(axis=0), F[y == 1].mean(axis=0)
    S = (np.cov(F[y == 0].T) + np.cov(F[y == 1].T)) / 2
    S = (1 - alpha) * S + alpha * np.eye(F.shape[1])
    w = np.linalg.solve(S, mu1 - mu0)
    b = -0.5 * (mu0 + mu1) @ w
    return w, b


def predict_lda(F, w, b):
    return (F @ w + b >= 0).astype(int)