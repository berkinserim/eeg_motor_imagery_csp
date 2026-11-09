import json
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

with open("best_params.json") as f:
    m = json.load(f)["lda"]["m"]


def load(name):
    d = np.load(f"data_{name}.npz")
    return d["X"].astype(np.float64), d["y"]


Xtr, ytr = load("train")
clf = make_pipeline(CSP(n_components=2 * m, log=True), LinearDiscriminantAnalysis())
clf.fit(Xtr, ytr)
for name, label in [("test", "MI"), ("exec", "ME")]:
    X, y = load(name)
    print(f"MNE CSP + sklearn LDA ({label}): {100 * clf.score(X, y):.2f}%")