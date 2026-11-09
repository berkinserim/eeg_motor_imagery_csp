import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter
from utilities import (M_VALUES, K_VALUES, N_FOLDS, SEED, subject_kfold, csp_filters,
                       component_variances, features_from_variances, knn_predict_multi,
                       train_lr, predict_lr, train_lda, predict_lda)

d = np.load("data_train.npz")
X, y, subj = d["X"].astype(np.float64), d["y"], d["subject"]
folds = subject_kfold(subj, N_FOLDS, SEED)

knn = np.zeros((N_FOLDS, len(M_VALUES), len(K_VALUES)))
lr = np.zeros((N_FOLDS, len(M_VALUES)))
lda = np.zeros((N_FOLDS, len(M_VALUES)))

for f, (tr, va) in enumerate(folds):
    W = csp_filters(X[tr], y[tr])                       # CSP fit on training folds only
    v_tr, v_va = component_variances(X[tr], W), component_variances(X[va], W)
    for i, m in enumerate(M_VALUES):
        f_tr, f_va = features_from_variances(v_tr, m), features_from_variances(v_va, m)
        preds = knn_predict_multi(f_tr, y[tr], f_va, K_VALUES)
        knn[f, i] = [np.mean(preds[k] == y[va]) for k in K_VALUES]
        lr[f, i] = np.mean(predict_lr(f_va, *train_lr(f_tr, y[tr])) == y[va])
        lda[f, i] = np.mean(predict_lda(f_va, *train_lda(f_tr, y[tr])) == y[va])
    print(f"Fold {f + 1}/{N_FOLDS} done ({len(np.unique(subj[va]))} validation subjects)")

knn_mean, lr_mean, lda_mean = knn.mean(axis=0), lr.mean(axis=0), lda.mean(axis=0)

smooth = uniform_filter(knn_mean, size=3, mode="nearest")
mi, ki = np.unravel_index(np.argmax(smooth), smooth.shape)


def smallest_within(scores, tol):
    return int(np.where(scores >= scores.max() - tol)[0].min())


li, di = smallest_within(lr_mean, 0.005), smallest_within(lda_mean, 0.01)

best = {
    "knn": {"m": M_VALUES[mi], "k": K_VALUES[ki],
            "cv_acc": float(knn_mean[mi, ki]), "cv_max": float(knn_mean.max())},
    "lr": {"m": M_VALUES[li], "cv_acc": float(lr_mean[li]), "cv_max": float(lr_mean.max())},
    "lda": {"m": M_VALUES[di], "cv_acc": float(lda_mean[di]), "cv_max": float(lda_mean.max())},
}
with open("best_params.json", "w") as f:
    json.dump(best, f, indent=2)
np.savez("cv_results.npz", knn=knn, lr=lr, lda=lda)
print(json.dumps(best, indent=2))

# Figure: k-NN heatmap
plt.figure(figsize=(16, 9))
x_labels = [str(k) if i % 2 == 0 else "" for i, k in enumerate(K_VALUES)]
ax = sns.heatmap(knn_mean, annot=False, fmt=".2f", cmap="magma",
                 xticklabels=x_labels, yticklabels=M_VALUES,
                 annot_kws={"size": 10, "fontweight": "bold"},
                 cbar_kws={"label": "Mean Accuracy"})
ax.invert_yaxis()
ax.add_patch(plt.Rectangle((ki, mi), 1, 1, fill=False, edgecolor="cyan", lw=5))
plt.title("k-NN CV (subject-level folds): m & k", fontsize=22, fontweight="bold", pad=20)
plt.xlabel("Number of Neighbors (k)", fontsize=16)
plt.ylabel("CSP Filter Pairs (m)", fontsize=16)
plt.tight_layout()
plt.savefig("hyperparameter_heatmap.png", dpi=300)
plt.close()

# Figures: LR and LDA curves
for name, scores, idx, tol, fname, color in [
        ("Logistic Regression", lr, li, 0.005, "log_reg_cv.png", "b"),
        ("LDA", lda, di, 0.01, "lda_cv.png", "forestgreen")]:
    mean, std = scores.mean(axis=0), scores.std(axis=0)
    plt.figure(figsize=(10, 5))
    plt.fill_between(M_VALUES, mean - std, mean + std, alpha=0.15, color=color)
    plt.plot(M_VALUES, mean, "-o", color=color, label="Mean validation accuracy")
    plt.axhline(mean.max() - tol, color="r", linestyle="--", label="Threshold")
    plt.axvline(M_VALUES[idx], color="k", label=f"Chosen m={M_VALUES[idx]}")
    plt.title(f"{name} CV (subject-level folds): Accuracy vs. m", fontsize=14)
    plt.xlabel("CSP Filter Pairs (m)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()