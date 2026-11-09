import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
from utilities import (csp_filters, component_variances, features_from_variances,
                       knn_predict_multi, train_lr, predict_lr, train_lda, predict_lda)

with open("best_params.json") as f:
    best = json.load(f)


def load(name):
    d = np.load(f"data_{name}.npz")
    return d["X"].astype(np.float64), d["y"], d["subject"]


Xtr, ytr, _ = load("train")
sets = {"MI": load("test"), "ME": load("exec")}

W = csp_filters(Xtr, ytr)                               # final CSP on all training subjects
v_tr = component_variances(Xtr, W)
v_te = {c: component_variances(sets[c][0], W) for c in sets}


def fit_predict(model, f_tr, f_te):
    if model == "knn":
        k = best["knn"]["k"]
        return knn_predict_multi(f_tr, ytr, f_te, [k])[k]
    if model == "lr":
        return predict_lr(f_te, *train_lr(f_tr, ytr))
    return predict_lda(f_te, *train_lda(f_tr, ytr))


results = {}
for model in ["knn", "lr", "lda"]:
    m = best[model]["m"]
    f_tr = features_from_variances(v_tr, m)
    results[model] = {"m": m}
    for c, (_, y, subj) in sets.items():
        pred = fit_predict(model, f_tr, features_from_variances(v_te[c], m))
        n, k = len(y), int(np.sum(pred == y))
        per_subj = {int(s): float(np.mean(pred[subj == s] == y[subj == s]))
                    for s in np.unique(subj)}
        accs = np.array(list(per_subj.values()))
        results[model][c] = {
            "accuracy": k / n, "correct": k, "n": n,
            "z": (k - n / 2) / np.sqrt(n / 4),
            "p_binomial_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue),
            "subject_mean": float(accs.mean()), "subject_sd": float(accs.std(ddof=1)),
            "subject_min": float(accs.min()), "subject_max": float(accs.max()),
            "per_subject": per_subj,
        }
    mi_s, me_s = results[model]["MI"]["per_subject"], results[model]["ME"]["per_subject"]
    common = sorted(set(mi_s) & set(me_s))
    diff = np.array([me_s[s] - mi_s[s] for s in common])
    stat, p = wilcoxon(diff)
    results[model]["ME_vs_MI_paired"] = {"mean_diff": float(diff.mean()),
                                         "wilcoxon_stat": float(stat), "p": float(p),
                                         "n_subjects": len(common)}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"{'Model':6s} {'Cond':4s} {'Acc':>7s} {'k/n':>10s} {'Z':>6s} {'p':>9s} {'Subj mean±sd':>14s}")
for model in ["knn", "lr", "lda"]:
    for c in ["MI", "ME"]:
        r = results[model][c]
        print(f"{model:6s} {c:4s} {100 * r['accuracy']:6.2f}% {r['correct']:4d}/{r['n']:<5d} "
              f"{r['z']:6.2f} {r['p_binomial_one_sided']:9.2e} "
              f"{100 * r['subject_mean']:5.1f}±{100 * r['subject_sd']:4.1f}")
    pm = results[model]["ME_vs_MI_paired"]
    print(f"{model:6s} ME-MI paired: diff {100 * pm['mean_diff']:+.2f} pts, "
          f"Wilcoxon p = {pm['p']:.3f} (n = {pm['n_subjects']})")
    print(f"{model:6s} CV acc at chosen params: {100 * best[model]['cv_acc']:.2f}%\n")

# Figure: per-subject accuracy
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharey=True)
for ax, model in zip(axes, ["knn", "lr", "lda"]):
    mi_s, me_s = results[model]["MI"]["per_subject"], results[model]["ME"]["per_subject"]
    subs = sorted(mi_s)
    x = np.arange(len(subs))
    ax.scatter(x, [mi_s[s] for s in subs], label="Imagery (MI)", zorder=3)
    ax.scatter(x, [me_s.get(s, np.nan) for s in subs], marker="s", label="Execution (ME)", zorder=3)
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=90)
    ax.set_title(model.upper())
    ax.set_xlabel("Test subject")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Accuracy")
axes[0].legend()
plt.tight_layout()
plt.savefig("per_subject_accuracy.png", dpi=300)
plt.close()