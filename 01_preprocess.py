import json
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from utilities import FS, CHANNELS, TMIN, TMAX, IMAGERY_RUNS, EXECUTION_RUNS, SEED, bandpass

mne.set_log_level("ERROR")
ALL_SUBJECTS = range(1, 110)
TEST_FRACTION = 0.2


def load_run(subject, run):
    path = eegbci.load_data(subject, [run], update_path=True)[0]
    raw = mne.io.read_raw_edf(path, preload=True)
    eegbci.standardize(raw)
    return raw


def extract_trials(raw):
    missing = [c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels {missing}. Available: {raw.ch_names}")
    data = raw.get_data(picks=CHANNELS) * 1e6            # V -> uV
    data = data - data.mean(axis=1, keepdims=True)
    filtered = bandpass(data)
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
    start, length = int(TMIN * FS), int((TMAX - TMIN) * FS)
    X, y = [], []
    for onset, _, code in events:
        s = onset - raw.first_samp + start
        if s + length <= filtered.shape[1]:
            X.append(filtered[:, s:s + length])
            y.append(0 if code == 1 else 1)              # T1 = left fist, T2 = right fist
    return X, y


def process_subject(subject, runs):
    X, y = [], []
    for run in runs:
        raw = load_run(subject, run)
        if raw.info["sfreq"] != FS:
            return None
        Xr, yr = extract_trials(raw)
        X += Xr
        y += yr
    if len(y) == 0:
        return None
    return np.array(X, dtype=np.float32), np.array(y)


def stack(results, ids):
    X = np.concatenate([results[s][0] for s in ids])
    y = np.concatenate([results[s][1] for s in ids])
    subj = np.concatenate([np.full(len(results[s][1]), s) for s in ids])
    return X, y, subj


imagery, excluded = {}, []
for s in ALL_SUBJECTS:
    out = process_subject(s, IMAGERY_RUNS)
    if out is None:
        excluded.append(s)
        print(f"Subject {s:3d}: excluded")
    else:
        imagery[s] = out
        print(f"Subject {s:3d}: {len(out[1])} imagery trials")

valid = np.array(sorted(imagery))
perm = np.random.default_rng(SEED).permutation(valid)
n_test = int(round(TEST_FRACTION * len(valid)))
test_ids = sorted(int(s) for s in perm[:n_test])
train_ids = sorted(int(s) for s in perm[n_test:])

execution = {}
for s in test_ids:
    out = process_subject(s, EXECUTION_RUNS)
    if out is not None:
        execution[s] = out

for name, res, ids in [("train", imagery, train_ids),
                       ("test", imagery, test_ids),
                       ("exec", execution, sorted(execution))]:
    X, y, subj = stack(res, ids)
    np.savez_compressed(f"data_{name}.npz", X=X, y=y, subject=subj)
    print(f"{name}: {X.shape[0]} trials, {len(ids)} subjects, "
          f"shape {X.shape}, right-hand ratio {y.mean():.3f}")

with open("split.json", "w") as f:
    json.dump({"train": train_ids, "test": test_ids, "excluded": excluded,
               "channels": CHANNELS, "window_s": [TMIN, TMAX]}, f, indent=2)
print(f"Excluded subjects: {excluded}")

# Figure 1: raw vs. filtered, same electrode (C3)
raw = load_run(1, IMAGERY_RUNS[0])
c3 = raw.get_data(picks=["C3"])[0] * 1e6
c3_filt = bandpass(c3 - c3.mean())
seg, t = slice(800, 800 + FS), np.arange(FS) / FS

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(t, c3[seg], color="firebrick", label="Raw signal")
ax[0].set_title("Graph A: Raw EEG Signal (C3) - Subject 1", fontsize=16, fontweight="bold")
ax[1].plot(t, c3_filt[seg], color="forestgreen", label="Filtered (8-30 Hz, zero-phase)")
ax[1].set_title("Graph B: Filtered EEG Signal (C3)", fontsize=16, fontweight="bold")
ax[1].set_xlabel("Time (seconds)", fontsize=12)
for a in ax:
    a.set_ylabel(r"Amplitude ($\mu V$)", fontsize=12)
    a.grid(True, alpha=0.3)
    a.legend(loc="upper right")
plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=300)
plt.close()