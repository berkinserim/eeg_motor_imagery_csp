# Cross-Subject Motor Imagery Decoding with CSP

Classifying left- vs. right-hand motor imagery from EEG in subjects the model has never seen. CSP, k-NN, logistic regression and LDA are implemented from scratch with NumPy/SciPy. Course project for EEE 485 (Statistical Learning and Data Analytics), Bilkent University.

## Results

Trained on the imagery trials of 85 subjects, tested on 21 unseen subjects (945 trials per condition).

| Model | CV (subject-level) | Test: imagery | Test: execution |
|---|---|---|---|
| k-NN (m=3, k=45) | 61.18% | 65.50% | 63.17% |
| Logistic regression (m=2) | 61.94% | 68.15% | 63.81% |
| LDA (m=2) | 62.07% | 68.36% | 64.44% |
| MNE CSP + scikit-learn LDA (reference) | - | 63.92% | 63.07% |

Accuracy varies strongly between subjects, from chance level to above 95% (see `per_subject_accuracy.png` and the report).

## Pipeline

- Data: PhysioNet EEG Motor Movement/Imagery Dataset, runs 4/8/12 (imagery) and 3/7/11 (execution). Subjects 88, 92 and 100 excluded (not sampled at 160 Hz).
- 21 sensorimotor channels (FC5 to CP6), 8-30 Hz zero-phase Butterworth filter, 0.5-2.5 s after cue onset.
- CSP with per-trial trace normalization and normalized log-variance features.
- Hyperparameters chosen with 5-fold subject-level cross-validation; CSP is refit in every fold to avoid leakage.

## How to run

```
pip install -r requirements.txt
python 01_preprocess.py        # downloads the data via MNE on the first run
python 02_cross_validation.py
python 03_test.py
python 04_mne_baseline.py
```

## Files

- `utilities.py`: filtering, subject-level folds, CSP, classifiers
- `report.pdf`: full project report

## Data

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043.

Pollard, T. et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
