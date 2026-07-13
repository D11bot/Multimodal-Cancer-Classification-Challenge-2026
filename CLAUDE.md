# CLAUDE.md — Multimodal Cancer Classification Challenge 2026


Multimodal Cancer Classification Challenge 2026. This repo is my entry for a Kaggle-style challenge that classifies single cells as malignant or benign from paired brightfield and fluorescence microscopy images. My approach joins two ResNet-18 encoders through a cross-modal attention fusion head, with cross-modal contrastive (self-supervised) pretraining on the paired cells before supervised fine-tuning, and selects models with patient-grouped cross-validation using out-of-fold ROC-AUC as the source of truth, since the public leaderboard is only a noisy subset of the test set. The best single model reaches about 0.88 out-of-fold AUC, and a rank-averaged ensemble with test-time augmentation lifts it slightly further.

## Task & metric
- Binary classification of oral-cytology **cells** as cancer (1) / healthy (0) from **paired
  brightfield (BF) + fluorescence (FL)** microscopy images, **128×128**.
- **Metric: ROC-AUC.** Goal: push from **~0.82 → past 0.85**.
- **Public vs private leaderboard:** public = noisy subset shown at submit time; private = full
  test set, decides final rank. **SOURCE OF TRUTH = local patient-grouped OOF AUC**, never the
  public board. Do not tune toward public.
- **Submission:** CSV with header `Name,Diagnosis`, `Diagnosis` = probability in [0,1], one row
  per test image. **Max 4 submissions/day.** Deadline shown by Kaggle: **2026-06-03 21:59 (UTC)**.

## Data (verified)
- Path: `data/multimodal-cancer-classification-challenge-2026/` → `BF/{train,test}`,
  `FL/{train,test}`, `train.csv`, `sampleSubmission.csv`.
- **Train: 114,302 paired cells** from **12 patients**; **Test: 59,040 paired cells**.
- BF and FL train filenames are **identical / perfectly paired** (verified). All 114,302
  `train.csv` Names exist as image files.
- **Labels are patient-level WEAK labels:** every cell of a cancer patient is labelled 1 (MAC
  assumption), even benign-looking ones. Effective signal size ≈ #patients, not #cells.
- **Class balance (cell level):** 70,000 healthy / 44,302 cancer (neg/pos ≈ 1.58, mild imbalance).
- **Patients:** 12 train patients = **5 cancer** (`03,05,16,17,18`) + **7 healthy**
  (`07,09,10,11,13,14,15`); ~10,000 cells each (pat_05 = 4,302). (Competition says 19 total;
  ~7 are in the test set.)
- **Patient id parse:** `Name.split('_')[1]` → `pat_03_image_1.jpg` → `'03'`.
- **Test filenames are anonymized** (`image_N.jpg`) → **no patient grouping at test time**;
  predict per cell only.

### ⚠️ CV consequence (important)
Only **5 cancer + 7 healthy** patients. `LeaveOneGroupOut` (old code) makes every validation fold
**single-class** → per-fold AUC undefined → checkpoints were selected on val loss. Use a fixed
**`StratifiedGroupKFold(n_splits=5)` grouped by `patient_id`, stratified by `Diagnosis`** so each
fold has both classes and we can select the best epoch by **val AUC**. With only 5 cancer patients,
each fold's positive signal = ~1 held-out cancer patient → **high per-fold variance**; trust the
**pooled global OOF AUC**, and consider repeated splits (different seeds) to stabilize.

## Scripts
- `src/pretrain_ssl.py` — cross-modal contrastive SSL: two ResNet18 encoders + projection heads,
  NT-Xent loss over (BF,FL) pairs from train+test. Saves encoder weights to
  `checkpoints/BFpretrained_resnet18_encoder.pth` / `FLpretrained_resnet18_encoder.pth`.
- `src/train_infer.py` — dual-branch classifier: loads the two SSL encoders, cross-modal attention
  fusion → MLP head; CV + mixup + label smoothing + TTA; writes `submissions/submission.csv`.
- **Run scripts from the project root** `C:\ml\mcc2026` (paths are root-relative).
- **No SSL encoder `.pth` exists locally** → `train_infer.py` would currently fall back to random
  init. Fast-track plan: initialize backbones from ImageNet (skip SSL) first; re-run SSL only if
  time allows. See `README.md`.

## Environment (Windows 11, RTX 5070 Ti Laptop 12GB — Blackwell sm_120)
- Use the **`py`** launcher → Python **3.14.0** (`python`/`python3` are Microsoft Store stubs).
- **torch 2.11.0+cu128, torchvision 0.26.0+cu128** — Blackwell REQUIRES the **cu128** build
  (`--index-url https://download.pytorch.org/whl/cu128`); older CUDA wheels crash at runtime.
- Blackwell gotchas (verified clear here): a passing `cuda.is_available()`/matmul does NOT prove
  training works — always run a real **backward + optimizer.step** smoke test. **Single GPU →
  never use DDP/`torch.distributed`** (breaks on Windows+Blackwell).
- `kaggle.exe` (v2.2.0) lives at
  `C:\Users\29485\AppData\Local\Programs\Python\Python314\Scripts\kaggle.exe` (NOT on PATH).
- Kaggle token: `%USERPROFILE%\.kaggle\access_token` (verified working). **Never open/print it.**
  Never submit without explicit per-time confirmation (4/day cap).

## Improvement plan
See **`README.md`** for the full prioritized plan. Summary:
- **P0:** StratifiedGroupKFold(5) + AUC-based checkpointing; synchronize BF/FL geometric aug +
  per-modality normalization; set seeds.
- **P1:** discriminative LR / frozen warmup; stronger stain aug; ensemble diversity
  (attention / concat / BF-only / FL-only) with rank-averaging.
- **P2:** stronger/longer SSL with an intra-modal SimCLR term.

## Hard rules
- Never create/read/print the Kaggle credential; reference by path only.
- Never run `kaggle competitions submit` without explicit in-the-moment confirmation.
- Never commit credentials or data (see `.gitignore`).
