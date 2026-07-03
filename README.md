# Multimodal Cancer Classification Challenge 2026 — Improvement Plan

**Goal:** push leaderboard AUC from **~0.82 → past 0.85**.
**Metric:** ROC-AUC. **Source of truth = local patient-grouped OOF AUC**, *not* the public
leaderboard (public = noisy subset of the test set; private = full test set decides ranking).
**Submission:** `Name,Diagnosis` (probability in [0,1]); **max 4 submissions/day**.

> Status note: no pretrained SSL encoder `.pth` files exist on this machine, so the existing
> 0.82 encoders are not available locally. See **"SSL encoder decision"** below — it gates the
> fast-track path.

---

## Why 0.82 is the ceiling right now (diagnosis)

- **Patient-level weak labels + only ~19 patients.** Every cell of a cancer patient is labelled 1
  even if it looks benign (Malignancy-Associated Changes). The effective sample size for the
  *signal* is ~patients, not ~cells. The dominant failure mode is the model learning
  **slide/stain/patient batch effects** instead of cancer, which does not transfer to unseen
  test patients → this is what caps generalization.
- **Two concrete bugs in the current code** (below) blunt the cross-modal signal and pick the
  wrong checkpoints. Fixing them is cheap and should be done first.

---

## Prioritized plan

### P0 — Correctness fixes (cheap, do first)

1. **CV redesign — `StratifiedGroupKFold(5)` grouped by `patient_id`, select best epoch by val AUC.**
   - Current code uses `LeaveOneGroupOut`. Because labels are patient-level, every held-out
     patient's validation fold is **single-class**, so per-fold `roc_auc_score` is *undefined* —
     that's why checkpoints are currently selected on **val loss** (a poor proxy for AUC).
   - Fix: one fixed `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on
     `patient_id`, stratified by `Diagnosis`. Each fold then contains both classes → **select the
     best epoch per fold by validation AUC**, and pool out-of-fold predictions for a global OOF AUC.
   - Bonus: 5 models instead of ~14, so ~3× faster, and a cleaner ensemble.

2. **Synchronize geometric augmentation across the BF/FL pair + per-modality normalization.**
   - Current `MultiModalCancerDataset` calls `transform` *independently* on BF and FL, so random
     flips/rotations/affine differ between the two — this **breaks the spatial correspondence the
     cross-modal attention fusion relies on**. Apply the *same* geometric params to both
     (e.g. `torchvision.transforms.v2` called on the pair, or shared functional params). Photometric
     jitter may stay per-modality.
   - Replace ImageNet `mean/std` with **per-modality (BF and FL) stats computed once from the
     training images and cached** — BF (brightfield) and FL (fluorescence) have very different
     intensity distributions.

3. **Reproducibility — set all seeds** (`torch`, `numpy`, CUDA, `cudnn.deterministic` where feasible)
   so variant comparisons are apples-to-apples.

### P1 — Generalization (the real lever from 0.82 → 0.85)

4. **Fine-tuning strategy to protect features on ~19 patients.**
   - Discriminative LR: low LR on the (pretrained) backbone, higher on fusion + classifier head;
     or a brief frozen-backbone warmup (1 epoch) then unfreeze. Avoids washing out transferred
     features and overfitting the tiny patient set.
   - Stronger stain/color augmentation to fight slide/batch overfitting; keep mixup + label
     smoothing (good regularizers against the weak-label noise).

5. **Ensemble diversity + rank-averaging.** Train diverse classifier variants over the *same* folds:
   `attention-fusion`, `simple-concat`, `BF-only`, `FL-only` (and optionally discriminative-LR).
   **Rank-average** their OOF predictions (AUC is rank-based) and keep the ensemble only if its OOF
   AUC beats the best single model. Simpler fusions usually overfit less on small data.

### P2 — Capacity / representation (only if time allows)

6. **Stronger SSL.** Re-pretrain longer and add an **intra-modal SimCLR term** (augment-two-views
   within BF and within FL) alongside the cross-modal NT-Xent. Plentiful unlabeled test images
   (~59k) make better representations valuable under weak labels. (Costs ~30–60 min GPU.)

---

## SSL encoder decision (gates the fast-track path)

The classifier's backbone is initialized from `BFpretrained_*.pth` / `FLpretrained_*.pth`. Those
files are **not present locally**, and the current code falls back to **random init** when they're
missing (bad). Options, fastest first:

- **(A) ImageNet init, skip SSL** — set the classifier backbones to `resnet18(weights=IMAGENET1K_V1)`.
  ~0 extra time; reasonable baseline; loses the cross-modal SSL benefit.
- **(B) Re-run SSL** (`pretrain_ssl.py`) to regenerate encoders. ~30–60 min GPU; preserves the
  original approach. Tight against a same-day deadline.
- **(C) Short SSL** (≈5–8 epochs) as a compromise (~10–20 min).

**Recommendation for the fast-track (~1–2 h, deadline today): start with (A)** to get a corrected,
submittable pipeline fast, then run (C)/(B) only if time remains.

---

## Fast-track execution order (proposed)

1. Build the shared CV harness with P0 fixes baked in (synced aug, per-modality norm, AUC-based
   selection on the fixed 5-fold split, seeds, TTA).
2. Smoke test (1 epoch, 1 fold, subset).
3. Train 1–2 strong variants (e.g. `simple-concat` and `attention-fusion`) with ImageNet init.
4. Rank-average ensemble; rank everything by OOF AUC in `results/leaderboard.md`.
5. Generate top-3 submission CSVs; **you pick** which 1–2 to submit (4/day cap).

## Caveats to keep in mind
- Trust **local OOF AUC**, never tune toward the public board.
- **Test filenames are anonymized** (`image_N.jpg`) → no patient grouping at test time; predict
  per cell only.
- Classes are **imbalanced**; AUC is threshold-free so calibration doesn't matter, ranking does.
