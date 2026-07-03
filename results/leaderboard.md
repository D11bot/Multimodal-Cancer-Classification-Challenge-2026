# Results — local OOF AUC (source of truth) vs public LB

Shared CV: `StratifiedGroupKFold(5)` by patient, seed=42. OOF = predictions pooled across folds.
**Caveat:** only 5 cancer patients → per-fold AUC swings hard (trust pooled OOF). And OOF is
**optimistic vs LB** — it holds out TRAIN patients, while the test set is *different* patients.

| Rank | Model | Global OOF AUC | Per-fold mean±std | Public LB | File |
|------|-------|---------------:|-------------------|----------:|------|
| 1 | **V3+V6 rank-avg ensemble** | **0.8924** | — | ? | `submissions/ensemble_v3_concat_v6_concat_ssl_rankavg.csv` |
| 2 | V6 = concat + SSL + strong aug + disc-LR | 0.8825 | 0.894 ± 0.147 | ? | `submissions/v6_concat_ssl_oof0.8825.csv` |
| 3 | V3 = concat, ImageNet init | 0.8823 | 0.904 ± 0.103 | **0.7694** | `submissions/v3_concat_oof0.8823.csv` (submitted) |

Per-fold AUC (held-out cancer patient in parens):
- **V3:** 0.988(17) · 0.986(16) · 0.717(18) · 0.874(03) · 0.956(05)
- **V6:** 0.992(17) · 0.989(16) · 0.605(18) · 0.919(03) · 0.965(05)

Notes:
- V6 dropped **mixup** (caused a Blackwell `illegal memory access` crash); kept SSL init +
  strong aug + discriminative LR.
- V3↔V6 prediction correlation = **0.82** → enough diversity that the rank-average lifts OOF
  +0.01 over either single model.
- V3's OOF→LB gap was ~0.11 (overfitting to the 12 train patients). The ensemble + V6's
  SSL-on-test adaptation target that gap, but OOF can't measure test-patient generalization,
  so the LB is the real test.
