"""Rank-average ensemble of saved per-model TEST prob arrays -> one submission CSV.
CPU only (numpy/scipy/pandas, NO torch) so it's safe to run alongside GPU jobs.
Run from project root:  py src/ensemble_submission.py [name1 name2 ...]
Defaults to v3_concat + v6_concat_ssl. Reads ./results/<name>_test_probs.npy (saved by
make_submission.py / predict_test_tta), aligned to sorted(BF/test) order.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA = "./data/multimodal-cancer-classification-challenge-2026"
BF_TEST = os.path.join(DATA, "BF", "test")

members = sys.argv[1:] if len(sys.argv) > 1 else ["v3_concat", "v6_concat_ssl"]
probs = {}
for name in members:
    p = f"./results/{name}_test_probs.npy"
    if os.path.exists(p):
        probs[name] = np.load(p)
    else:
        print(f"  WARN: {p} missing -> skipping {name}")
if not probs:
    raise SystemExit("no member test-prob arrays found")

test_names = sorted(f for f in os.listdir(BF_TEST) if not f.startswith("."))
for name, p in probs.items():
    assert len(p) == len(test_names), f"{name}: {len(p)} != {len(test_names)} test images"

# rank-average (AUC is rank-based; robust to per-model scale differences)
ranks = np.vstack([rankdata(p) for p in probs.values()]).mean(0)
ranks = ranks / ranks.max()  # scale to [0,1]; AUC-invariant

tag = "_".join(probs.keys())
out = f"./submissions/ensemble_{tag}_rankavg.csv"
pd.DataFrame({"Name": test_names, "Diagnosis": ranks}).to_csv(out, index=False)
print(f"members: {list(probs.keys())}")
print(f"wrote {out} | rows={len(test_names)} range=[{ranks.min():.4f}, {ranks.max():.4f}] "
      f"mean={ranks.mean():.4f}")
