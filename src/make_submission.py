"""STEP E driver: run TTA inference for an already-trained variant and write a submission CSV.
Does NOT train. Requires the variant's fold checkpoints in ./checkpoints/ (from run_variant.py)
and its ./results/<name>_result.json (for the OOF AUC used in the filename).
Usage:  py -u src/make_submission.py <v1|v3|v4|v5>
Note: TTA uses the harness's 4 modes (identity/hflip/vflip/rot90). OOF (the decision metric) does
NOT use TTA -- this only builds the test-set submission.
"""
import sys, os, glob, json
import pandas as pd
import cv_harness as H

VARIANTS = {
    "v1": ("attn",    "v1_attn"),
    "v3": ("concat",  "v3_concat"),
    "v4": ("bf_only", "v4_bf_only"),
    "v5": ("fl_only", "v5_fl_only"),
    "v6": ("concat",  "v6_concat_ssl"),
    "v7": ("bf_only", "v7_bf_ssl"),
    "v8": ("fl_only", "v8_fl_ssl"),
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "v3"
    if key not in VARIANTS:
        raise SystemExit(f"unknown variant '{key}'; choose {list(VARIANTS)}")
    variant, name = VARIANTS[key]

    H.set_seed(42)
    train_df = pd.read_csv(os.path.join(H.DATA_DIR, "train.csv"))
    norms = H.get_norms(train_df["Name"].tolist())
    test_df = pd.DataFrame({"Name": sorted(f for f in os.listdir(H.BF_TEST)
                                           if not f.startswith("."))})

    paths = sorted(glob.glob(f"./checkpoints/{name}_fold*.pth"))
    if not paths:
        raise SystemExit(f"no checkpoints found for {name}; run run_variant.py {key} first")
    print(f"=== {name}: TTA inference over {len(paths)} folds x 4 modes x {len(test_df)} imgs ===",
          flush=True)

    config = dict(batch_size=128, variant=variant, pretrained=True, name=name)
    probs = H.predict_test_tta(paths, config, test_df, norms)

    rj = f"./results/{name}_result.json"
    oof = json.load(open(rj))["global_oof_auc"] if os.path.exists(rj) else 0.0
    out = f"./submissions/{name}_oof{oof:.4f}.csv"
    H.write_submission(test_df, probs, out)
    print(f"=== wrote {out} (OOF AUC {oof:.4f}) ===", flush=True)
