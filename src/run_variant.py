"""Driver: run ONE variant's 5-fold CV (CV-only, NO TTA) and report global OOF AUC.
Resilient: each fold's OOF is saved immediately and reused on resume, so an interruption
never loses completed folds. Uses the harness's train_one_fold + the IDENTICAL fixed split
(StratifiedGroupKFold(5, shuffle=True, random_state=42)) so OOF is comparable across variants.
TTA/submission is deferred to STEP E. Usage:  py -u src/run_variant.py <v1|v3|v4|v5> [epochs]
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import cv_harness as H

VARIANTS = {
    "v1": ("attn",    "v1_attn"),
    "v3": ("concat",  "v3_concat"),
    "v4": ("bf_only", "v4_bf_only"),
    "v5": ("fl_only", "v5_fl_only"),
    # SSL-initialized + hardened (strong aug + mixup + discriminative LR)
    "v6": ("concat",  "v6_concat_ssl"),
    "v7": ("bf_only", "v7_bf_ssl"),
    "v8": ("fl_only", "v8_fl_ssl"),
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "v3"
    if key not in VARIANTS:
        raise SystemExit(f"unknown variant '{key}'; choose {list(VARIANTS)}")
    variant, name = VARIANTS[key]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    H.set_seed(42)
    for d in ("./checkpoints", "./results", "./submissions"):
        os.makedirs(d, exist_ok=True)
    train_df = pd.read_csv(os.path.join(H.DATA_DIR, "train.csv"))
    train_df["patient_id"] = train_df["Name"].apply(lambda x: x.split("_")[1])
    norms = H.get_norms(train_df["Name"].tolist())

    SSL_BF = "./checkpoints/BFpretrained_resnet18_encoder.pth"
    SSL_FL = "./checkpoints/FLpretrained_resnet18_encoder.pth"
    config = dict(seed=42, batch_size=128, lr=1e-3, epochs=epochs,
                  n_per_patient=4000, use_mixup=False, pretrained=True,  # mixup off: Blackwell illegal-mem crash
                  backbone_lr_mult=0.1, ssl_bf=SSL_BF, ssl_fl=SSL_FL,
                  variant=variant, name=name)
    print(f"=== {name}: variant={variant}, epochs={epochs}, CV-only (no TTA) | "
          f"mixup={config['use_mixup']} ssl_init={os.path.exists(SSL_BF)} ===", flush=True)

    # SAME fixed split as run_cv -> OOF comparable across variants
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    splits = list(sgkf.split(train_df, train_df["Diagnosis"], groups=train_df["patient_id"]))

    oof = np.full(len(train_df), np.nan)
    fold_aucs = [np.nan] * len(splits)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_file = f"./results/{name}_fold{fold}_oof.npz"
        if os.path.exists(fold_file):
            d = np.load(fold_file)
            oof[d["va_idx"]] = d["preds"]; fold_aucs[fold] = float(d["auc"])
            print(f"[resume] fold{fold} loaded, AUC={fold_aucs[fold]:.4f}", flush=True)
            continue
        print(f"--- fold {fold} (val patients: "
              f"{sorted(train_df.iloc[va_idx]['patient_id'].unique())}) ---", flush=True)
        auc, preds, path = H.train_one_fold(train_df.iloc[tr_idx], train_df.iloc[va_idx],
                                            fold, config, norms)
        oof[va_idx] = preds; fold_aucs[fold] = auc
        np.savez(fold_file, preds=preds, va_idx=va_idx, auc=auc)
        print(f"[done] fold{fold} best val AUC={auc:.4f}", flush=True)

    mask = ~np.isnan(oof)
    g = roc_auc_score(train_df["Diagnosis"].values[mask], oof[mask])
    np.savez(f"./results/{name}_oof.npz", oof=oof, y=train_df["Diagnosis"].values)
    result = {"name": name, "variant": variant, "epochs": epochs,
              "global_oof_auc": float(g),
              "per_fold_aucs": [float(a) for a in fold_aucs],
              "per_fold_mean": float(np.nanmean(fold_aucs)),
              "per_fold_std": float(np.nanstd(fold_aucs))}
    with open(f"./results/{name}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {name} DONE ===", flush=True)
    print(f"GLOBAL OOF AUC = {g:.4f}")
    print(f"per-fold = {result['per_fold_mean']:.4f} +/- {result['per_fold_std']:.4f} "
          f"{[round(a, 4) for a in fold_aucs]}", flush=True)
