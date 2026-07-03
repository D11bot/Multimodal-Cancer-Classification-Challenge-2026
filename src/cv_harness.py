"""
src/cv_harness.py — reusable CV harness for multimodal-cancer-classification-challenge-2026.
StratifiedGroupKFold(5) by patient_id | best epoch by VAL AUC | synced BF/FL geometric aug +
per-modality normalization | optional per-patient subsample (resampled each epoch) | TTA |
OOF aligned to train_df row order for cross-variant rank-averaging | seeds set.
"""
import os, json, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DATA_DIR = "./data/multimodal-cancer-classification-challenge-2026"
BF_TRAIN, FL_TRAIN = os.path.join(DATA_DIR,"BF","train"), os.path.join(DATA_DIR,"FL","train")
BF_TEST,  FL_TEST  = os.path.join(DATA_DIR,"BF","test"),  os.path.join(DATA_DIR,"FL","test")

# ---------- per-modality normalization (computed once, cached) ----------
def _stats(names, img_dir, n=2000, seed=42):
    rng = random.Random(seed); sample = rng.sample(names, min(n, len(names)))
    ms, ss = [], []
    for nm in sample:
        x = TF.to_tensor(Image.open(os.path.join(img_dir, nm)).convert("RGB"))
        ms.append(x.mean(dim=(1,2))); ss.append(x.std(dim=(1,2)))
    return torch.stack(ms).mean(0).tolist(), torch.stack(ss).mean(0).tolist()

def get_norms(train_names, cache="./results/norm_stats.json"):
    if os.path.exists(cache):
        with open(cache) as f: return json.load(f)
    norms = {"bf": _stats(train_names, BF_TRAIN), "fl": _stats(train_names, FL_TRAIN)}
    os.makedirs("./results", exist_ok=True)
    with open(cache, "w") as f: json.dump(norms, f, indent=2)
    return norms

# ---------- transforms: geometry SYNCED across BF/FL; photometric per-modality ----------
class PairedTransform:
    def __init__(self, norms, train=True):
        self.train = train
        self.bf_n = transforms.Normalize(*norms["bf"]); self.fl_n = transforms.Normalize(*norms["fl"])
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)
        self.erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0)
    def __call__(self, bf, fl):
        if self.train:
            # ---- synced geometry: SAME params on BF & FL (keeps spatial alignment) ----
            if random.random() < 0.5: bf, fl = TF.hflip(bf), TF.hflip(fl)
            if random.random() < 0.5: bf, fl = TF.vflip(bf), TF.vflip(fl)
            ang = random.uniform(-45, 45)
            w, h = bf.size
            tx, ty = random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h
            sc, sh = random.uniform(0.9, 1.1), random.uniform(-10, 10)
            bf = TF.affine(bf, angle=ang, translate=[tx, ty], scale=sc, shear=[sh])
            fl = TF.affine(fl, angle=ang, translate=[tx, ty], scale=sc, shear=[sh])
            # ---- independent photometric (BF & FL look different) ----
            bf, fl = self.color(bf), self.color(fl)
        bf, fl = TF.to_tensor(bf), TF.to_tensor(fl)
        if self.train:
            bf, fl = self.erase(bf), self.erase(fl)
        return self.bf_n(bf), self.fl_n(fl)

def tta_apply(bf, fl, mode, norms):
    if   mode == "hflip": bf, fl = TF.hflip(bf), TF.hflip(fl)
    elif mode == "vflip": bf, fl = TF.vflip(bf), TF.vflip(fl)
    elif mode == "rot90": bf, fl = TF.rotate(bf, 90), TF.rotate(fl, 90)
    bf, fl = TF.to_tensor(bf), TF.to_tensor(fl)
    return transforms.Normalize(*norms["bf"])(bf), transforms.Normalize(*norms["fl"])(fl)

# ---------- datasets ----------
class PairedDataset(Dataset):
    def __init__(self, df, transform): self.df = df.reset_index(drop=True); self.t = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        nm = self.df.loc[i,"Name"]
        bf = Image.open(os.path.join(BF_TRAIN, nm)).convert("RGB")
        fl = Image.open(os.path.join(FL_TRAIN, nm)).convert("RGB")
        bf, fl = self.t(bf, fl)
        return bf, fl, torch.tensor(float(self.df.loc[i,"Diagnosis"]), dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, df, norms, mode): self.df = df.reset_index(drop=True); self.n = norms; self.m = mode
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        nm = self.df.loc[i,"Name"]
        bf = Image.open(os.path.join(BF_TEST, nm)).convert("RGB")
        fl = Image.open(os.path.join(FL_TEST, nm)).convert("RGB")
        bf, fl = tta_apply(bf, fl, self.m, self.n)
        return bf, fl, nm

def subsample_per_patient(df, n, seed):
    if n is None: return df
    return (df.groupby("patient_id", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), n), random_state=seed)).reset_index(drop=True))

# ---------- models ----------
def make_head(d):
    return nn.Sequential(nn.Dropout(0.5), nn.Linear(d,512), nn.BatchNorm1d(512), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(512,128), nn.BatchNorm1d(128), nn.ReLU(),
                         nn.Linear(128,1))

class Backbone(nn.Module):
    def __init__(self, pretrained=True, ssl_path=None):
        super().__init__()
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=w)
        self.features = nn.Sequential(*list(net.children())[:-2])   # [B,512,4,4]
        self.pool = nn.AdaptiveAvgPool2d(1)
        if ssl_path and os.path.exists(ssl_path):
            sd = torch.load(ssl_path, map_location="cpu")  # SSL encoder keys 0..7 == features keys
            res = self.features.load_state_dict(sd, strict=False)
            print(f"  SSL init <- {os.path.basename(ssl_path)} "
                  f"(missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})", flush=True)
    def forward(self, x): return self.pool(self.features(x)).flatten(1)   # [B,512]

class ConcatNet(nn.Module):        # V3
    def __init__(self, pretrained=True, ssl_bf=None, ssl_fl=None):
        super().__init__()
        self.bf = Backbone(pretrained, ssl_bf)
        self.fl = Backbone(pretrained, ssl_fl)
        self.head = make_head(1024)
    def forward(self, bf_x, fl_x): return self.head(torch.cat([self.bf(bf_x), self.fl(fl_x)], 1))

class SingleNet(nn.Module):        # V4 bf_only / V5 fl_only
    def __init__(self, which, pretrained=True, ssl_path=None):
        super().__init__(); self.which = which
        self.enc = Backbone(pretrained, ssl_path); self.head = make_head(512)
    def forward(self, bf_x, fl_x): return self.head(self.enc(bf_x if self.which=="bf" else fl_x))

def build_model(config):
    v = config["variant"]
    ssl_bf, ssl_fl = config.get("ssl_bf"), config.get("ssl_fl")
    if v == "concat":  return ConcatNet(config["pretrained"], ssl_bf, ssl_fl)
    if v == "bf_only": return SingleNet("bf", config["pretrained"], ssl_bf)
    if v == "fl_only": return SingleNet("fl", config["pretrained"], ssl_fl)
    if v == "attn":
        from train_infer import DualBranchMultiModalNet   # reuse the ORIGINAL attn model
        return DualBranchMultiModalNet(DEVICE)
    raise ValueError(v)

# ---------- loss ----------
class SmoothBCE(nn.Module):
    def __init__(self, s=0.05, pos_weight=None):
        super().__init__(); self.s = s; self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, logits, y): return self.bce(logits, y*(1-self.s)+0.5*self.s)

def mixup(bf, fl, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha); idx = torch.randperm(bf.size(0), device=bf.device)
    return lam*bf+(1-lam)*bf[idx], lam*fl+(1-lam)*fl[idx], y, y[idx], lam

# ---------- one fold ----------
def train_one_fold(tr_df, va_df, fold, config, norms):
    set_seed(config["seed"] + fold)
    va_loader = DataLoader(PairedDataset(va_df, PairedTransform(norms, train=False)),
                           batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=False)
    model = build_model(config).to(DEVICE)
    # discriminative LR: protect pretrained backbone (low LR), train head fast (high LR)
    head_keys = ("head", "classifier", "fusion")
    head_p = [p for n, p in model.named_parameters() if any(k in n for k in head_keys)]
    back_p = [p for n, p in model.named_parameters() if not any(k in n for k in head_keys)]
    opt = torch.optim.AdamW(
        [{"params": back_p, "lr": config["lr"] * config.get("backbone_lr_mult", 0.1)},
         {"params": head_p, "lr": config["lr"]}], weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=config["epochs"])
    scaler = GradScaler("cuda")
    pos_w = torch.tensor([(tr_df["Diagnosis"]==0).sum()/(tr_df["Diagnosis"]==1).sum()],
                         dtype=torch.float32, device=DEVICE)
    crit = SmoothBCE(0.05, pos_weight=pos_w)
    best_auc, best_preds = -1.0, None
    path = f"./checkpoints/{config['name']}_fold{fold}.pth"
    for ep in range(config["epochs"]):
        ep_df = subsample_per_patient(tr_df, config["n_per_patient"], config["seed"]+ep)
        tr_loader = DataLoader(PairedDataset(ep_df, PairedTransform(norms, train=True)),
                               batch_size=config["batch_size"], shuffle=True, num_workers=2,
                               pin_memory=False, drop_last=True)
        model.train()
        for bf, fl, y in tqdm(tr_loader, desc=f"{config['name']} f{fold} e{ep+1}", leave=False):
            bf, fl, y = bf.to(DEVICE), fl.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast("cuda"):
                if config["use_mixup"] and random.random() < 0.5:
                    bfm, flm, ya, yb, lam = mixup(bf, fl, y)
                    out = model(bfm, flm).squeeze(1); loss = lam*crit(out, ya) + (1-lam)*crit(out, yb)
                else:
                    out = model(bf, fl).squeeze(1); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); preds, tgts = [], []
        with torch.no_grad():
            for bf, fl, y in va_loader:
                with autocast("cuda"): out = model(bf.to(DEVICE), fl.to(DEVICE)).squeeze(1)
                preds.append(torch.sigmoid(out).float().cpu()); tgts.append(y)
        preds, tgts = torch.cat(preds).numpy(), torch.cat(tgts).numpy()
        auc = roc_auc_score(tgts, preds)
        print(f"  {config['name']} fold{fold} ep{ep+1} val AUC {auc:.4f}")
        if auc > best_auc:                              # <-- select by AUC, not loss
            best_auc, best_preds = auc, preds
            torch.save(model.state_dict(), path)
    return best_auc, best_preds, path

# ---------- run CV ----------
def run_cv(config, train_df, norms, max_folds=None):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    splits = list(sgkf.split(train_df, train_df["Diagnosis"], groups=train_df["patient_id"]))
    oof = np.full(len(train_df), np.nan); fold_aucs, paths = [], []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if max_folds and fold >= max_folds: break
        auc, preds, path = train_one_fold(train_df.iloc[tr_idx], train_df.iloc[va_idx], fold, config, norms)
        oof[va_idx] = preds; fold_aucs.append(auc); paths.append(path)
    mask = ~np.isnan(oof)
    g = roc_auc_score(train_df["Diagnosis"].values[mask], oof[mask])
    print(f"[{config['name']}] global OOF AUC = {g:.4f} | per-fold {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    np.savez(f"./results/{config['name']}_oof.npz", oof=oof, y=train_df["Diagnosis"].values)
    return oof, g, fold_aucs, paths

# ---------- inference + TTA ----------
def predict_test_tta(paths, config, test_df, norms, test_limit=None):
    df = test_df.iloc[:test_limit] if test_limit else test_df
    modes = ["identity","hflip","vflip","rot90"]; fold_probs = []
    for path in paths:
        model = build_model(config).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE)); model.eval()
        mp = []
        for m in modes:
            loader = DataLoader(TestDataset(df, norms, m), batch_size=config["batch_size"],
                                shuffle=False, num_workers=2, pin_memory=False)
            p = []
            with torch.no_grad():
                for bf, fl, _ in tqdm(loader, desc=f"TTA {m}", leave=False):
                    with autocast("cuda"): out = model(bf.to(DEVICE), fl.to(DEVICE)).squeeze(1)
                    p.append(torch.sigmoid(out).float().cpu())
            mp.append(torch.cat(p))
        fold_probs.append(torch.stack(mp).mean(0))
    probs = torch.stack(fold_probs).mean(0).numpy()
    np.save(f"./results/{config['name']}_test_probs.npy", probs)
    return probs

def write_submission(test_df, probs, out_path):
    sub = pd.DataFrame({"Name": test_df["Name"].values[:len(probs)], "Diagnosis": probs})
    assert list(sub.columns) == ["Name","Diagnosis"]
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} | rows={len(sub)} range=[{probs.min():.4f},{probs.max():.4f}]")

# ---------- rank-average ensemble (validate on OOF before trusting) ----------
def rank_average(prob_list):
    r = np.vstack([rankdata(p) for p in prob_list]).mean(0); return r / r.max()
def oof_rank_average_auc(oof_list, y):
    r = np.vstack([rankdata(o) for o in oof_list]).mean(0); return roc_auc_score(y, r)

if __name__ == "__main__":
    set_seed(42)
    for d in ("./checkpoints","./results","./submissions"): os.makedirs(d, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df["patient_id"] = train_df["Name"].apply(lambda x: x.split("_")[1])
    norms = get_norms(train_df["Name"].tolist())
    test_df = pd.DataFrame({"Name": sorted(f for f in os.listdir(BF_TEST) if not f.startswith("."))})

    base = dict(seed=42, batch_size=128, lr=1e-3, epochs=8,
                n_per_patient=4000, use_mixup=False, pretrained=True)  # use_mixup: knob; 0.82 used it

    # ===== SMOKE TEST FIRST (1 fold, 1 epoch, tiny subset, 200 test imgs) =====
    smoke = {**base, "name":"smoke_concat", "variant":"concat", "epochs":1,
             "n_per_patient":300, "batch_size":64}
    oof, auc, faucs, paths = run_cv(smoke, train_df, norms, max_folds=1)
    probs = predict_test_tta(paths, smoke, test_df, norms, test_limit=200)
    write_submission(test_df.iloc[:200], probs, "./submissions/SMOKE.csv")
    # >>> STOP GATE 2: report smoke pass/fail + a time estimate for one FULL 5-fold V3 run,
    #     and wait for my "go" before any full training. <<<

    # ===== FULL RUNS (uncomment after I say "go"; V3 first, then V4/V5 if time) =====
    # v3 = {**base, "name":"v3_concat", "variant":"concat"}
    # _, a3, _, p3 = run_cv(v3, train_df, norms)
    # pr3 = predict_test_tta(p3, v3, test_df, norms)
    # write_submission(test_df, pr3, f"./submissions/v3_concat_oof{a3:.4f}.csv")
    # # V4/V5 likewise -> then rank_average ensembles -> validate via oof_rank_average_auc
    # # -> write top-3 submissions -> STOP GATE 3 (do NOT submit; I choose).
