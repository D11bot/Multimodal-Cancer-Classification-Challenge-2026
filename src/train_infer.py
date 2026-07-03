import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ==========================================
# 1. DATASET
# ==========================================

class MultiModalCancerDataset(Dataset):
    def __init__(self, dataframe, bf_dir, fl_dir, transform=None, is_test=False):
        self.df        = dataframe.reset_index(drop=True)
        self.bf_dir    = bf_dir
        self.fl_dir    = fl_dir
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name   = self.df.loc[idx, 'Name']
        img_bf = Image.open(os.path.join(self.bf_dir, name)).convert('RGB')
        img_fl = Image.open(os.path.join(self.fl_dir, name)).convert('RGB')

        if self.transform:
            img_bf = self.transform(img_bf)
            img_fl = self.transform(img_fl)

        if self.is_test:
            return img_bf, img_fl, name

        label = float(self.df.loc[idx, 'Diagnosis'])
        return img_bf, img_fl, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 2. MODEL
# ==========================================

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, feat_dim=512, heads=8):
        super().__init__()
        self.bf_to_fl = nn.MultiheadAttention(feat_dim, heads, batch_first=True)
        self.fl_to_bf = nn.MultiheadAttention(feat_dim, heads, batch_first=True)
        
        self.norm_bf_pre  = nn.LayerNorm(feat_dim)
        self.norm_fl_pre  = nn.LayerNorm(feat_dim)

    def forward(self, bf_spatial, fl_spatial):
        # 1. Normalize first
        bf_norm = self.norm_bf_pre(bf_spatial)
        fl_norm = self.norm_fl_pre(fl_spatial)

        # 2. Attend
        bf_attended, _ = self.bf_to_fl(bf_norm, fl_norm, fl_norm)
        fl_attended, _ = self.fl_to_bf(fl_norm, bf_norm, bf_norm)

        # 3. Residual connection (without a second norm, standard for Pre-Norm)
        bf_out = bf_spatial + bf_attended  # [B, SeqLen, D]
        fl_out = fl_spatial + fl_attended  # [B, SeqLen, D]

        # Global Average Pooling across the sequence dimension
        bf_pooled = bf_out.mean(dim=1) # [B, D]
        fl_pooled = fl_out.mean(dim=1) # [B, D]

        return torch.cat([bf_pooled, fl_pooled], dim=1)  # [B, 2D]


class DualBranchMultiModalNet(nn.Module):
    def __init__(self, device, feat_dim=512):
        super().__init__()

        # ---- BF Encoder (Strip linear AND avgpool -> [:-2]) ----
        resnet_bf = models.resnet18()
        self.bf_branch = nn.Sequential(*list(resnet_bf.children())[:-2])
        if os.path.exists('./checkpoints/BFpretrained_resnet18_encoder.pth'):
            # strict=False allows loading weights even if avgpool is missing from our current definition
            self.bf_branch.load_state_dict(
                torch.load('./checkpoints/BFpretrained_resnet18_encoder.pth', map_location=device), strict=False
            )
            print("  Loaded BF SSL encoder.")
        else:
            print("  WARNING: BFpretrained_resnet18_encoder.pth not found. Using random init.")

        # ---- FL Encoder (Strip linear AND avgpool -> [:-2]) ----
        resnet_fl = models.resnet18()
        self.fl_branch = nn.Sequential(*list(resnet_fl.children())[:-2])
        if os.path.exists('./checkpoints/FLpretrained_resnet18_encoder.pth'):
            self.fl_branch.load_state_dict(
                torch.load('./checkpoints/FLpretrained_resnet18_encoder.pth', map_location=device), strict=False
            )
            print("  Loaded FL SSL encoder.")
        else:
            print("  WARNING: FLpretrained_resnet18_encoder.pth not found. Using random init.")

        self.fusion = CrossModalAttentionFusion(feat_dim=feat_dim, heads=8)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, bf_x, fl_x):
        bf_feat = self.bf_branch(bf_x) # [B, 512, H, W]
        fl_feat = self.fl_branch(fl_x) # [B, 512, H, W]
        
        # Flatten spatial dims: [B, 512, H, W] -> [B, 512, H*W] -> [B, H*W, 512]
        B, C, H, W = bf_feat.size()
        bf_spatial = bf_feat.view(B, C, -1).permute(0, 2, 1) 
        fl_spatial = fl_feat.view(B, C, -1).permute(0, 2, 1)
        
        fused = self.fusion(bf_spatial, fl_spatial)  # [B, 1024]
        return self.classifier(fused)                # [B, 1]


# ==========================================
# 3. LOSS
# ==========================================

class LabelSmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing  = smoothing
        self.bce        = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)


# ==========================================
# 4. MIXUP
# ==========================================

def mixup_batch(bf, fl, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(bf.size(0), device=bf.device)
    return (
        lam * bf     + (1 - lam) * bf[idx],
        lam * fl     + (1 - lam) * fl[idx],
        labels,
        labels[idx],
        lam
    )


# ==========================================
# 5. TRANSFORMS
# ==========================================

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_transforms_list = [
    val_test_transforms,
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]


# ==========================================
# 6. MAIN BLOCK
# ==========================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    DATA_DIR     = './data/multimodal-cancer-classification-challenge-2026'
    BF_TRAIN_DIR = os.path.join(DATA_DIR, 'BF', 'train')
    FL_TRAIN_DIR = os.path.join(DATA_DIR, 'FL', 'train')
    BF_TEST_DIR  = os.path.join(DATA_DIR, 'BF', 'test')
    FL_TEST_DIR  = os.path.join(DATA_DIR, 'FL', 'test')

    NUM_EPOCHS   = 8
    BATCH_SIZE   = 128
    LR           = 1e-3
    WEIGHT_DECAY = 1e-2
    USE_MIXUP    = True
    MIXUP_ALPHA  = 0.4
    TTA_ROUNDS   = len(tta_transforms_list)

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_df['patient_id'] = train_df['Name'].apply(lambda x: x.split('_')[1])

    unique_patients = train_df['patient_id'].unique()
    print(f"\nTraining patients ({len(unique_patients)}): {sorted(unique_patients)}")

    logo = LeaveOneGroupOut()
    fold_model_paths = []
    
    # Track Out-Of-Fold predictions for the Global AUC
    oof_preds = []
    oof_targets = []
    
    scaler = GradScaler('cuda')

    splits = list(logo.split(train_df, train_df['Diagnosis'], groups=train_df['patient_id']))
    NUM_FOLDS = len(splits)
    print(f"\nRunning {NUM_FOLDS}-fold Leave-One-Patient-Out CV")

    for fold, (train_idx, val_idx) in enumerate(splits):
        val_patient = train_df.iloc[val_idx]['patient_id'].unique()[0]
        print(f"\n{'='*20} FOLD {fold+1}/{NUM_FOLDS} | Val patient: {val_patient} {'='*20}")

        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data   = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = MultiModalCancerDataset(train_data, BF_TRAIN_DIR, FL_TRAIN_DIR, train_transforms)
        val_dataset   = MultiModalCancerDataset(val_data,   BF_TRAIN_DIR, FL_TRAIN_DIR, val_test_transforms)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model     = DualBranchMultiModalNet(device).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS, pct_start=0.1, anneal_strategy='cos')

        num_healthy = (train_data['Diagnosis'] == 0).sum()
        num_cancer  = (train_data['Diagnosis'] == 1).sum()
        pos_weight  = torch.tensor([num_healthy / num_cancer]).to(device)
        criterion   = LabelSmoothBCEWithLogitsLoss(smoothing=0.05, pos_weight=pos_weight)

        best_val_loss     = float('inf')
        best_fold_preds   = []
        best_fold_targets = []
        
        model_save_path = f'./checkpoints/best_model_fold_{fold+1}.pth'
        fold_model_paths.append(model_save_path)

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0

            for bf_images, fl_images, labels in tqdm(train_loader, desc=f"  Train E{epoch+1}"):
                bf_images = bf_images.to(device, non_blocking=True)
                fl_images = fl_images.to(device, non_blocking=True)
                labels    = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with autocast('cuda'):
                    if USE_MIXUP and np.random.random() < 0.5:
                        bf_mix, fl_mix, la, lb, lam = mixup_batch(bf_images, fl_images, labels, MIXUP_ALPHA)
                        outputs = model(bf_mix, fl_mix).squeeze()
                        if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                        loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
                    else:
                        outputs = model(bf_images, fl_images).squeeze()
                        if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                        loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * bf_images.size(0)

            epoch_loss = running_loss / len(train_dataset)

            # ---- Validate ----
            model.eval()
            val_running_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for bf_images, fl_images, labels in val_loader:
                    bf_images = bf_images.to(device, non_blocking=True)
                    fl_images = fl_images.to(device, non_blocking=True)
                    labels    = labels.to(device, non_blocking=True)
                    
                    with autocast('cuda'):
                        outputs = model(bf_images, fl_images).squeeze()
                        if outputs.dim() == 0: 
                            outputs = outputs.unsqueeze(0)
                        loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * bf_images.size(0)
                    
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            epoch_val_loss = val_running_loss / len(val_dataset)
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:>2} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_fold_preds = all_preds
                best_fold_targets = all_targets
                torch.save(model.state_dict(), model_save_path)
                print(f"  --> Saved best model (Val Loss: {best_val_loss:.4f})")

        oof_preds.extend(best_fold_preds)
        oof_targets.extend(best_fold_targets)
        print(f"\nFold {fold+1} finished. Best Val Loss: {best_val_loss:.4f}  (val patient: {val_patient})")

    print(f"\n{'='*50}")
    global_oof_auc = roc_auc_score(oof_targets, oof_preds)
    print(f"Global LOPO Cross-Validation AUC: {global_oof_auc:.4f}")
    print(f"{'='*50}")

    # ==========================================
    # INFERENCE WITH TTA
    # ==========================================
    test_names = sorted([f for f in os.listdir(BF_TEST_DIR) if not f.startswith('.')])
    test_df    = pd.DataFrame({'Name': test_names})
    print(f"\nTest set: {len(test_df)} images")
    print(f"Running inference: {NUM_FOLDS} folds × {TTA_ROUNDS} TTA = {NUM_FOLDS * TTA_ROUNDS} forward passes per image")

    all_fold_tta_preds = []

    for fold_idx, model_path in enumerate(fold_model_paths):
        print(f"\nFold {fold_idx+1}/{NUM_FOLDS}: {model_path}")
        model = DualBranchMultiModalNet(device).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for tta_idx, tta_tf in enumerate(tta_transforms_list):
            tta_dataset = MultiModalCancerDataset(test_df, BF_TEST_DIR, FL_TEST_DIR, transform=tta_tf, is_test=True)
            tta_loader  = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            tta_preds  = []
            name_order = [] 

            with torch.no_grad():
                for bf_images, fl_images, names in tqdm(tta_loader, desc=f"  TTA {tta_idx+1}/{TTA_ROUNDS}", leave=False):
                    bf_images = bf_images.to(device, non_blocking=True)
                    fl_images = fl_images.to(device, non_blocking=True)
                    with autocast('cuda'):
                        outputs = model(bf_images, fl_images).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    probs = torch.sigmoid(outputs)
                    tta_preds.extend(probs.cpu().numpy())
                    if fold_idx == 0 and tta_idx == 0:
                        name_order.extend(names)

            all_fold_tta_preds.append(tta_preds)

    final_name_order = name_order if name_order else test_names
    pred_matrix      = np.array(all_fold_tta_preds) 
    final_predictions = pred_matrix.mean(axis=0) 

    submission = pd.DataFrame({
        'Name':      final_name_order,
        'Diagnosis': final_predictions
    })
    submission.to_csv('./submissions/submission.csv', index=False)
    print(f"\nsubmission.csv saved — {len(submission)} predictions")
    print(f"  Prediction range : [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
    print(f"  Mean prediction  : {final_predictions.mean():.4f}")