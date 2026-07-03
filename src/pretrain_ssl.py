import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ==========================================
# 1. CLASS DEFINITIONS
# ==========================================

class PairedCellDataset(Dataset):
    """
    Returns a matched (BF, FL) image pair for the same cell.
    Used for cross-modal contrastive pretraining.
    """
    def __init__(self, image_names, bf_dir, fl_dir, transform):
        self.names = image_names
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        bf = Image.open(os.path.join(self.bf_dir, name)).convert('RGB')
        fl = Image.open(os.path.join(self.fl_dir, name)).convert('RGB')
        return self.transform(bf), self.transform(fl)


class CrossModalContrastiveNet(nn.Module):
    """
    Two ResNet18 encoders with projection heads.
    """
    def __init__(self, proj_dim=128):
        super().__init__()

        resnet_bf = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_fl = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Encoders: output [B, 512, 1, 1] → flattened to [B, 512]
        self.bf_encoder = nn.Sequential(*list(resnet_bf.children())[:-1])
        self.fl_encoder = nn.Sequential(*list(resnet_fl.children())[:-1])

        # Projection heads (SimCLR-style): 512 → 256 → proj_dim
        # These are DISCARDED after pretraining
        self.bf_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.fl_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, bf_img, fl_img):
        bf_feat = self.bf_encoder(bf_img).flatten(1)  # [B, 512]
        fl_feat = self.fl_encoder(fl_img).flatten(1)  # [B, 512]
        z_bf = self.bf_proj(bf_feat)                  # [B, proj_dim]
        z_fl = self.fl_proj(fl_feat)                  # [B, proj_dim]
        return z_bf, z_fl


# ==========================================
# 2. LOSS FUNCTION
# ==========================================

def nt_xent_loss(z1, z2, temperature=0.07):
    """
    NT-Xent loss (SimCLR). For each image i, its positive pair is the
    other-modality image of the same cell. All other 2(B-1) images are negatives.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)              # [2B, proj_dim]
    sim = torch.mm(z, z.T) / temperature         # [2B, 2B]

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels + B, labels])     # [2B]

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss


# ==========================================
# 3. EXECUTION BLOCK
# ==========================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    DATA_DIR = './data/multimodal-cancer-classification-challenge-2026'

    BF_TRAIN_DIR = os.path.join(DATA_DIR, 'BF', 'train')
    FL_TRAIN_DIR = os.path.join(DATA_DIR, 'FL', 'train')
    BF_TEST_DIR  = os.path.join(DATA_DIR, 'BF', 'test')
    FL_TEST_DIR  = os.path.join(DATA_DIR, 'FL', 'test')

    # ---- Hyperparameters ----
    BATCH_SIZE  = 256 
    EPOCHS      = 12
    LR          = 3e-4
    WEIGHT_DECAY = 1e-4
    TEMPERATURE  = 0.07
    PROJ_DIM     = 128

    # ---- Transforms ----
    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- Datasets ----
    print("Locating images...")
    train_names = [f for f in os.listdir(BF_TRAIN_DIR) if not f.startswith('.')]
    test_names  = [f for f in os.listdir(BF_TEST_DIR)  if not f.startswith('.')]
    print(f"  Train images : {len(train_names)}")
    print(f"  Test images  : {len(test_names)}")
    print(f"  Total SSL    : {len(train_names) + len(test_names)}")

    ds_train = PairedCellDataset(train_names, BF_TRAIN_DIR, FL_TRAIN_DIR, ssl_transform)
    ds_test  = PairedCellDataset(test_names,  BF_TEST_DIR,  FL_TEST_DIR,  ssl_transform)
    ssl_dataset = ConcatDataset([ds_train, ds_test])

    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True   # NT-Xent needs consistent batch sizes
    )

    # ---- Model / Optimizer / Scheduler ----
    model     = CrossModalContrastiveNet(proj_dim=PROJ_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler('cuda')

    best_loss = float('inf')

    # ---- Training Loop ----
    print("\nStarting cross-modal contrastive pretraining...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for bf_imgs, fl_imgs in tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{EPOCHS}"):
            bf_imgs = bf_imgs.to(device, non_blocking=True)
            fl_imgs = fl_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda'):
                z_bf, z_fl = model(bf_imgs, fl_imgs)
                loss = nt_xent_loss(z_bf, z_fl, temperature=TEMPERATURE)

            scaler.scale(loss).backward()
            # Gradient clipping for stable training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(ssl_loader)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:>2}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint of full model (encoders + proj heads)
            torch.save(model.state_dict(), './checkpoints/ssl_checkpoint_best.pth')

        # Save encoders EVERY epoch (resilience: usable even if pretraining is stopped early)
        torch.save(model.bf_encoder.state_dict(), './checkpoints/BFpretrained_resnet18_encoder.pth')
        torch.save(model.fl_encoder.state_dict(), './checkpoints/FLpretrained_resnet18_encoder.pth')

    # ---- Save Encoder Weights Only ----
    # The projection heads are discarded — only the encoders transfer to the classifier.
    torch.save(model.bf_encoder.state_dict(), './checkpoints/BFpretrained_resnet18_encoder.pth')
    torch.save(model.fl_encoder.state_dict(), './checkpoints/FLpretrained_resnet18_encoder.pth')
    print("\nPretraining complete.")
    print(f"  Saved: BFpretrained_resnet18_encoder.pth")
    print(f"  Saved: FLpretrained_resnet18_encoder.pth")
    print(f"  Best contrastive loss: {best_loss:.4f}")