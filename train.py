# ============================================
# Mini AlphaFold – Binding (Core: 2 models + 1 calibrator)
# Baseline: ECFP4 -> Logistic Regression
# Fusion: [ECFP4 || ESM2-sequence embedding] -> tiny MLP
# Calibrator: Temperature scaling on validation logits
# ============================================

# ---- Imports
import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# ECFP4``
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# ---- Config
TRAIN_CSV = "bindingdb_kinase_top10_train.csv"  # your prepared training set (no split needed)
VAL_CSV   = "bindingdb_kinase_top10_val.csv"     # your prepared validation set
SEED = 42
ECFP_BITS = 2048
ECFP_RADIUS = 2           # ECFP4 => radius=2
BATCH_SIZE = 32
LR = 2e-3
EPOCHS = 32
PROT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-trained model loading options
LOAD_PRETRAINED = True  # Set to True to load pre-trained models
PRETRAINED_DIR = "saved_models"  # Directory containing pre-trained models
LOAD_BASELINE = True     # Load pre-trained baseline model
LOAD_FUSION = True       # Load pre-trained fusion model  
LOAD_SCALER = True       # Load pre-trained temperature scaler
LOAD_PROTEIN_CACHE = True  # Load pre-computed protein embeddings
FINE_TUNE_MODE = False   # If True, use lower learning rate for fine-tuning
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------
# Utils
# --------------------------------------------
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

def smiles_to_ecfp(smiles: str, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # empty vector for invalid SMILES
        return np.zeros(n_bits, dtype=np.float32)
    
    # Use the modern MorganGenerator approach instead of deprecated GetMorganFingerprintAsBitVect
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

# --------------------------------------------
# Pre-trained model loading functions
# --------------------------------------------
def load_pretrained_baseline(pretrained_dir=PRETRAINED_DIR):
    """Load pre-trained baseline logistic regression model"""
    baseline_path = os.path.join(pretrained_dir, "baseline_logreg.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Loaded pre-trained baseline model from {baseline_path}")
        return model
    else:
        print(f"✗ Pre-trained baseline model not found at {baseline_path}")
        return None

def load_pretrained_fusion(pretrained_dir=PRETRAINED_DIR):
    """Load pre-trained fusion model"""
    fusion_path = os.path.join(pretrained_dir, "fusion_mlp.pth")
    if os.path.exists(fusion_path):
        checkpoint = torch.load(fusion_path, map_location=DEVICE)
        print(f"✓ Loaded pre-trained fusion model from {fusion_path}")
        print(f"  - Previous best AUROC: {checkpoint['training_info']['best_auroc']:.3f}")
        print(f"  - Trained for {checkpoint['training_info']['epochs']} epochs")
        return checkpoint
    else:
        print(f"✗ Pre-trained fusion model not found at {fusion_path}")
        return None

def load_pretrained_scaler(pretrained_dir=PRETRAINED_DIR):
    """Load pre-trained temperature scaler"""
    scaler_path = os.path.join(pretrained_dir, "temperature_scaler.pth")
    if os.path.exists(scaler_path):
        checkpoint = torch.load(scaler_path, map_location=DEVICE)
        print(f"✓ Loaded pre-trained temperature scaler from {scaler_path}")
        print(f"  - Temperature: {checkpoint['temperature']:.3f}")
        return checkpoint
    else:
        print(f"✗ Pre-trained temperature scaler not found at {scaler_path}")
        return None

def load_protein_cache(pretrained_dir=PRETRAINED_DIR):
    """Load pre-computed protein embeddings cache"""
    cache_path = os.path.join(pretrained_dir, "protein_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"✓ Loaded protein embeddings cache from {cache_path}")
        print(f"  - Contains {len(cache)} pre-computed protein embeddings")
        return cache
    else:
        print(f"✗ Protein cache not found at {cache_path}")
        return {}

# --------------------------------------------
# Load data
# Expecting columns: sequence, smiles, label (0/1). Optional: px (float)
# --------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

for df in (train_df, val_df):
    df["sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True)
    df["smiles"]   = df["smiles"].astype(str).str.strip()

# =====================================================================================
# 1) BASELINE MODEL (Logistic Regression on ECFP4)  — "fast & trusty"
# =====================================================================================
print("\n[1/3] Training Baseline (LogReg on ECFP4) ...")

X_train_ecfp = np.stack([smiles_to_ecfp(s) for s in train_df["smiles"].tolist()], axis=0)
y_train = train_df["label"].astype(int).to_numpy()

X_val_ecfp = np.stack([smiles_to_ecfp(s) for s in val_df["smiles"].tolist()], axis=0)
y_val = val_df["label"].astype(int).to_numpy()

# Optional: scale features (binary bits can be left unscaled; here we skip scaling)
# Load pre-trained baseline model or train new one
if LOAD_PRETRAINED and LOAD_BASELINE:
    logreg = load_pretrained_baseline()
    if logreg is None:
        print("Pre-trained baseline not found, training new model...")
        logreg = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
        logreg.fit(X_train_ecfp, y_train)
    else:
        print("Using pre-trained baseline model (skipping training)")
else:
    # Train new baseline model
    logreg = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    logreg.fit(X_train_ecfp, y_train)

val_proba_baseline = logreg.predict_proba(X_val_ecfp)[:, 1]
val_pred_baseline  = (val_proba_baseline >= 0.5).astype(int)

print("Baseline metrics (ECFP4->LogReg):")
print("  AUROC :", roc_auc_score(y_val, val_proba_baseline))
print("  AUPRC :", average_precision_score(y_val, val_proba_baseline))
print("  Acc   :", accuracy_score(y_val, val_pred_baseline))
print("  F1    :", f1_score(y_val, val_pred_baseline))
print("  Brier :", brier_score_loss(y_val, val_proba_baseline))

# =====================================================================================
# 2) MINI FUSION MODEL (ECFP4 || ESM2 embedding) -> tiny MLP (binary classification)
# Optional second head for pKd regression if 'px' provided
# =====================================================================================

print("\n[2/3] Preparing protein embeddings with ESM2 (frozen)...")

# Load pre-computed protein embeddings if available
if LOAD_PRETRAINED and LOAD_PROTEIN_CACHE:
    prot_cache = load_protein_cache()
else:
    prot_cache = {}

# Initialize protein model
prot_tok = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME).to(DEVICE)
for p in prot_model.parameters():
    p.requires_grad = False
prot_dim = prot_model.config.hidden_size  # (e.g., 320)

@torch.no_grad()
def embed_protein(seq: str, max_len=1024):
    toks = prot_tok(seq, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    ids = toks['input_ids'].to(DEVICE)
    mask = toks['attention_mask'].to(DEVICE)
    out = prot_model(input_ids=ids, attention_mask=mask).last_hidden_state  # [1, L, H]
    # mean-pool with mask
    emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [H]

# Precompute per-unique sequence (speeds up a ton)
def build_protein_embedding_cache(df, existing_cache=None):
    if existing_cache is None:
        existing_cache = {}
    cache = existing_cache.copy()  # Start with existing cache
    
    uniq = df["sequence"].unique().tolist()
    new_sequences = [seq for seq in uniq if seq not in cache]
    
    if len(new_sequences) == 0:
        print(f"  All {len(uniq)} sequences already in cache!")
        return cache
    
    print(f"  Found {len(new_sequences)} new sequences to embed (out of {len(uniq)} total)")
    for i, seq in enumerate(new_sequences, 1):
        cache[seq] = embed_protein(seq)
        if i % 50 == 0:
            print(f"  embedded {i}/{len(new_sequences)} new sequences")
    
    return cache

# Build or update protein embedding cache
prot_cache = build_protein_embedding_cache(pd.concat([train_df, val_df], axis=0), prot_cache)

# Construct fusion features: [ECFP(2048) || ProtEmbed(prot_dim)]
def fuse_features(df, prot_cache):
    X_ecfp = np.stack([smiles_to_ecfp(s) for s in df["smiles"].tolist()], axis=0)
    X_prot = np.stack([prot_cache[seq] for seq in df["sequence"].tolist()], axis=0)
    X = np.concatenate([X_ecfp, X_prot], axis=1).astype(np.float32)
    y_cls = df["label"].astype(int).to_numpy()
    y_reg = df["px"].to_numpy(dtype=np.float32) if "px" in df.columns else None
    return X, y_cls, y_reg

Xtr_fuse, ytr_cls, ytr_reg = fuse_features(train_df, prot_cache)
Xva_fuse, yva_cls, yva_reg = fuse_features(val_df, prot_cache)

# Tiny PyTorch dataset/loaders for the fusion model
class FusionDS(Dataset):
    def __init__(self, X, y_cls, y_reg=None):
        self.X = torch.from_numpy(X)
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_reg = None if y_reg is None else torch.from_numpy(y_reg).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        out = {"x": self.X[i], "y_cls": self.y_cls[i]}
        if self.y_reg is not None:
            out["y_reg"] = self.y_reg[i]
        return out

train_ds = FusionDS(Xtr_fuse, ytr_cls, ytr_reg)
val_ds   = FusionDS(Xva_fuse, yva_cls, yva_reg)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Small MLP head (CPU-friendly)
class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, reg_head=False):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(hidden, 1)        # binary logit
        self.reg_head = nn.Linear(hidden, 1) if reg_head else None

    def forward(self, x):
        z = self.backbone(x)
        logit = self.cls_head(z).squeeze(-1)
        reg = self.reg_head(z).squeeze(-1) if self.reg_head is not None else None
        return logit, reg

in_dim = Xtr_fuse.shape[1]
reg_on = (ytr_reg is not None)

# Load pre-trained fusion model or create new one
if LOAD_PRETRAINED and LOAD_FUSION:
    fusion_checkpoint = load_pretrained_fusion()
    if fusion_checkpoint is not None:
        # Verify model compatibility
        config = fusion_checkpoint['model_config']
        if config['in_dim'] == in_dim and config['reg_head'] == reg_on:
            model = FusionMLP(in_dim=in_dim, hidden=config['hidden'], reg_head=reg_on).to(DEVICE)
            model.load_state_dict(fusion_checkpoint['model_state_dict'])
            print("✓ Successfully loaded pre-trained fusion model")
            # Adjust learning rate for fine-tuning if specified
            lr = LR * 0.1 if FINE_TUNE_MODE else LR
            if FINE_TUNE_MODE:
                print(f"Fine-tuning mode: Using reduced learning rate {lr}")
        else:
            print(f"✗ Model architecture mismatch:")
            print(f"  Expected: in_dim={in_dim}, reg_head={reg_on}")
            print(f"  Found: in_dim={config['in_dim']}, reg_head={config['reg_head']}")
            print("Creating new model...")
            model = FusionMLP(in_dim=in_dim, hidden=256, reg_head=reg_on).to(DEVICE)
            lr = LR
    else:
        print("Pre-trained fusion model not found, creating new model...")
        model = FusionMLP(in_dim=in_dim, hidden=256, reg_head=reg_on).to(DEVICE)
        lr = LR
else:
    # Create new fusion model
    model = FusionMLP(in_dim=in_dim, hidden=256, reg_head=reg_on).to(DEVICE)
    lr = LR

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def evaluate(model, loader):
    model.eval()
    all_logits, all_probs, all_labels = [], [], []
    all_regs_pred, all_regs_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y_cls"].to(DEVICE)
            logit, reg = model(x)
            prob = torch.sigmoid(logit)
            all_logits.append(logit.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            if reg is not None and "y_reg" in batch:
                all_regs_pred.append(reg.cpu().numpy())
                all_regs_true.append(batch["y_reg"].numpy())
    logits = np.concatenate(all_logits)
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, (probs>=0.5).astype(int)),
        "f1": f1_score(labels, (probs>=0.5).astype(int)),
        "brier": brier_score_loss(labels, probs),
    }
    reg_metrics = None
    if reg_on and len(all_regs_pred) > 0:
        rpred = np.concatenate(all_regs_pred).astype(np.float32)
        rtrue = np.concatenate(all_regs_true).astype(np.float32)
        rmse = float(np.sqrt(np.mean((rpred - rtrue)**2)))
        mae  = float(np.mean(np.abs(rpred - rtrue)))
        metrics.update({"rmse_px": rmse, "mae_px": mae})
        reg_metrics = (rpred, rtrue)
    return metrics, logits, (reg_metrics if reg_on else None)

print("\n[2/3] Training Fusion MLP ...")
best_auroc, best_state = -1.0, None
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        x = batch["x"].to(DEVICE)
        y = batch["y_cls"].float().to(DEVICE)

        logits, reg = model(x)
        loss_cls = F.binary_cross_entropy_with_logits(logits, y)

        if reg_on and "y_reg" in batch:
            yreg = batch["y_reg"].to(DEVICE)
            loss_reg = F.smooth_l1_loss(reg, yreg)
            loss = loss_cls + 0.2*loss_reg  # small weight on regression head
        else:
            loss = loss_cls

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    val_metrics, val_logits, _ = evaluate(model, val_loader)
    if val_metrics["auroc"] > best_auroc:
        best_auroc = val_metrics["auroc"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_loader):.4f} | "
          f"VAL auroc={val_metrics['auroc']:.3f} auprc={val_metrics['auprc']:.3f} "
          f"acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f} brier={val_metrics['brier']:.4f}"
          + (f" | rmse_px={val_metrics['rmse_px']:.3f} mae_px={val_metrics['mae_px']:.3f}" if reg_on else ""))

# Load best
if best_state is not None:
    model.load_state_dict(best_state)

# =====================================================================================
# 3) CALIBRATOR — Temperature scaling on validation logits
# =====================================================================================
print("\n[3/3] Calibrating with Temperature Scaling (on fusion logits)...")

# Gather logits/labels from val
with torch.no_grad():
    model.eval()
    all_logits = []
    all_labels = []
    for batch in val_loader:
        x = batch["x"].to(DEVICE)
        y = batch["y_cls"].to(DEVICE)
        logit, _ = model(x)
        all_logits.append(logit.detach().cpu())
        all_labels.append(y.detach().cpu())
val_logits = torch.cat(all_logits)           # shape [N]
val_labels = torch.cat(all_labels).float()   # shape [N]

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()
        # learn log-temperature for positivity
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

def fit_temperature(logits, labels, max_iter=200, lr=0.05):
    scaler = TemperatureScaler().to(DEVICE)
    opt = torch.optim.LBFGS([scaler.log_temp], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    bce = nn.BCEWithLogitsLoss()

    logits = logits.to(DEVICE)
    labels = labels.to(DEVICE)

    def closure():
        opt.zero_grad()
        loss = bce(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    # final loss
    with torch.no_grad():
        final_loss = bce(scaler(logits), labels).item()
    return scaler, final_loss

scaler, cal_loss = fit_temperature(val_logits, val_labels)
print(f"  Learned temperature = {float(torch.exp(scaler.log_temp).detach().cpu()):.3f} (val BCE={cal_loss:.4f})")

# Metrics pre/post calibration
def probs_from_logits(l):
    return torch.sigmoid(l).cpu().detach().numpy()

pre_probs = probs_from_logits(val_logits)
post_probs = probs_from_logits(scaler(val_logits))

print("Calibration effect (Fusion on VAL):")
for name, probs in [("pre", pre_probs), ("post", post_probs)]:
    print(f"  {name:>4} | AUROC={roc_auc_score(yva_cls, probs):.3f} "
          f"AUPRC={average_precision_score(yva_cls, probs):.3f} "
          f"Acc={accuracy_score(yva_cls, (probs>=0.5).astype(int)):.3f} "
          f"Brier={brier_score_loss(yva_cls, probs):.4f}")

# =====================================================================================
# Tiny demo: predict function
#   - Baseline: ECFP -> probability
#   - Fusion: ECFP||Prot -> uncalibrated & calibrated prob
# =====================================================================================
@torch.no_grad()
def predict_baseline(smiles: str):
    x = smiles_to_ecfp(smiles)[None, :]
    p = float(logreg.predict_proba(x)[:, 1][0])
    return p

@torch.no_grad()
def predict_fusion(sequence: str, smiles: str):
    prot_emb = embed_protein(sequence)           # [prot_dim]
    ecfp = smiles_to_ecfp(smiles)                # [2048]
    x = np.concatenate([ecfp, prot_emb], axis=0)[None, :].astype(np.float32)
    xt = torch.from_numpy(x).to(DEVICE)
    logit, _ = model(xt)
    p_uncal = float(torch.sigmoid(logit).item())
    p_cal = float(torch.sigmoid(scaler(logit)).item())
    return p_uncal, p_cal

# Example on first val row
ex_seq = val_df.iloc[0]["sequence"]
ex_smi = val_df.iloc[0]["smiles"]
print("\nExample prediction on 1st VAL row:")
print("  Baseline prob   :", predict_baseline(ex_smi))
p_u, p_c = predict_fusion(ex_seq, ex_smi)
print("  Fusion prob     :", p_u)
print("  Fusion calibrated:", p_c)

# Save artifacts (optional)
pd.DataFrame({
    "smiles": val_df["smiles"],
    "sequence": val_df["sequence"],
    "label": yva_cls,
    "baseline_proba": val_proba_baseline,
    "fusion_proba_pre": pre_probs,
    "fusion_proba_post": post_probs
}).to_csv("val_predictions_core.csv", index=False)

# Save trained models to disk
print("\nSaving models...")

# Create models directory
os.makedirs("saved_models", exist_ok=True)

# 1. Save baseline logistic regression model
with open("saved_models/baseline_logreg.pkl", "wb") as f:
    pickle.dump(logreg, f)

# 2. Save fusion neural network model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'in_dim': in_dim,
        'hidden': 256,
        'reg_head': reg_on
    },
    'training_info': {
        'best_auroc': best_auroc,
        'epochs': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE
    }
}, "saved_models/fusion_mlp.pth")

# 3. Save temperature scaler
torch.save({
    'scaler_state_dict': scaler.state_dict(),
    'temperature': float(torch.exp(scaler.log_temp).detach().cpu())
}, "saved_models/temperature_scaler.pth")

# 4. Save protein embedding cache (optional - saves time on future runs)
with open("saved_models/protein_cache.pkl", "wb") as f:
    pickle.dump(prot_cache, f)

print("\nDone. Artifacts:")
print("  - Trained baseline (saved to saved_models/baseline_logreg.pkl)")
print("  - Trained fusion MLP (saved to saved_models/fusion_mlp.pth)")
print("  - Temperature scaler (saved to saved_models/temperature_scaler.pth)")
print("  - Protein embeddings cache (saved to saved_models/protein_cache.pkl)")
print("  - CSV: val_predictions_core.csv")

# =====================================================================================
# HOW TO USE PRE-TRAINED MODELS:
# =====================================================================================
# To load and continue training from pre-trained models, modify the config section:
#
# LOAD_PRETRAINED = True       # Enable pre-trained model loading
# LOAD_BASELINE = True         # Load pre-trained baseline model
# LOAD_FUSION = True           # Load pre-trained fusion model
# LOAD_SCALER = True           # Load pre-trained temperature scaler
# LOAD_PROTEIN_CACHE = True    # Load pre-computed protein embeddings
# FINE_TUNE_MODE = True        # Use lower learning rate for fine-tuning
#
# This will:
# - Skip training if pre-trained models are found and compatible
# - Load protein embedding cache to save computation time
# - Use reduced learning rate (LR * 0.1) if FINE_TUNE_MODE = True
# - Fall back to training new models if pre-trained ones are not found
# =====================================================================================
