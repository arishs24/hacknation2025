#!/usr/bin/env python3
"""
Test script for loading and evaluating the trained models on validation data.
Run this after training to test your saved models.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

# RDKit imports
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# ---- Config (should match training script)
VAL_CSV = "bindingdb_kinase_top10_val.csv"
ECFP_BITS = 2048
ECFP_RADIUS = 2
PROT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Utils (copied from training script)
def smiles_to_ecfp(smiles: str, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

@torch.no_grad()
def embed_protein(seq: str, prot_model, prot_tok, max_len=1024):
    toks = prot_tok(seq, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    ids = toks['input_ids'].to(DEVICE)
    mask = toks['attention_mask'].to(DEVICE)
    out = prot_model(input_ids=ids, attention_mask=mask).last_hidden_state
    emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

# Model definition (copied from training script)
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
        self.cls_head = nn.Linear(hidden, 1)
        self.reg_head = nn.Linear(hidden, 1) if reg_head else None

    def forward(self, x):
        z = self.backbone(x)
        logit = self.cls_head(z).squeeze(-1)
        reg = self.reg_head(z).squeeze(-1) if self.reg_head is not None else None
        return logit, reg

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

def load_and_test_models():
    print("Loading validation data...")
    val_df = pd.read_csv(VAL_CSV)
    val_df["sequence"] = val_df["sequence"].astype(str).str.replace(r"\s+", "", regex=True)
    val_df["smiles"] = val_df["smiles"].astype(str).str.strip()
    y_val = val_df["label"].astype(int).to_numpy()
    
    print("Checking saved models...")
    if not os.path.exists("saved_models"):
        print("ERROR: saved_models directory not found. Run train.py first.")
        return
    
    # =====================================================================================
    # 1. Test Baseline Model
    # =====================================================================================
    print("\n[1/3] Testing Baseline (LogReg on ECFP4)...")
    
    if os.path.exists("saved_models/baseline_logreg.pkl"):
        with open("saved_models/baseline_logreg.pkl", "rb") as f:
            logreg = pickle.load(f)
        
        X_val_ecfp = np.stack([smiles_to_ecfp(s) for s in val_df["smiles"].tolist()], axis=0)
        val_proba_baseline = logreg.predict_proba(X_val_ecfp)[:, 1]
        val_pred_baseline = (val_proba_baseline >= 0.5).astype(int)
        
        print("Baseline metrics (ECFP4->LogReg):")
        print("  AUROC :", roc_auc_score(y_val, val_proba_baseline))
        print("  AUPRC :", average_precision_score(y_val, val_proba_baseline))
        print("  Acc   :", accuracy_score(y_val, val_pred_baseline))
        print("  F1    :", f1_score(y_val, val_pred_baseline))
        print("  Brier :", brier_score_loss(y_val, val_proba_baseline))
    else:
        print("Baseline model not found!")
        return
    
    # =====================================================================================
    # 2. Test Fusion Model
    # =====================================================================================
    print("\n[2/3] Testing Fusion Model...")
    
    if not os.path.exists("saved_models/fusion_mlp.pth"):
        print("Fusion model not found!")
        return
    
    # Load fusion model
    checkpoint = torch.load("saved_models/fusion_mlp.pth", map_location=DEVICE)
    model_config = checkpoint['model_config']
    
    model = FusionMLP(
        in_dim=model_config['in_dim'],
        hidden=model_config['hidden'],
        reg_head=model_config['reg_head']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with best AUROC: {checkpoint['training_info']['best_auroc']:.3f}")
    
    # Load protein embeddings or compute them
    if os.path.exists("saved_models/protein_cache.pkl"):
        print("Loading protein embeddings cache...")
        with open("saved_models/protein_cache.pkl", "rb") as f:
            prot_cache = pickle.load(f)
    else:
        print("Computing protein embeddings...")
        prot_tok = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
        prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME).to(DEVICE)
        for p in prot_model.parameters():
            p.requires_grad = False
        
        prot_cache = {}
        for seq in val_df["sequence"].unique():
            prot_cache[seq] = embed_protein(seq, prot_model, prot_tok)
    
    # Prepare fusion features
    X_ecfp = np.stack([smiles_to_ecfp(s) for s in val_df["smiles"].tolist()], axis=0)
    X_prot = np.stack([prot_cache[seq] for seq in val_df["sequence"].tolist()], axis=0)
    X_fuse = np.concatenate([X_ecfp, X_prot], axis=1).astype(np.float32)
    
    # Get predictions
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_fuse).to(DEVICE)
        logits, _ = model(X_tensor)
        probs_uncal = torch.sigmoid(logits).cpu().numpy()
    
    preds_uncal = (probs_uncal >= 0.5).astype(int)
    
    print("Fusion metrics (uncalibrated):")
    print("  AUROC :", roc_auc_score(y_val, probs_uncal))
    print("  AUPRC :", average_precision_score(y_val, probs_uncal))
    print("  Acc   :", accuracy_score(y_val, preds_uncal))
    print("  F1    :", f1_score(y_val, preds_uncal))
    print("  Brier :", brier_score_loss(y_val, probs_uncal))
    
    # =====================================================================================
    # 3. Test Calibrated Model
    # =====================================================================================
    print("\n[3/3] Testing Calibrated Model...")
    
    if os.path.exists("saved_models/temperature_scaler.pth"):
        scaler_checkpoint = torch.load("saved_models/temperature_scaler.pth", map_location=DEVICE)
        scaler = TemperatureScaler().to(DEVICE)
        scaler.load_state_dict(scaler_checkpoint['scaler_state_dict'])
        print(f"Loaded temperature scaler (T={scaler_checkpoint['temperature']:.3f})")
        
        with torch.no_grad():
            logits_tensor = torch.from_numpy(logits.cpu().numpy()).to(DEVICE)
            calibrated_logits = scaler(logits_tensor)
            probs_cal = torch.sigmoid(calibrated_logits).cpu().numpy()
        
        preds_cal = (probs_cal >= 0.5).astype(int)
        
        print("Fusion metrics (calibrated):")
        print("  AUROC :", roc_auc_score(y_val, probs_cal))
        print("  AUPRC :", average_precision_score(y_val, probs_cal))
        print("  Acc   :", accuracy_score(y_val, preds_cal))
        print("  F1    :", f1_score(y_val, preds_cal))
        print("  Brier :", brier_score_loss(y_val, probs_cal))
    else:
        print("Temperature scaler not found!")
        probs_cal = probs_uncal
    
    # =====================================================================================
    # 4. Save Test Results
    # =====================================================================================
    print("\nSaving test results...")
    
    results_df = pd.DataFrame({
        "smiles": val_df["smiles"],
        "sequence": val_df["sequence"],
        "label": y_val,
        "baseline_proba": val_proba_baseline,
        "fusion_proba_uncal": probs_uncal,
        "fusion_proba_cal": probs_cal
    })
    
    results_df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")
    
    # Example prediction function
    def predict_sample(sequence: str, smiles: str):
        """Make a prediction for a single protein-ligand pair"""
        prot_emb = prot_cache.get(sequence)
        if prot_emb is None:
            # If sequence not in cache, compute it
            if 'prot_model' in locals():
                prot_emb = embed_protein(sequence, prot_model, prot_tok)
            else:
                print("Cannot compute protein embedding - sequence not in cache")
                return None, None, None
        
        ecfp = smiles_to_ecfp(smiles)
        x = np.concatenate([ecfp, prot_emb], axis=0)[None, :].astype(np.float32)
        
        # Baseline prediction
        baseline_prob = float(logreg.predict_proba(x[:, :ECFP_BITS])[:, 1][0])
        
        # Fusion predictions
        with torch.no_grad():
            xt = torch.from_numpy(x).to(DEVICE)
            logit, _ = model(xt)
            prob_uncal = float(torch.sigmoid(logit).item())
            prob_cal = float(torch.sigmoid(scaler(logit)).item())
        
        return baseline_prob, prob_uncal, prob_cal
    
    # Test on first validation sample
    ex_seq = val_df.iloc[0]["sequence"]
    ex_smi = val_df.iloc[0]["smiles"]
    ex_label = val_df.iloc[0]["label"]
    
    print(f"\nExample prediction (true label: {ex_label}):")
    baseline_p, fusion_p, fusion_cal_p = predict_sample(ex_seq, ex_smi)
    if baseline_p is not None:
        print(f"  Baseline prob   : {baseline_p:.3f}")
        print(f"  Fusion prob     : {fusion_p:.3f}")
        print(f"  Fusion calibrated: {fusion_cal_p:.3f}")

if __name__ == "__main__":
    load_and_test_models()
