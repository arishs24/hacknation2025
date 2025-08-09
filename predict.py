#!/usr/bin/env python3
"""
Simple prediction script for new protein-ligand pairs.
Use this to make predictions with your trained models.
"""

import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# RDKit imports
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# ---- Config
ECFP_BITS = 2048
ECFP_RADIUS = 2
PROT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model definitions (must match training script)
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

class BindingPredictor:
    """
    Wrapper class for making binding predictions with trained models
    """
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.baseline_model = None
        self.fusion_model = None
        self.scaler = None
        self.prot_cache = {}
        self.prot_model = None
        self.prot_tok = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        # Load baseline model
        baseline_path = os.path.join(self.models_dir, "baseline_logreg.pkl")
        if os.path.exists(baseline_path):
            with open(baseline_path, "rb") as f:
                self.baseline_model = pickle.load(f)
            print("✓ Baseline logistic regression model loaded")
        else:
            print("✗ Baseline model not found")
        
        # Load fusion model
        fusion_path = os.path.join(self.models_dir, "fusion_mlp.pth")
        if os.path.exists(fusion_path):
            checkpoint = torch.load(fusion_path, map_location=DEVICE)
            config = checkpoint['model_config']
            
            self.fusion_model = FusionMLP(
                in_dim=config['in_dim'],
                hidden=config['hidden'],
                reg_head=config['reg_head']
            ).to(DEVICE)
            self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
            self.fusion_model.eval()
            print("✓ Fusion MLP model loaded")
        else:
            print("✗ Fusion model not found")
        
        # Load temperature scaler
        scaler_path = os.path.join(self.models_dir, "temperature_scaler.pth")
        if os.path.exists(scaler_path):
            scaler_checkpoint = torch.load(scaler_path, map_location=DEVICE)
            self.scaler = TemperatureScaler().to(DEVICE)
            self.scaler.load_state_dict(scaler_checkpoint['scaler_state_dict'])
            print("✓ Temperature scaler loaded")
        else:
            print("✗ Temperature scaler not found")
        
        # Load protein cache
        cache_path = os.path.join(self.models_dir, "protein_cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.prot_cache = pickle.load(f)
            print(f"✓ Protein embeddings cache loaded ({len(self.prot_cache)} sequences)")
        else:
            print("✗ Protein cache not found - will compute embeddings on demand")
        
        # Initialize protein model for on-demand embedding
        if len(self.prot_cache) == 0 or self.fusion_model is not None:
            print("Loading protein model for embeddings...")
            self.prot_tok = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
            self.prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME).to(DEVICE)
            for p in self.prot_model.parameters():
                p.requires_grad = False
            print("✓ Protein model loaded")
    
    def smiles_to_ecfp(self, smiles: str):
        """Convert SMILES to ECFP fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(ECFP_BITS, dtype=np.float32)
        
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
        fp = generator.GetFingerprint(mol)
        
        arr = np.zeros((ECFP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    
    @torch.no_grad()
    def embed_protein(self, sequence: str):
        """Get protein embedding"""
        if sequence in self.prot_cache:
            return self.prot_cache[sequence]
        
        if self.prot_model is None:
            raise ValueError("Protein model not loaded and sequence not in cache")
        
        toks = self.prot_tok(sequence, padding='max_length', truncation=True, 
                           max_length=1024, return_tensors='pt')
        ids = toks['input_ids'].to(DEVICE)
        mask = toks['attention_mask'].to(DEVICE)
        out = self.prot_model(input_ids=ids, attention_mask=mask).last_hidden_state
        
        # Mean pool with mask
        emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        # Cache for future use
        self.prot_cache[sequence] = emb
        return emb
    
    def predict(self, sequence: str, smiles: str):
        """
        Make binding prediction for a protein-ligand pair
        
        Returns:
            dict with keys: baseline_prob, fusion_prob, fusion_prob_calibrated
        """
        # Get ECFP fingerprint
        ecfp = self.smiles_to_ecfp(smiles)
        
        results = {}
        
        # Baseline prediction
        if self.baseline_model is not None:
            baseline_prob = float(self.baseline_model.predict_proba(ecfp[None, :])[:, 1][0])
            results['baseline_prob'] = baseline_prob
        
        # Fusion prediction
        if self.fusion_model is not None:
            prot_emb = self.embed_protein(sequence)
            fused_features = np.concatenate([ecfp, prot_emb], axis=0)[None, :].astype(np.float32)
            
            with torch.no_grad():
                x_tensor = torch.from_numpy(fused_features).to(DEVICE)
                logits, _ = self.fusion_model(x_tensor)
                
                # Uncalibrated probability
                fusion_prob = float(torch.sigmoid(logits).item())
                results['fusion_prob'] = fusion_prob
                
                # Calibrated probability
                if self.scaler is not None:
                    calibrated_logits = self.scaler(logits)
                    fusion_prob_cal = float(torch.sigmoid(calibrated_logits).item())
                    results['fusion_prob_calibrated'] = fusion_prob_cal
                else:
                    results['fusion_prob_calibrated'] = fusion_prob
        
        return results
    
    def batch_predict(self, sequences, smiles_list):
        """
        Make predictions for multiple protein-ligand pairs
        
        Args:
            sequences: list of protein sequences
            smiles_list: list of SMILES strings
            
        Returns:
            list of prediction dictionaries
        """
        if len(sequences) != len(smiles_list):
            raise ValueError("Number of sequences must match number of SMILES")
        
        results = []
        for seq, smi in zip(sequences, smiles_list):
            pred = self.predict(seq, smi)
            results.append(pred)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BindingPredictor()
    
    # Example protein sequence (SRC kinase)
    example_sequence = "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"
    
    # Example SMILES (kinase inhibitor)
    example_smiles = "CCN(CC)CCN(C)C(=O)C1=CC=C(NC2=NC(=NC=C2)N3CCN(C)CC3)C=C1"
    
    print("\nExample prediction:")
    print(f"Sequence length: {len(example_sequence)}")
    print(f"SMILES: {example_smiles}")
    
    prediction = predictor.predict(example_sequence, example_smiles)
    
    print("\nPrediction results:")
    for key, value in prediction.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\nPredicted binding: {'YES' if prediction.get('fusion_prob_calibrated', 0) > 0.5 else 'NO'}")
