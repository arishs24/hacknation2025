"""
Final Model Testing Script
Tests the complete pipeline (baseline + fusion + calibrator) on the test set
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Fix numpy compatibility issue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set numpy to use legacy behavior to avoid _slice import error
try:
    np._NoValue = np._NoValue
except AttributeError:
    np._NoValue = object()

# Import shared utilities
try:
    from step1compare import smiles_to_ecfp, SEED, ECFP_BITS, ECFP_RADIUS
except ImportError:
    print("Warning: step1compare not found, using fallback constants")
    SEED = 42
    ECFP_BITS = 2048
    ECFP_RADIUS = 2
    
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    
    def smiles_to_ecfp(smiles: str, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)

# Configuration
TEST_CSV = "bindingdb_kinase_top10_test.csv"  # Test set CSV
MODELS_DIR = "saved_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)

class FusionMLP(nn.Module):
    """Original Fusion MLP architecture for loading saved models"""
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
    """Temperature scaling for calibration"""
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

def load_protein_embeddings():
    """Load pre-computed protein embeddings"""
    cache_path = os.path.join(MODELS_DIR, "protein_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"✅ Loaded protein embeddings cache: {len(cache)} sequences")
        return cache
    else:
        print("❌ Protein cache not found - will compute embeddings on the fly")
        return {}

def embed_protein_on_fly(seq: str):
    """Compute protein embedding on the fly if not cached"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
        model = AutoModel.from_pretrained(PROT_MODEL_NAME).to(DEVICE)
        
        with torch.no_grad():
            tokens = tokenizer(seq, padding='max_length', truncation=True, 
                             max_length=1024, return_tensors='pt')
            ids = tokens['input_ids'].to(DEVICE)
            mask = tokens['attention_mask'].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask).last_hidden_state
            emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Warning: Could not compute protein embedding: {e}")
        return np.zeros(320, dtype=np.float32)

def load_models():
    """Load all trained models"""
    models = {}
    
    # 1. Load baseline model
    baseline_path = os.path.join(MODELS_DIR, "baseline_logreg.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            models['baseline'] = pickle.load(f)
        print("✅ Loaded baseline model")
    else:
        print("❌ Baseline model not found")
        return None
    
    # 2. Load fusion model
    fusion_path = os.path.join(MODELS_DIR, "fusion_mlp.pth")
    if os.path.exists(fusion_path):
        checkpoint = torch.load(fusion_path, map_location=DEVICE)
        config = checkpoint['model_config']
        
        fusion_model = FusionMLP(
            in_dim=config['in_dim'], 
            hidden=config.get('hidden', 256), 
            reg_head=config.get('reg_head', False)
        ).to(DEVICE)
        
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.eval()
        models['fusion'] = fusion_model
        print("✅ Loaded fusion model")
    else:
        print("❌ Fusion model not found")
        return None
    
    # 3. Load temperature scaler
    scaler_path = os.path.join(MODELS_DIR, "temperature_scaler.pth")
    if os.path.exists(scaler_path):
        scaler_checkpoint = torch.load(scaler_path, map_location=DEVICE)
        scaler = TemperatureScaler()
        scaler.load_state_dict(scaler_checkpoint['scaler_state_dict'])
        scaler.eval()
        models['scaler'] = scaler
        print(f"✅ Loaded temperature scaler (T={scaler_checkpoint['temperature']:.3f})")
    else:
        print("❌ Temperature scaler not found")
        return None
    
    return models

def prepare_test_data(test_df, protein_cache):
    """Prepare test data for evaluation"""
    print("🔄 Preparing test data...")
    
    # Generate ECFP features
    print("  Computing ECFP4 fingerprints...")
    X_test_ecfp = np.stack([smiles_to_ecfp(s) for s in test_df["smiles"].tolist()], axis=0)
    
    # Get protein embeddings
    print("  Getting protein embeddings...")
    X_test_protein = []
    for seq in test_df["sequence"].tolist():
        if seq in protein_cache:
            X_test_protein.append(protein_cache[seq])
        else:
            print(f"  Computing embedding for sequence (length {len(seq)})...")
            emb = embed_protein_on_fly(seq)
            X_test_protein.append(emb)
    
    X_test_protein = np.stack(X_test_protein, axis=0)
    
    # Combine features for fusion model
    X_test_fusion = np.concatenate([X_test_ecfp, X_test_protein], axis=1).astype(np.float32)
    
    # Get labels
    y_test = test_df["label"].astype(int).to_numpy()
    
    print(f"  Test data shape: ECFP {X_test_ecfp.shape}, Protein {X_test_protein.shape}, Fusion {X_test_fusion.shape}")
    print(f"  Test labels: {len(y_test)} samples, {sum(y_test)} active ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    return X_test_ecfp, X_test_fusion, y_test

def evaluate_models(models, X_test_ecfp, X_test_fusion, y_test):
    """Evaluate all models and return predictions"""
    print("\n🧪 Evaluating models on test set...")
    
    results = {}
    
    # 1. Baseline model evaluation
    print("  Testing baseline model...")
    baseline_proba = models['baseline'].predict_proba(X_test_ecfp)[:, 1]
    baseline_pred = (baseline_proba >= 0.5).astype(int)
    
    results['baseline'] = {
        'predictions': baseline_pred,
        'probabilities': baseline_proba,
        'auroc': roc_auc_score(y_test, baseline_proba),
        'auprc': average_precision_score(y_test, baseline_proba),
        'accuracy': accuracy_score(y_test, baseline_pred),
        'f1': f1_score(y_test, baseline_pred),
        'brier': brier_score_loss(y_test, baseline_proba)
    }
    
    # 2. Fusion model evaluation (uncalibrated)
    print("  Testing fusion model...")
    with torch.no_grad():
        X_fusion_tensor = torch.from_numpy(X_test_fusion).to(DEVICE)
        fusion_logits, _ = models['fusion'](X_fusion_tensor)
        fusion_proba_uncal = torch.sigmoid(fusion_logits).cpu().numpy()
        fusion_pred_uncal = (fusion_proba_uncal >= 0.5).astype(int)
    
    results['fusion_uncalibrated'] = {
        'predictions': fusion_pred_uncal,
        'probabilities': fusion_proba_uncal,
        'logits': fusion_logits.cpu().numpy(),
        'auroc': roc_auc_score(y_test, fusion_proba_uncal),
        'auprc': average_precision_score(y_test, fusion_proba_uncal),
        'accuracy': accuracy_score(y_test, fusion_pred_uncal),
        'f1': f1_score(y_test, fusion_pred_uncal),
        'brier': brier_score_loss(y_test, fusion_proba_uncal)
    }
    
    # 3. Fusion model evaluation (calibrated)
    print("  Testing calibrated fusion model...")
    with torch.no_grad():
        calibrated_logits = models['scaler'](fusion_logits)
        fusion_proba_cal = torch.sigmoid(calibrated_logits).cpu().numpy()
        fusion_pred_cal = (fusion_proba_cal >= 0.5).astype(int)
    
    results['fusion_calibrated'] = {
        'predictions': fusion_pred_cal,
        'probabilities': fusion_proba_cal,
        'logits': calibrated_logits.cpu().numpy(),
        'auroc': roc_auc_score(y_test, fusion_proba_cal),
        'auprc': average_precision_score(y_test, fusion_proba_cal),
        'accuracy': accuracy_score(y_test, fusion_pred_cal),
        'f1': f1_score(y_test, fusion_pred_cal),
        'brier': brier_score_loss(y_test, fusion_proba_cal)
    }
    
    return results

def print_results(results, y_test):
    """Print comprehensive evaluation results"""
    print("\n" + "="*80)
    print("🏆 FINAL MODEL EVALUATION RESULTS")
    print("="*80)
    
    print(f"{'Model':<20} {'AUROC':<8} {'AUPRC':<8} {'Accuracy':<10} {'F1':<8} {'Brier':<8}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['auroc']:<8.4f} "
              f"{metrics['auprc']:<8.4f} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<8.4f} "
              f"{metrics['brier']:<8.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['auroc'])
    print(f"\n🎯 Best model by AUROC: {best_model[0]} ({best_model[1]['auroc']:.4f})")
    
    # Print detailed results for best model
    print(f"\n📊 Detailed Results for {best_model[0]}:")
    print(f"   AUROC: {best_model[1]['auroc']:.4f}")
    print(f"   AUPRC: {best_model[1]['auprc']:.4f}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"   F1 Score: {best_model[1]['f1']:.4f}")
    print(f"   Brier Score: {best_model[1]['brier']:.4f}")
    
    # Confusion matrix for best model
    print(f"\n📈 Confusion Matrix for {best_model[0]}:")
    cm = confusion_matrix(y_test, best_model[1]['predictions'])
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # Classification report
    print(f"\n📋 Classification Report for {best_model[0]}:")
    print(classification_report(y_test, best_model[1]['predictions'], 
                              target_names=['Inactive', 'Active']))

def create_visualizations(results, y_test, test_df):
    """Create comprehensive visualizations"""
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. ROC Curves
    from sklearn.metrics import roc_curve
    
    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
        axes[0,0].plot(fpr, tpr, label=f"{model_name} (AUC={metrics['auroc']:.3f})", linewidth=2)
    
    axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curves')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    from sklearn.metrics import precision_recall_curve
    
    for model_name, metrics in results.items():
        precision, recall, _ = precision_recall_curve(y_test, metrics['probabilities'])
        axes[0,1].plot(recall, precision, label=f"{model_name} (AP={metrics['auprc']:.3f})", linewidth=2)
    
    baseline_precision = sum(y_test) / len(y_test)
    axes[0,1].axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.6, 
                     label=f'Random ({baseline_precision:.3f})')
    axes[0,1].set_xlabel('Recall')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].set_title('Precision-Recall Curves')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Model Comparison Bar Chart
    metrics_names = ['AUROC', 'AUPRC', 'Accuracy', 'F1']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics['auroc'], metrics['auprc'], metrics['accuracy'], metrics['f1']]
        axes[0,2].bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    axes[0,2].set_xlabel('Metrics')
    axes[0,2].set_ylabel('Score')
    axes[0,2].set_title('Model Performance Comparison')
    axes[0,2].set_xticks(x + width)
    axes[0,2].set_xticklabels(metrics_names)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3, axis='y')
    
    # 4. Probability Distributions
    best_model_name = max(results.items(), key=lambda x: x[1]['auroc'])[0]
    best_probs = results[best_model_name]['probabilities']
    
    active_probs = best_probs[y_test == 1]
    inactive_probs = best_probs[y_test == 0]
    
    axes[1,0].hist(inactive_probs, bins=30, alpha=0.7, label=f'Inactive (n={len(inactive_probs)})', 
                  color='red', edgecolor='black')
    axes[1,0].hist(active_probs, bins=30, alpha=0.7, label=f'Active (n={len(active_probs)})', 
                  color='green', edgecolor='black')
    axes[1,0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title(f'Probability Distribution - {best_model_name}')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Confusion Matrices
    for i, (model_name, metrics) in enumerate(list(results.items())[:2]):  # Show first 2 models
        cm = confusion_matrix(y_test, metrics['predictions'])
        axes[1,1+i].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[1,1+i].set_title(f'Confusion Matrix - {model_name}')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                axes[1,1+i].text(col, row, format(cm[row, col], 'd'),
                                ha="center", va="center",
                                color="white" if cm[row, col] > thresh else "black")
        
        axes[1,1+i].set_ylabel('True Label')
        axes[1,1+i].set_xlabel('Predicted Label')
        axes[1,1+i].set_xticks([0, 1])
        axes[1,1+i].set_yticks([0, 1])
        axes[1,1+i].set_xticklabels(['Inactive', 'Active'])
        axes[1,1+i].set_yticklabels(['Inactive', 'Active'])
    
    plt.tight_layout()
    plt.savefig('final_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Visualizations saved as 'final_model_evaluation.png'")

def save_test_results(results, test_df, y_test):
    """Save detailed test results to CSV"""
    print("\n💾 Saving test results...")
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['true_label'] = y_test
    
    # Add predictions from all models
    for model_name, metrics in results.items():
        results_df[f'{model_name}_prediction'] = metrics['predictions']
        results_df[f'{model_name}_probability'] = metrics['probabilities']
    
    # Save to CSV
    results_df.to_csv('test_results_detailed.csv', index=False)
    print("📄 Detailed results saved to 'test_results_detailed.csv'")
    
    # Save summary statistics
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'model': model_name,
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'brier': metrics['brier']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('test_results_summary.csv', index=False)
    print("📊 Summary results saved to 'test_results_summary.csv'")

def main():
    """Main evaluation function"""
    print("🚀 Starting Final Model Evaluation")
    print("="*50)
    
    # Load test data
    if not os.path.exists(TEST_CSV):
        print(f"❌ Test CSV not found: {TEST_CSV}")
        print("Please ensure you have run the data processing notebook to create the test set.")
        return
    
    test_df = pd.read_csv(TEST_CSV)
    print(f"📄 Loaded test set: {len(test_df)} samples")
    
    # Clean test data
    test_df["sequence"] = test_df["sequence"].astype(str).str.replace(r"\s+", "", regex=True)
    test_df["smiles"] = test_df["smiles"].astype(str).str.strip()
    
    # Load models
    models = load_models()
    if models is None:
        print("❌ Failed to load models. Please ensure you have trained models in the saved_models directory.")
        return
    
    # Load protein embeddings
    protein_cache = load_protein_embeddings()
    
    # Prepare test data
    X_test_ecfp, X_test_fusion, y_test = prepare_test_data(test_df, protein_cache)
    
    # Evaluate models
    results = evaluate_models(models, X_test_ecfp, X_test_fusion, y_test)
    
    # Print results
    print_results(results, y_test)
    
    # Create visualizations
    create_visualizations(results, y_test, test_df)
    
    # Save results
    save_test_results(results, test_df, y_test)
    
    print("\n✅ Final model evaluation completed!")
    print("📁 Check the following files for detailed results:")
    print("   - final_model_evaluation.png (visualizations)")
    print("   - test_results_detailed.csv (per-sample predictions)")
    print("   - test_results_summary.csv (summary metrics)")

if __name__ == "__main__":
    main()
