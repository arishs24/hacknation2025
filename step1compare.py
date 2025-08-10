"""
Baseline Model Comparison Tool
Tests multiple models on ECFP4 features and finds the best performer
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
import pickle
import os

# ECFP4 molecular fingerprints
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# ---- Configuration Constants (shared with train.py)
SEED = 42
ECFP_BITS = 2048
ECFP_RADIUS = 2
TRAIN_CSV = "bindingdb_kinase_top10_train.csv"
VAL_CSV = "bindingdb_kinase_top10_val.csv"


def smiles_to_ecfp(smiles: str, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    """
    Convert SMILES string to ECFP4 fingerprint
    
    Args:
        smiles: SMILES string
        n_bits: Number of bits in fingerprint
        radius: Radius for Morgan fingerprint (2 for ECFP4)
    
    Returns:
        np.array: ECFP fingerprint as numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # empty vector for invalid SMILES
        return np.zeros(n_bits, dtype=np.float32)
    
    # Use the modern MorganGenerator approach
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def get_baseline_models():
    """
    Define all baseline models to test
    
    Returns:
        dict: Dictionary of model name -> model object
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=500, 
            solver="lbfgs", 
            random_state=SEED
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=SEED
        ),
        'SVM_Linear': SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=SEED
        ),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    return models


def evaluate_models(X_train, y_train, X_val, y_val, models, verbose=True):
    """
    Core function to train and evaluate multiple models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        models: Dictionary of models to test
        verbose: Whether to print progress
    
    Returns:
        dict: Results dictionary with metrics for each model
    """
    results = {}
    
    if verbose:
        print(f"{'Model':<18} {'AUROC':<8} {'AUPRC':<8} {'Accuracy':<10} {'F1':<8} {'Brier':<8} {'Status'}")
        print("-" * 80)
    
    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'auroc': roc_auc_score(y_val, val_proba),
                'auprc': average_precision_score(y_val, val_proba),
                'accuracy': accuracy_score(y_val, val_pred),
                'f1': f1_score(y_val, val_pred),
                'brier': brier_score_loss(y_val, val_proba),
                'model': model,
                'predictions': val_proba
            }
            
            results[model_name] = metrics
            
            if verbose:
                print(f"{model_name:<18} {metrics['auroc']:<8.3f} {metrics['auprc']:<8.3f} "
                      f"{metrics['accuracy']:<10.3f} {metrics['f1']:<8.3f} {metrics['brier']:<8.3f} ✓")
            
        except Exception as e:
            if verbose:
                print(f"{model_name:<18} {'---':<8} {'---':<8} {'---':<10} {'---':<8} {'---':<8} ✗ {str(e)[:20]}...")
            continue
    
    if verbose:
        print("-" * 80)
    
    return results


def find_best_model(results, metric='auroc', verbose=True):
    """
    Find and display the best model based on specified metric
    
    Args:
        results: Results dictionary from evaluate_models
        metric: Metric to optimize ('auroc', 'accuracy', 'f1', 'auprc')
        verbose: Whether to print results
    
    Returns:
        tuple: (best_model, model_name, best_metrics)
    """
    if not results:
        return None, None, None
    
    # Find best models by different metrics
    best_by_metric = {
        'auroc': max(results.items(), key=lambda x: x[1]['auroc']),
        'accuracy': max(results.items(), key=lambda x: x[1]['accuracy']),
        'f1': max(results.items(), key=lambda x: x[1]['f1']),
        'auprc': max(results.items(), key=lambda x: x[1]['auprc'])
    }
    
    if verbose:
        print("\n🏆 BEST MODELS BY METRIC:")
        for met, (name, metrics) in best_by_metric.items():
            print(f"  {met.upper():<8}: {name:<18} ({metrics[met]:.3f})")
    
    # Select best by specified metric
    best_name, best_metrics = best_by_metric[metric]
    
    if verbose:
        print(f"\n🎯 SELECTED BEST MODEL (by {metric.upper()}): {best_name}")
        print(f"   AUROC: {best_metrics['auroc']:.3f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
        print(f"   F1: {best_metrics['f1']:.3f}")
        print(f"   Brier: {best_metrics['brier']:.3f}")
    
    return best_metrics['model'], best_name, best_metrics


def save_comparison_results(results, filename='baseline_model_comparison.csv', sort_by='auroc'):
    """
    Save model comparison results to CSV
    
    Args:
        results: Results dictionary from evaluate_models
        filename: Output CSV filename
        sort_by: Metric to sort by
    
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    if not results:
        return None
    
    comparison_df = pd.DataFrame({
        'model': list(results.keys()),
        'auroc': [results[k]['auroc'] for k in results.keys()],
        'auprc': [results[k]['auprc'] for k in results.keys()],
        'accuracy': [results[k]['accuracy'] for k in results.keys()],
        'f1': [results[k]['f1'] for k in results.keys()],
        'brier': [results[k]['brier'] for k in results.keys()]
    }).sort_values(sort_by, ascending=False)
    
    comparison_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    return comparison_df


def test_baseline_models(save_results=True, save_best_model=True):
    """
    Test multiple baseline models using data from CSV files
    
    Args:
        save_results (bool): Save comparison results to CSV
        save_best_model (bool): Save the best model to disk
    
    Returns:
        dict: Results dictionary with model performance and best model info
    """
    print("🔬 BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    
    for df in (train_df, val_df):
        df["sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True)
        df["smiles"] = df["smiles"].astype(str).str.strip()
    
    # Generate ECFP features
    print("Generating ECFP4 features...")
    X_train_ecfp = np.stack([smiles_to_ecfp(s) for s in train_df["smiles"].tolist()], axis=0)
    y_train = train_df["label"].astype(int).to_numpy()
    
    X_val_ecfp = np.stack([smiles_to_ecfp(s) for s in val_df["smiles"].tolist()], axis=0)
    y_val = val_df["label"].astype(int).to_numpy()
    
    # Run the comparison
    return run_step1_comparison_external(X_train_ecfp, y_train, X_val_ecfp, y_val, 
                                       save_results=save_results, save_best_model=save_best_model, 
                                       return_dict=True)


def run_step1_comparison_external(X_train, y_train, X_val, y_val, metric='auroc', 
                                save_results=False, save_best_model=False, return_dict=False):
    """
    External interface for step1 comparison - takes features directly
    
    Args:
        X_train: Training ECFP4 features
        y_train: Training labels  
        X_val: Validation ECFP4 features
        y_val: Validation labels
        metric: Metric to optimize ('auroc', 'accuracy', 'f1', 'auprc')
        save_results: Whether to save comparison CSV
        save_best_model: Whether to save best model to disk
        return_dict: Whether to return full results dict (for test_baseline_models)
    
    Returns:
        tuple: (best_model, model_name, predictions) or dict if return_dict=True
    """
    print("\n🔬 STEP 1: BASELINE MODEL COMPARISON")
    print("=" * 80)
    
    # Get models and evaluate them
    models = get_baseline_models()
    results = evaluate_models(X_train, y_train, X_val, y_val, models, verbose=True)
    
    if not results:
        print("❌ No models were successfully trained! Falling back to LogisticRegression...")
        fallback_model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
        fallback_model.fit(X_train, y_train)
        fallback_predictions = fallback_model.predict_proba(X_val)[:, 1]
        
        if return_dict:
            return None  # No successful results
        return fallback_model, "LogisticRegression", fallback_predictions
    
    # Find best model
    best_model, best_name, best_metrics = find_best_model(results, metric=metric, verbose=True)
    
    # Save results if requested
    if save_results:
        save_comparison_results(results, sort_by=metric)
    
    # Save best model if requested
    if save_best_model:
        os.makedirs("saved_models", exist_ok=True)
        with open(f"saved_models/best_baseline_{best_name.lower()}.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print(f"💾 Best model saved to: saved_models/best_baseline_{best_name.lower()}.pkl")
    
    # Return format depends on use case
    if return_dict:
        # For test_baseline_models compatibility
        return {
            'results': results,
            'best_model_name': best_name,
            'best_model': best_model,
            'best_metrics': {k: v for k, v in best_metrics.items() if k != 'model'},
            'comparison_df': save_comparison_results(results, sort_by=metric) if save_results else None
        }
    else:
        # For train.py integration
        return best_model, best_name, best_metrics['predictions']


def get_best_baseline_model():
    """
    Quick function to get the best baseline model without saving results
    Returns the model object that can be used in place of logistic regression
    """
    results = test_baseline_models(save_results=False, save_best_model=False)
    if results:
        return results['best_model'], results['best_model_name']
    else:
        # Fallback to logistic regression
        print("Falling back to LogisticRegression...")
        return LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED), "LogisticRegression"


if __name__ == "__main__":
    # Run the comparison
    results = test_baseline_models()
    if results:
        print(f"\n Best model: {results['best_model_name']}")
        print("Use this in your main script:")
        print("from step1compare import run_step1_comparison_external")
        print("best_model, model_name, predictions = run_step1_comparison_external(X_train, y_train, X_val, y_val)")
