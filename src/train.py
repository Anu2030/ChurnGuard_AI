import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def train_and_evaluate():
    print("Loading preprocessed data...")
    df = pd.read_csv(config.PREPROCESSED_DATA_PATH)
    
    # Split features (X) and target (y)
    X = df.drop(config.TARGET_COL, axis=1)
    y = df[config.TARGET_COL]
    
    # Stratified split to maintain churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    
    # Calculate scale_pos_weight for XGBoost
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_class_count / pos_class_count
    
    # Initialize models
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight='balanced'),
            'tune': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            'tune': True,
            'grid': config.GRID_PARAMS_RF
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_STATE, scale_pos_weight=scale_pos_weight_val),
            'tune': True,
            'grid': config.GRID_PARAMS_XGB
        }
    }
    
    results = {}
    best_model = None
    best_auc = 0
    best_name = ""
    
    for name, config_dict in models.items():
        print(f"\nTraining {name}...")
        model_obj = config_dict['model']
        
        if config_dict['tune']:
            print(f"Running GridSearchCV for {name}...")
            grid_search = GridSearchCV(
                estimator=model_obj,
                param_grid=config_dict['grid'],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            fitted_model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            fitted_model = model_obj.fit(X_train, y_train)
            
        # Predict
        preds = fitted_model.predict(X_test)
        probs = fitted_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        # Save results
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1': float(f1),
            'ROC-AUC': float(auc)
        }
        
        print(f"{name} Performance:")
        print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        # Track best model based on ROC-AUC
        if auc > best_auc:
            best_auc = auc
            best_model = fitted_model
            best_name = name

    print(f"\nBest Performing Model: {best_name} (ROC-AUC: {best_auc:.4f})")
    
    # Save the best model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    print(f"Best model saved to {config.BEST_MODEL_PATH}")
    
    # Run detailed evaluation on best model
    best_preds = best_model.predict(X_test)
    best_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Create evaluation plot directory in app or notebooks
    plots_dir = os.path.join(config.BASE_DIR, 'app', 'static_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Churn', 'Churn'], yticklabels=['Non-Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    
    # Save test set and features list for the simulator
    X_test_with_target = X_test.copy()
    X_test_with_target[config.TARGET_COL] = y_test
    X_test_with_target.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    
    # Save column features order
    features_list = list(X.columns)
    with open(os.path.join(config.MODELS_DIR, 'features.json'), 'w') as f:
        json.dump(features_list, f)
        
    # Save metric outputs
    metrics_path = os.path.join(config.MODELS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'best_model_name': best_name, 'all_results': results}, f, indent=4)
        
    # Generate and save SHAP Explainer
    print("Generating SHAP Explainer...")
    if best_name in ['XGBoost', 'Random Forest']:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.Explainer(best_model, X_train)
        
    explainer_path = os.path.join(config.MODELS_DIR, 'shap_explainer.pkl')
    joblib.dump(explainer, explainer_path)
    print(f"SHAP explainer saved to {explainer_path}")
        
    print("Model training and evaluation successfully completed!")

if __name__ == "__main__":
    train_and_evaluate()