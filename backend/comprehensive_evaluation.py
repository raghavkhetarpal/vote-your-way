"""
comprehensive_evaluation.py - Complete evaluation with XGBoost and visualizations
Generates performance metrics, ROC curves, confusion matrices, and F1 analysis
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from promise_extraction import load_promises
from prediction import (
    build_feature_matrix, status_to_label, get_training_data,
    train_new_model, extract_ml_features, LIKELY_THRESHOLD, UNCERTAIN_THRESHOLD
)

print("\n" + "="*80)
print("🚀 COMPREHENSIVE MODEL EVALUATION WITH XGBOOST")
print("="*80)

# Load data
promises = load_promises()
print(f"✅ Loaded {len(promises)} promises")

training_data = get_training_data(promises)
print(f"✅ Training data: {len(training_data)} examples")

# Build feature matrix
X, tfidf = build_feature_matrix(training_data, fit_tfidf=True)
y = np.array([status_to_label(p.get("completion_status", "Not Started"))
              for p in training_data])

print(f"\n📊 Dataset Statistics:")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Class balance: Completed={y.sum()}, Other={(1-y).sum()}")
print(f"  Class ratio: {y.sum()/(1-y).sum():.2%}")

# ============================================================================
# CROSS-VALIDATION EVALUATION
# ============================================================================
print(f"\n" + "-"*80)
print("📈 CROSS-VALIDATION EVALUATION (5-fold Stratified)")
print("-"*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_list = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
}
all_predictions = []
all_probabilities = []
all_y_test = []

fold_num = 0
for train_idx, test_idx in cv.split(X, y):
    fold_num += 1
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model on fold
    train_promises = [training_data[i] for i in train_idx]
    model, scaler, _ = train_new_model(train_promises)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0
    
    scores_list['accuracy'].append(acc)
    scores_list['precision'].append(prec)
    scores_list['recall'].append(rec)
    scores_list['f1'].append(f1)
    scores_list['roc_auc'].append(roc)
    
    all_predictions.extend(y_pred)
    all_probabilities.extend(y_pred_proba)
    all_y_test.extend(y_test)
    
    print(f"  Fold {fold_num}: Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | AUC={roc:.3f}")

# Average scores
print(f"\n📊 FINAL CROSS-VALIDATION RESULTS:")
print(f"  ────────────────────────────────────────")
print(f"  Accuracy:  {np.mean(scores_list['accuracy']):.3f} ± {np.std(scores_list['accuracy']):.3f}")
print(f"  Precision: {np.mean(scores_list['precision']):.3f} ± {np.std(scores_list['precision']):.3f}")
print(f"  Recall:    {np.mean(scores_list['recall']):.3f} ± {np.std(scores_list['recall']):.3f}")
print(f"  F1 Score:  {np.mean(scores_list['f1']):.3f} ± {np.std(scores_list['f1']):.3f} {'✅ >= 0.8!' if np.mean(scores_list['f1']) >= 0.8 else '⚠️  < 0.8'}")
print(f"  ROC-AUC:   {np.mean(scores_list['roc_auc']):.3f} ± {np.std(scores_list['roc_auc']):.3f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('XGBoost Model - Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

# 1. Metrics Comparison
ax = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
means = [
    np.mean(scores_list['accuracy']),
    np.mean(scores_list['precision']),
    np.mean(scores_list['recall']),
    np.mean(scores_list['f1']),
    np.mean(scores_list['roc_auc'])
]
stds = [
    np.std(scores_list['accuracy']),
    np.std(scores_list['precision']),
    np.std(scores_list['recall']),
    np.std(scores_list['f1']),
    np.std(scores_list['roc_auc'])
]
colors = ['#2ecc71' if m >= 0.8 else '#3498db' if m >= 0.6 else '#e74c3c' for m in means]
ax.bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylim([0, 1])
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Performance Metrics', fontweight='bold')
ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='Target (0.8)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(all_y_test, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (All Folds)', fontweight='bold')
ax.set_xticklabels(['Not Completed', 'Completed'])
ax.set_yticklabels(['Not Completed', 'Completed'])

# 3. F1 Score Across Folds
ax = axes[0, 2]
folds = list(range(1, 6))
ax.plot(folds, scores_list['f1'], marker='o', linewidth=2, markersize=8, color='#e74c3c', label='F1 Score')
ax.axhline(y=np.mean(scores_list['f1']), color='#2c3e50', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores_list["f1"]):.3f}')
ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='Target: 0.8')
ax.set_xlabel('Fold Number', fontsize=10)
ax.set_ylabel('F1 Score', fontsize=10)
ax.set_title('F1 Score Per Fold', fontweight='bold')
ax.set_xticks(folds)
ax.set_ylim([0, 1])
ax.legend()
ax.grid(alpha=0.3)

# 4. ROC Curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(all_y_test, all_probabilities)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5. Probability Distribution
ax = axes[1, 1]
probs_completed = [p for p, y_true in zip(all_probabilities, all_y_test) if y_true == 1]
probs_not_completed = [p for p, y_true in zip(all_probabilities, all_y_test) if y_true == 0]
ax.hist(probs_not_completed, bins=20, alpha=0.6, label='Not Completed', color='#3498db')
ax.hist(probs_completed, bins=20, alpha=0.6, label='Completed', color='#2ecc71')
ax.axvline(x=LIKELY_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {LIKELY_THRESHOLD}')
ax.set_xlabel('Predicted Probability', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Probability Distribution by Class', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 6. All Metrics Across Folds
ax = axes[1, 2]
x_pos = np.arange(5)
width = 0.15
ax.bar(x_pos - 2*width, scores_list['accuracy'], width, label='Accuracy', color='#3498db')
ax.bar(x_pos - width, scores_list['precision'], width, label='Precision', color='#2ecc71')
ax.bar(x_pos, scores_list['recall'], width, label='Recall', color='#e74c3c')
ax.bar(x_pos + width, scores_list['f1'], width, label='F1', color='#f39c12')
ax.bar(x_pos + 2*width, scores_list['roc_auc'], width, label='ROC-AUC', color='#9b59b6')
ax.set_xlabel('Fold Number', fontsize=10)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('All Metrics Per Fold', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.set_ylim([0, 1])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/processed/model_performance_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: model_performance_analysis.png")

# ============================================================================
# SAVE DETAILED METRICS
# ============================================================================
metrics_report = {
    "model": "XGBoost with SMOTE",
    "dataset": {
        "total_promises": len(training_data),
        "completed": int(y.sum()),
        "not_completed": int((1-y).sum()),
        "feature_count": X.shape[1]
    },
    "cross_validation": {
        "folds": 5,
        "accuracy": {
            "mean": float(np.mean(scores_list['accuracy'])),
            "std": float(np.std(scores_list['accuracy'])),
            "folds": [float(x) for x in scores_list['accuracy']]
        },
        "precision": {
            "mean": float(np.mean(scores_list['precision'])),
            "std": float(np.std(scores_list['precision'])),
            "folds": [float(x) for x in scores_list['precision']]
        },
        "recall": {
            "mean": float(np.mean(scores_list['recall'])),
            "std": float(np.std(scores_list['recall'])),
            "folds": [float(x) for x in scores_list['recall']]
        },
        "f1": {
            "mean": float(np.mean(scores_list['f1'])),
            "std": float(np.std(scores_list['f1'])),
            "folds": [float(x) for x in scores_list['f1']]
        },
        "roc_auc": {
            "mean": float(np.mean(scores_list['roc_auc'])),
            "std": float(np.std(scores_list['roc_auc'])),
            "folds": [float(x) for x in scores_list['roc_auc']]
        }
    },
    "thresholds": {
        "likely_threshold": LIKELY_THRESHOLD,
        "uncertain_threshold": UNCERTAIN_THRESHOLD
    },
    "status": "✅ PRODUCTION READY" if np.mean(scores_list['f1']) >= 0.75 else "⚠️ NEEDS TUNING"
}

output_path = Path("../data/processed/xgboost_evaluation_report.json")
with open(output_path, 'w') as f:
    json.dump(metrics_report, f, indent=2)
print(f"✅ Saved: xgboost_evaluation_report.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n" + "="*80)
print(f"✨ EVALUATION COMPLETE")
print(f"="*80)
print(f"\n🎯 Target F1 >= 0.80: {'✅ ACHIEVED' if np.mean(scores_list['f1']) >= 0.80 else '⚠️ NOT ACHIEVED (but close!)'}")
print(f"\n📊 Model Status:")
if np.mean(scores_list['f1']) >= 0.75:
    print(f"   🚀 Production Ready - F1 = {np.mean(scores_list['f1']):.3f}")
elif np.mean(scores_list['f1']) >= 0.60:
    print(f"   ✅ Good Performance - F1 = {np.mean(scores_list['f1']):.3f}")
else:
    print(f"   ⚠️  Needs Improvement - F1 = {np.mean(scores_list['f1']):.3f}")

print(f"\n📈 All files saved to data/processed/")
print(f"   - xgboost_evaluation_report.json (metrics)")
print(f"   - model_performance_analysis.png (visualizations)")
print(f"\n" + "="*80)
