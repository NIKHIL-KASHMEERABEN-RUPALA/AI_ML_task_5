import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import graphviz
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

print("HEART DISEASE PREDICTION - Decision Trees & Random Forests")
print("="*100)

heart = fetch_openml(name='heart', version=1, as_frame=True)
df = heart.frame

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df['target'].map({'present': 1, 'absent': 0})

print(f"Dataset Shape: {df.shape}")
print(f"Heart Disease Cases: {df['target'].sum()} / {len(df)} ({df['target'].mean():.2%})")
print("\nFirst 5 rows:")
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
X[cat_features] = X[cat_features].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

print("\n" + "="*50)
print("1. DECISION TREE CLASSIFIER")
print("="*50)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

dt_pruned = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=5, ccp_alpha=0.01, random_state=42
)
dt_pruned.fit(X_train, y_train)

print(f"Full Tree Depth       : {dt_full.get_depth()}")
print(f"Full Tree Leaves      : {dt_full.get_n_leaves()}")
print(f"Pruned Tree Depth     : {dt_pruned.get_depth()}")
print(f"Pruned Tree Leaves    : {dt_pruned.get_n_leaves()}")

acc_full = accuracy_score(y_test, dt_full.predict(X_test))
acc_pruned = accuracy_score(y_test, dt_pruned.predict(X_test))
print(f"Full Tree Accuracy    : {acc_full:.4f} ← Likely overfitted")
print(f"Pruned Tree Accuracy  : {acc_pruned:.4f} ← Better generalization")

plt.figure(figsize=(24, 14))
plot_tree(dt_pruned, 
          feature_names=X.columns,
          class_names=['No Disease', 'Heart Disease'],
          filled=True, 
          rounded=True, 
          fontsize=12,
          proportion=True,
          precision=2)
plt.title("Decision Tree for Heart Disease Prediction (Pruned - Max Depth = 4)", 
          fontsize=22, fontweight='bold', pad=20)
plt.show()

export_graphviz(dt_pruned, out_file='heart_tree.dot', 
                feature_names=X.columns, class_names=['No', 'Yes'],
                filled=True, rounded=True, special_characters=True)

print("\n" + "="*50)
print("2. RANDOM FOREST - Ensemble of 500 Trees")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print(f"OOB Score (built-in CV): {rf.oob_score_:.4f}")
print(f"Test Accuracy          : {accuracy_score(y_test, rf.predict(X_test)):.4f}")
print(f"Test ROC-AUC           : {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")

param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                           param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV AUC    : {grid_search.best_score_:.4f}")

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 Most Important Features for Heart Disease:")
for i in range(10):
    print(f"{i+1:2}. {X.columns[indices[i]]:20} → {importances[indices[i]]:.4f}")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
top_n = 10
colors = plt.cm.viridis(np.linspace(0, 1, top_n))
bars = ax1.barh(range(top_n), importances[indices[:top_n]][::-1], color=colors[::-1])
ax1.set_yticks(range(top_n))
ax1.set_yticklabels(X.columns[indices[:top_n]][::-1])
ax1.set_xlabel('Gini Importance')
ax1.set_title('Top 10 Feature Importance (Random Forest)', fontweight='bold', fontsize=16)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
models = {
    'Single Tree (Pruned)': dt_pruned,
    'Full Decision Tree': dt_full,
    'Random Forest (Best)': best_rf
}
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    ax2.plot(fpr, tpr, lw=3, label=f'{name} (AUC = {auc_score:.4f})')

ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve Comparison', fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, best_rf.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'], ax=ax3, annot_kws={"size": 18, "weight": "bold"})
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title('Confusion Matrix - Random Forest', fontweight='bold')

ax4 = fig.add_subplot(gs[1, 0])
depths = range(1, 21)
train_acc = []
test_acc = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    test_acc.append(accuracy_score(y_test, dt.predict(X_test)))

ax4.plot(depths, train_acc, 'o-', label='Training Accuracy', linewidth=3, markersize=8)
ax4.plot(depths, test_acc, 'o-', label='Test Accuracy', linewidth=3, markersize=8)
ax4.axvline(4, color='red', linestyle='--', label='Chosen Depth = 4')
ax4.set_xlabel('Max Tree Depth')
ax4.set_ylabel('Accuracy')
ax4.set_title('Overfitting Analysis: Tree Depth vs Accuracy', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.4)

from sklearn.inspection import PartialDependenceDisplay
ax
