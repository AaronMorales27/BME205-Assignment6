import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.decomposition import PCA


# --- Load datasets ---
print("Loading Datasets...")
df_train = pd.read_csv("train.csv") # 18,702 samples × 5,461 features
df_test = pd.read_csv("test.csv") # 4,676 samples × 5,461 features

### One Hot Encoding ###
print("One Hot Encoding Datasets...")
# --- Initialize encoder ---
enc = OneHotEncoder(drop='first', sparse_output=False)

# --- Fit on training data ---
encoded_train = enc.fit_transform(df_train[['predicted_ancestry']])
encoded_test = enc.transform(df_test[['predicted_ancestry']])  # use same encoder!

# --- Convert encoded arrays to DataFrames ---
df_train_enc = pd.DataFrame(
    encoded_train,
    columns=enc.get_feature_names_out(['predicted_ancestry']),
    index=df_train.index
)
df_test_enc = pd.DataFrame(
    encoded_test,
    columns=enc.get_feature_names_out(['predicted_ancestry']),
    index=df_test.index
)

# --- Concatenate with the original data (and drop the original categorical column) ---
df_train_final = pd.concat(
    [df_train.drop(columns=['predicted_ancestry']), df_train_enc],
    axis=1
)
df_test_final = pd.concat(
    [df_test.drop(columns=['predicted_ancestry']), df_test_enc],
    axis=1
)

print('Assigning X/y split')
# --- Separate features (X) and target (y) ---
X = df_train_final.drop(columns=['breast_cancer'])
y = df_train_final['breast_cancer']

def plot_roc_curves(models, X_val, y_val):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_val)[:,1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        auc = roc_auc_score(y_val, y_pred_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig('my_plot.png')
    plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ===============================
# PARAMETERS
# ===============================
random_state = 42
cv_folds = 5
# For Logistic/SGD, PCA components can be ignored if not using PCA here
# n_components = None  

# ===============================
# FEATURE SELECTION
# ===============================
aucs = X.apply(lambda col: roc_auc_score(y, col), axis=0)
top_features = aucs[aucs >= 0.52].index.tolist() # 0.52 sweet spot
print(f'Number of features: {len(top_features)}')
X_sel = X[top_features]

# Ensure y is 1D
y_vec = y.values.ravel()

# ===============================
# CROSS-VALIDATION SETUP
# ===============================
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

# ===============================
# PIPELINES
# ===============================

# Logistic Regression Pipeline
pipe_logreg = Pipeline([
    ('scale', StandardScaler()),
    ('logreg', LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=random_state
    ))
])

# SGD Classifier Pipeline
pipe_sgd = Pipeline([
    ('scale', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=random_state
    ))
])

# Random Forest (no scaling needed)
rf = RandomForestClassifier(
    class_weight='balanced',
    n_jobs=1,
    random_state=random_state
)   

# ===============================
# HYPERPARAMETER GRIDS
# ===============================
param_grid_logreg = {'logreg__C': [0.01, 0.1, 0.5, 1, 2, 5]}
param_grid_sgd = {'sgd__alpha': [1e-4, 1e-3, 1e-2]}
param_grid_rf = {
    'n_estimators': [200, 300], # 500?
    'max_depth': [10, 15], # 20?
    'min_samples_leaf': [2, 5] # 1?
}

# ===============================
# GRID SEARCH + CROSS-VALIDATION
# ===============================
''''''
print("Tuning Logistic Regression...")
grid_logreg = GridSearchCV(pipe_logreg, param_grid_logreg, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_logreg.fit(X_sel, y_vec)
best_logreg = grid_logreg.best_estimator_
print("Best Logistic Regression params:", grid_logreg.best_params_)
# Best Logistic Regression params: {'logreg__C': 2}
'''
print("Tuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_rf.fit(X_sel, y_vec)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)
# Best RF params: {'max_depth': 15, 'min_samples_leaf': 5, 'n_estimators': 300}
'''

# ===============================
# CROSS-VALIDATION AUC COMPARISON
# ===============================
for name, model in [
    ('Logistic Regression', best_logreg)
    # ('SGD Classifier', best_sgd),
    # ('Random Forest', best_rf)
]:
    aucs = cross_val_score(model, X_sel, y_vec, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"\n{name}:")
    print(f"  AUC mean ± std: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"  Tested with top {len(top_features)} features ({cv_folds}-fold CV)")
