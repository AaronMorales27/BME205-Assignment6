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

from sklearn.pipeline import Pipeline

from sklearn.cross_decomposition import PLSRegression

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
# New Train/Test Shapes
'''
    TRAIN shape (18702, 5465)
        1 id column
        5460 PRS columns
        1 target/label BRC column
        3 OHE columns
            ['EUR' 'EAS' 'AFR' 'AMR'], drop='first' AFR/baseline

    TEST shape (18702, 5464)
        1 id column
        5460 PRS columns
        3 OHE columns
            ['EUR' 'EAS' 'AFR' 'AMR'], drop='first' AFR/baseline

'''
# Verify OHE Results

### Train Test Split Test Data ###
print('Performing Train/Test Split...')
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


### PCA Exploration ###
# Explained variance ≠ predictive power.
# PCA maximizes variance, not necessarily class separability (e.g., breast cancer vs. control).
# ~95% variance at 2130 comps
# ~90% variance at 1500 comps
# ~85% variance at 1100 comps
# ~80% variance at 830  comps
# Standardize before PCA

'''
X_scaled = StandardScaler().fit_transform(X)

pca = PCA().fit(X_scaled)

# Get cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

# Print in increments (adjust step size as you wish)
step = 10
print(f"{'Components':<12}{'Explained Variance (%)':>25}")
print("-" * 37)
for i in range(0, len(cum_var), step):
    print(f"{i+1:<12}{cum_var[i]*100:>25.2f}")
'''

# ===============================
# CONFIG
# ===============================
top_k = 50                     # number of your top predictive features
n_components = 100   # <-- change this number to test 500, 1000, 1500, etc.
# AUC 0.70 at 300 comps
# AUC 0.72 at 800 comps
cv_folds = 5
random_state = 42

aucs = X.apply(lambda col: roc_auc_score(y, col), axis=0)
top_features = aucs[aucs >= 0.6].index.tolist() # indices of the top features + threshold
print(f'Number of features" {len(top_features)}')
X_sel = X[top_features]

# ===============================
# CROSS-VALIDATION SETUP
# ===============================
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

print('Log Reg Pipe...')
# ===============================
# LOGISTIC REGRESSION PIPELINE
# ===============================
pipe_logreg = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)),
    ('logreg', LogisticRegression(
        C=0.01,              # smaller C = stronger regularization
        max_iter=2000,
        solver='lbfgs',
        n_jobs=-1,
    ))
])
print('SGD Pipe...')
# ===============================
# SGD CLASSIFIER PIPELINE
# ===============================
pipe_sgd = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)),
    ('sgd', SGDClassifier(
        loss='log_loss',     # logistic regression objective
        penalty='l2',
        alpha=1e-4,          # regularization strength
        max_iter=1000,
        tol=1e-3,
        random_state=random_state
    ))
])
'''
# ======================================
# PIPELINE: PLS → Logistic Regression
# ======================================
pipe_pls_logreg = Pipeline([
    ('scale', StandardScaler()),
    ('pls', PLSRegression(n_components=n_components)),
    ('logreg', LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=0.5,
        random_state=random_state
    ))
])
'''
print('Cross Validation...')
# ===============================
# CROSS-VALIDATION AUC COMPARISON
# ===============================
# ('SGD Classifier', pipe_sgd)
# ('Logistic Regression', pipe_logreg)

y = y.values.ravel()   # ensures 1D array, avoids 3D propagation for

for name, model in [
    ('Logistic Regression', pipe_logreg),
    ('SGD Classifier', pipe_sgd)
    ]:
    aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1) # X_sel fix
    print(f"\n{name}:")
    print(f"  AUC mean ± std: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"  Tested with {n_components} principal components ({cv_folds}-fold CV)")


