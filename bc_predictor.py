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
'''
# --- Verify results ---
print("Train shape:", df_train_final.shape)
print("Test shape:", df_test_final.shape)
print(df_test_final.columns)
'''

### Train Test Split Test Data ###
print('Performing Train/Test Split...')
# --- Separate features (X) and target (y) ---
X = df_train_final.drop(columns=['breast_cancer'])
y = df_train_final['breast_cancer']

# --- Split into train/validation sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,      # 20% of training data goes to validation
    random_state=42,    # ensures reproducibility
    stratify=y          # keeps class proportions the same (recommended for classification)
)
# Train and Validation Shape(0.2)
'''
    Train Shape (14961, 5464) (14961,)
    Valid Shape (3741, 5464) (3741,)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)
'''
# Sanity Checks, Feature Analysis/selection

# '''
print("SANITY CHECKS")
print("shape:", X.shape)
print("y distribution:", np.bincount(y.astype(int)))
print("any nulls:", X.isnull().any().any())
print("y unique values:", np.unique(y))
print("baseline AUC (predict mean):", roc_auc_score(y, np.full(len(y), y.mean())))
# '''

### FEATURE SELECTION #


# ROC AUC for each PRS feature individually (treating each column as a single predictor)
# values 0.7 - 0.6 show moderate predictive power, most promising candidates for pred model
aucs = X.apply(lambda col: roc_auc_score(y, col), axis=0)
# print(aucs.sort_values(ascending=False).head(30))
top_features = aucs[aucs >= 0.6].index.tolist() # indices of the top features
print(f'Number of Features: {len(top_features)}')
# X_top = X[top_features] 
# mi = mutual_info_classif(X, y, discrete_features=False)
# print(pd.Series(mi, index=X.columns).sort_values(ascending=False).head(30))

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

### Feature Selection ###
# raw space FAR TOO NOISY, need feature selection!

### LINEAR EVALUATION ###

# Logistic Regression

# Logistic regression’s runtime grows roughly linearly with the number of features × samples.


print("Fitting Log. Reg...")
log_reg = LogisticRegression(
    solver='lbfgs',        # good for small dense data
    penalty='l2',          # L2 regularization (smooths coefficients)
    C=2.0,                 # default; try [0.1, 0.5, 1, 2, 5]
    max_iter=2000,
    verbose = 0,
    class_weight='balanced',  # optional if classes imbalanced
    n_jobs=-1
)
log_reg.fit(X_train[top_features], y_train)

'''
# Best Params C = 2.0

params = {'C': [0.01, 0.1, 0.5, 1, 2, 5, 10]}
grid = GridSearchCV(log_reg, params, scoring='roc_auc', cv=5)
grid.fit(X_train[top_features], y_train)
print("Best C:", grid.best_params_)
'''


# SGD
'''
print("Fitting SGD...")
sgd = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.001,   # equivalent to 1/C
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1
)
sgd.fit(X_train, y_train)
'''

### NON-LINEAR EVALUATION


# Random Forest # BEST 
print("Fitting Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10, # tweak ~10-15 to see overfitting affects
    verbose=0,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train[top_features], y_train)

'''
Best Parameters: {'max_depth': 20, 'min_samples_leaf': 5, 'n_estimators': 500}
Best CV AUC: 0.7464795120268887

# Best params
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [5, 10, 20],
    'min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=2
)

grid.fit(X_train[top_features], y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)
'''

# Cross validation score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_lr = cross_val_score(log_reg, X[top_features], y, cv=cv, scoring='roc_auc')
print(f"Logistic Regression  AUC: {cv_auc_lr.mean():.3f} ± {cv_auc_lr.std():.3f}")

cv_auc_rf = cross_val_score(rf, X[top_features], y, cv=cv, scoring='roc_auc')
print(f"Random Forest        AUC: {cv_auc_rf.mean():.3f} ± {cv_auc_rf.std():.3f}")

'''
# "Logistic Regression": log_reg
# "RandomForest": rf
# "Stoch.Grad.Desc.": sgd, 
models = {"RandomForest": rf}
plot_roc_curves(models, X_val[top_features], y_val)
'''

### PCA Exploration ###
# Explained variance ≠ predictive power.
# PCA maximizes variance, not necessarily class separability (e.g., breast cancer vs. control).
# ~95% variance at 2130 comps
# ~90% variance at 1500 comps
# ~85% variance at 1100 comps
# ~80% variance at 830  comps
# Standardize before PCA
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
pca = PCA().fit(X)  # X = your standardized feature matrix
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA variance explained')
plt.show()
'''


