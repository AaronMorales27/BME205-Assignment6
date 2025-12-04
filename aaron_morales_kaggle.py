import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ===============================
# PARAMETERS
# ===============================
random_state = 42
cv_folds = 5
feature_threshold = 0.5175  # top-feature AUC cutoff

# ===============================
# COMMAND LINE ARGUMENTS
# ===============================
if len(sys.argv) != 3:
    print("Usage: python script.py <train_csv> <test_csv>")
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]

# ===============================
# LOAD DATA
# ===============================
print("Loading datasets...")
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print("Raw shapes before encoding:")
print("  Train:", df_train.shape)
print("  Test:", df_test.shape)

print("Test unique IDs:", df_test['id'].nunique())

# ===============================
# ONE-HOT ENCODE CATEGORICALS
# ===============================

print("One-hot encoding ancestry...")

# Recreate index properly to avoid duplication
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

enc = OneHotEncoder(drop='first', sparse_output=False)

# Fit only on training data
enc.fit(df_train[['predicted_ancestry']])

# Transform both datasets
train_anc = pd.DataFrame(
    enc.transform(df_train[['predicted_ancestry']]),
    columns=enc.get_feature_names_out(['predicted_ancestry'])
)

test_anc = pd.DataFrame(
    enc.transform(df_test[['predicted_ancestry']]),
    columns=enc.get_feature_names_out(['predicted_ancestry'])
)

# Reattach indices
train_anc.index = df_train.index
test_anc.index = df_test.index

# Merge safely (avoids misalignment)
df_train_final = pd.concat(
    [df_train.drop(columns=['predicted_ancestry']), train_anc],
    axis=1
)

df_test_final = pd.concat(
    [df_test.drop(columns=['predicted_ancestry']), test_anc],
    axis=1
)

print("Train shape:", df_train_final.shape)
print("Test shape:", df_test_final.shape)


# ===============================
# FEATURE SELECTION
# ===============================
X = df_train_final.drop(columns=['breast_cancer'])
y = df_train_final['breast_cancer']

print("Selecting top features based on univariate AUC...")
aucs = X.apply(lambda col: roc_auc_score(y, col), axis=0)
top_features = aucs[aucs >= feature_threshold].index.tolist()
print(f"Number of selected features: {len(top_features)}")
X_sel = X[top_features]

y_vec = y.values.ravel()  # ensure 1D
print(f'num labels: {len(y_vec)}')
# ===============================
# CROSS-VALIDATION SETUP
# ===============================
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

# ===============================
# PIPELINES
# ===============================
pipe_logreg = Pipeline([
    ('scale', StandardScaler()),
    ('logreg', LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=random_state
    ))
])

rf = RandomForestClassifier(
    class_weight='balanced',
    n_jobs=-1,
    random_state=random_state
)

# ===============================
# HYPERPARAMETER GRIDS
# ===============================
param_grid_logreg = {'logreg__C': [0.1, 0.5, 0.75, 1, 1.5, 2, 5]}
param_grid_rf = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15],
    'min_samples_leaf': [2, 5]
}

# ===============================
# GRID SEARCH
# ===============================
print("Tuning Logistic Regression...")
grid_logreg = GridSearchCV(pipe_logreg, param_grid_logreg, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_logreg.fit(X_sel, y_vec)
best_logreg = grid_logreg.best_estimator_
print("Best Logistic Regression params:", grid_logreg.best_params_)
'''
print("Tuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_rf.fit(X_sel, y_vec)
best_rf = grid_rf.best_estimator_
print("Best Random Forest params:", grid_rf.best_params_)
'''
# ===============================
# CROSS-VALIDATION AUC
# ===============================
for name, model in [
    # ('Random Forest', best_rf),
    ('Logistic Regression', best_logreg)
]:
    aucs = cross_val_score(model, X_sel, y_vec, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"\n{name}:")
    print(f"  AUC mean ± std: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"  Tested with top {len(top_features)} features ({cv_folds}-fold CV)")

# ===============================
# TRAIN FINAL MODEL ON FULL DATA
# ===============================
final_model = best_logreg  # choose either best_rf or best_logreg
final_model.fit(X_sel, y_vec)

# ===============================
# PREPARE TEST SET & PREDICTIONS
# ===============================
X_test_sel = df_test_final[top_features]
y_test_pred_prob = final_model.predict_proba(X_test_sel)[:, 1]

# Ensure probabilities are 0-1
assert np.all(y_test_pred_prob >= 0) and np.all(y_test_pred_prob <= 1)

# ===============================
# CREATE SUBMISSION FILE
# ===============================
submission_df = pd.DataFrame({
    'id': df_test_final['id'],
    'breast_cancer_prob': y_test_pred_prob
})

submission_df.to_csv('bc_submission.csv', index=False)
print("Submission file saved as 'bc_submission.csv'.")
