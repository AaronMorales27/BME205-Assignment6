from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import lightgbm as lgb

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

aucs = X.apply(lambda col: roc_auc_score(y, col), axis=0)
top_features = aucs[aucs >= 0.6].index.tolist() # indices of the top features + threshold
print(f'Number of features" {len(top_features)}')
X_sel = X[top_features]

# ============================
# ASSUME YOU ALREADY HAVE:
# X_sel -> your selected top features (e.g., 50-55 columns)
# y -> target labels
# ============================

# USER CONFIG
n_components = 15      # set to 0 to skip PCA
use_pca = True
cv_folds = 3
random_state = 42

# ============================
# STRATIFIED CV SETUP
# ============================
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

# ============================
# PIPELINE HELPER
# ============================
def make_pipe(model, use_pca=use_pca, n_components=n_components):
    steps = [('scaler', StandardScaler())]
    if use_pca and n_components > 0:
        steps.append(('pca', PCA(n_components=n_components, random_state=random_state)))
    steps.append(('model', model))
    return Pipeline(steps)

# ============================
# DEFINE MODELS
# ============================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=random_state,
    n_jobs=-1
)

lgb_model = LGBMClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1
)

pipe_rf = make_pipe(rf_model)
pipe_lgb = make_pipe(lgb_model)



# ============================
# CROSS-VALIDATION AUC
# ============================
for name, model in [('Random Forest', pipe_rf), ('LightGBM', pipe_lgb)]:
    print(f"Running CV for {name}...")
    aucs = cross_val_score(model, X_sel, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name} AUC mean ± std: {aucs.mean():.4f} ± {aucs.std():.4f}")
    if use_pca and n_components > 0:
        print(f"  Tested with {n_components} principal components ({cv_folds}-fold CV)")
    else:
        print(f"  Tested on {X_sel.shape[1]} features without PCA")
