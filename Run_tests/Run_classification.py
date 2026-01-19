import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import os
from sklearn.preprocessing import MinMaxScaler
from smt_explainability.shap import compute_shap_values,ShapDisplay
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from smt.surrogate_models import KRG, KPLS, LS, QP, RBF,IDW,RMTB,RMTC,GPX,GENN,SGP
from sklearn.base import BaseEstimator, RegressorMixin
import egobox as egx
from egobox import SparseGpx as SGPX
import time
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from smt.utils.sklearn_adapter import ScikitLearnAdapter 
from smt.surrogate_models import KPLS
from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np
import itertools
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score
from itertools import combinations
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr

# =======================
# 1. Load Dataset
# =======================
dataset="iris"
#dataset = "cmc"
dfdata = pd.read_csv("Classification/"+dataset+".tsv", sep="\t")
dfdata = dfdata.drop_duplicates(subset=dfdata.columns[:-1], keep='first')
feature_names =list(dfdata.columns.values)[:-1]

npdata = dfdata.to_numpy()

print("Numpy shape:", npdata.shape)

DATABASEx = npdata[:,:-1]
DATABASEy = npdata[:,-1]

scal= MinMaxScaler()
DATABASEx = scal.fit_transform(DATABASEx)

xt, X_test, yt, y_test = train_test_split(DATABASEx, DATABASEy, test_size=0.3, random_state=42,stratify=DATABASEy)
xval = DATABASEx
yval =DATABASEy

# Reproducibility seed
SEED = 42


  
# =======================
# 2. Define models
# =======================


class SMTWrapper(ScikitLearnAdapter):
    def __init__(self, model_cls, n_comp=None, poly=None, n_start=None, random_state=None,
                 n_inducing=None, print_global=False,corr=None,poly_degree=None,reg=None, p=None, kpls_dim=None, **kwargs):
        # Save exposed hyperparameters
        self.n_comp = n_comp
        self.kpls_dim =kpls_dim
        self.poly = poly
        self.n_start = n_start
        self.corr = corr
        self.random_state = random_state
        self.print_global = print_global
        self.poly_degree = poly_degree
        self.p = p
        self.n_inducing = n_inducing
        self.reg = reg
        self.extra_kwargs = kwargs  # save any other passed kwargs
        
        # Call parent constructor
        super().__init__(model_cls)

    def fit(self, X, y):
        # Merge explicit params with extra kwargs
        merged = dict(self.extra_kwargs)
        if self.n_comp is not None:
            merged["n_comp"] = self.n_comp
        if self.kpls_dim is not None:
            merged["kpls_dim"] = self.kpls_dim
        if self.poly is not None:
            merged["poly"] = self.poly
        if self.n_start is not None:
            merged["n_start"] = self.n_start
        if self.random_state is not None:
            merged["random_state"] = self.random_state
        if self.corr is not None:
            merged["corr"] = self.corr
        if self.poly_degree is not None:
            merged["poly_degree"] = self.poly_degree
        if self.p is not None:
            merged["p"] = self.p
        if self.reg is not None : 
            merged["reg"] = self.reg
       
        merged["print_global"] = False
        # Set inducing points automatically if needed

        # Instantiate the SMT model with merged params
        self.model_ = self.model_cls(**merged)
        self.model_.set_training_values(X, y)
        if hasattr(self.model_, "set_inducing_inputs") and self.n_inducing is not None:
            rng = np.random.default_rng(42)  # reproducible randomness
            indices = rng.choice(X.shape[0], size=self.n_inducing, replace=False)
            self.model_.set_inducing_inputs(X[indices, :])


        self.model_.train()
        return self

    def predict(self, X):
        return np.round(self.model_.predict_values(X).ravel())
    
  
models = {
    # Linear models
    "LogisticRegression": (
        LogisticRegression(max_iter=500, random_state=SEED),
        {"C": [0.1, 1, 10]}
    ),
    "RidgeClassifier": (
        RidgeClassifier(random_state=SEED),
        {"alpha": [0.1, 1.0, 10.0]}
    ),
    "SGDClassifier": (
        SGDClassifier(max_iter=1000, tol=1e-3, random_state=SEED),
        {"loss": ["hinge", "log_loss"],     "alpha": [0.0001, 0.001]}
    ),

    # SVM family
    "SVC": (
        SVC(probability=True, random_state=SEED),
        {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
    ),
    "LinearSVC": (
        LinearSVC(max_iter=2000, random_state=SEED),
        {"C": [0.1, 1, 10]}
    ),

    # Tree-based
    "DecisionTree": (
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": [3, 5, None]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [None, 10]}
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [None, 10]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=SEED),
        {"n_estimators": [50, 100]}
    ),

    # Boosting libs
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED),
        {"n_estimators": [100, 200]}
    ),
    "LightGBM": (
        LGBMClassifier(verbose=-1, random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [-1, 10]}
    ),
    "CatBoost": (
        CatBoostClassifier(verbose=0, random_state=SEED),
        {"iterations": [200, 500],   "depth": [4, 6]}
    ),

    # Neighbors
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7]}
    ),

    # Naive Bayes
    "GaussianNB": (GaussianNB(), {}),

    # Neural Networks
    "MLP": (
        MLPClassifier(max_iter=500, random_state=SEED),
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001]
        }
    ),
   "KPLS": (
        SMTWrapper(GPX, print_global=False,eval_noise=True, seed=SEED),
        {
            "corr" : ["abs_exp", "pow_exp"],
             "poly" : ["constant", "linear"],
             "kpls_dim" : [2,3,4],
        }
    ),
    
    "LeastSquares": (
        SMTWrapper(LS, print_global=False),
        {}
    ),
    "RBF": (
        SMTWrapper(RBF,  print_global=False),
        {
            "poly_degree": [-1, 0, 1],
            "reg" : [1e-10,1e-9,1e-8,1e-7],
        }
    ),
    "QP": (
        SMTWrapper(QP, print_global=False),
        {}
    ),
    "IDW": (
        SMTWrapper(IDW,  print_global=False),
        {
        "p" : [1,2,3],
            }
    ),
    "SGP": (
        SMTWrapper(SGP, print_global=False, seed=SEED),
        {"n_inducing" : [20,30,50],
         }
    ),
    
}

# =======================
# 3. Create output folder for SHAP values
# =======================
os.makedirs("shap_outputs", exist_ok=True)

def get_model_predict_function(model):
    """Return a function compatible with SHAP for any sklearn-like model."""
    return model.predict

# --- Dictionnaire pour stocker résultats SHAP et importances ---
shap_values_dict = {}      # pour stocker toutes les valeurs SHAP
best_models = {}           # pour stocker les modèles entraînés
best_params_dict = {}  # best hyperparameters
computational_time = {}
results_list = []


# =======================
# 4. Loop Over Models
# =======================
for name, (model, params) in models.items():
    print(f"\n===== {name} =====")
    computational_time[name] = {}
    a = time.time()
    # Pipeline: scaling for sklearn models only
    if isinstance(model, SMTWrapper):
        pipe = model
        param_grid = params
    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        # prefix grid parameters for pipeline
        param_grid = {("clf__" + k if not k.startswith("clf__") else k): v for k, v in params.items()}

    # GridSearchCV
    grid = GridSearchCV(pipe, param_grid=param_grid or {}, scoring="accuracy", cv=4, n_jobs=-1)
    grid.fit(xt, yt)

    # Store best model and parameters
    best_model = grid.best_estimator_
    best_models[name] = best_model
    best_params_dict[name] = grid.best_params_

    print("Best Params:", grid.best_params_)    
    # Best model
  #  best_model = model
    best_model.fit(xt,yt)
    y_pred = best_model.predict(xval)
    acc = accuracy_score(yval, y_pred)

    print("Accuracy:", acc)

    # =======================
    # Fit best model on whole dataset
    # =======================
    best_model.fit(xval, yval)
    computational_time[name]["training"] = time.time()-a

    # =======================
    # SHAP Explainability
    # =======================
    # If best_model is a pipeline, extract the final estimator
    model_to_explain = best_model
    if hasattr(best_model, "steps"):  # it's a Pipeline
        model_to_explain = best_model.steps[-1][1]

    predictions = model_to_explain.predict(xval)

    # Get correct prediction function
    predict_fn = get_model_predict_function(model_to_explain)
    explainer = shap.Explainer(predict_fn, xval)
    a= time.time()
    shap_values = explainer(xval).values
    computational_time[name]["shap_pred"]  = time.time()-a
  
    shap_values_dict[name] = shap_values  # matrice n_samples x n_features   
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    # Ajouter au résumé
    results_list.append({
        "model": name,
        "accuracy": acc,
        "best_params": grid.best_params_,
        "training_time": computational_time[name]["training"],
        "shap_time":     computational_time[name]["shap_pred"],
    })
    # Export SHAP values
    shap_file = f"shap_outputs/{name}_shap_values_{dataset}.csv"
    shap_df.to_csv(shap_file, index=False)
    print(f"SHAP values saved to {shap_file}")
    # Convertir en DataFrame récapitulatif sans shap_df (car c’est une DataFrame par modèle)
    df_results = pd.DataFrame(results_list)

    # Sauvegarder le résumé
    summary_file = f"shap_outputs/model_summary_{dataset}.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
          



# =======================
# 5. Post-process
# ======================= 


models = list(models.keys())


from ignite.metrics import MaximumMeanDiscrepancy
from ignite.engine import Engine
import torch
from ignite.metrics import MaximumMeanDiscrepancy

mmd_metric = MaximumMeanDiscrepancy(var=1.0)  # Adjust 'var' as needed


num_models = len(models)
ndcg = np.eye(num_models)
mmd  = np.eye(num_models,num_models)

for i, j in combinations(range(num_models), 2):
    key_i = list(shap_values_dict.keys())[i]
    key_j = list(shap_values_dict.keys())[j]
    model_i = shap_values_dict[key_i]
    model_j =  shap_values_dict[key_j]

    ndcg[i,j] = ndcg_score(np.abs(model_i), np.abs(model_j))
    ndcg[j,i] = ndcg_score(np.abs(model_j), np.abs(model_i))
    # Convert SHAP values to tensors

    mmd_metric.update((torch.from_numpy(model_i), torch.from_numpy(model_j)))
    mmd_value = mmd_metric.compute()
    mmd[i,j] = 1-mmd_value
    mmd[j,i] = 1-mmd_value
    
# =============================================================================
#     correlation = 0
#     for col in range(len(feature_names)):
#         rho, pval = pearsonr(abs(model_i[:, col]),
#                          abs(model_j[:, col]))
#         correlation+= rho
#     mmd[i,j] = correlation
#     mmd[j,i] = correlation
#     
# 
#     # Access the computed MMD
#     
# =============================================================================

# =============================================================================
# ndcg = mmd
# =============================================================================


# -----------------------------
# 1. Prepare similarity matrix
# -----------------------------
labels = list(shap_values_dict.keys())
df_cm = pd.DataFrame(np.int32(ndcg * 100), index=labels, columns=labels)


# -----------------------------
# 2. Compute linkage
# -----------------------------
# Use "1 - similarity" as distance if your matrix is similarity (NDCG)
dist_matrix = 100 - df_cm.values  
linkage = sch.linkage(dist_matrix, method="average")

# -----------------------------
# 3. Force exactly 2 groups
# -----------------------------
clusters = fcluster(linkage, t=2, criterion="maxclust")

# Order labels by cluster assignment
ordered_idx = np.argsort(clusters)
ordered_labels = [labels[i] for i in ordered_idx]
df_cm_ordered = df_cm.loc[ordered_labels, ordered_labels]



# -----------------------------
# 4. Plot heatmap with 2-group dendrogram
# -----------------------------
n = 6  # number of discrete colors you want
base_cmap = plt.cm.get_cmap("Blues")  # continuous colormap

# Sample n colors evenly from the continuous cmap
colors = base_cmap(np.linspace(0, 1, n))


discrete_cmap = ListedColormap(colors)



# Définir les bornes pour chaque couleur selon min/max de df
bounds = np.linspace(np.min(df_cm_ordered)-1, np.max(df_cm_ordered)+1, n+1)

norm = BoundaryNorm(bounds, discrete_cmap.N)



g = sns.clustermap(
    df_cm_ordered,
    row_cluster=True,   # no reclustering, keep 2-group order
    col_cluster=True,
    cmap=discrete_cmap,
    cbar=False,
    annot=True,
    norm = norm,
    fmt=".0f",
    annot_kws={"size": 16},
    linewidths=0.25,
    linecolor="black",
    square=True,
    figsize=(10, 10),
    dendrogram_ratio=(0.1, 0.1),

)

# Beautify ticks
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(),
    rotation=45,
    ha="right",
    fontsize=14,
)
g.ax_heatmap.set_yticklabels(
    g.ax_heatmap.get_yticklabels(),
    rotation=0,
    fontsize=14,
)

plt.show()


n_cols = df_cm_ordered.shape[1]

# Compute linkage from distance matrices
# Compute condensed distance matrices
row_dist = pdist(df_cm_ordered.values, metric='euclidean')
col_dist = pdist(df_cm_ordered.values.T, metric='euclidean')

row_linkage = sch.linkage(row_dist, method='weighted')
col_linkage = sch.linkage(col_dist, method='weighted')


g = sns.clustermap(
    df_cm_ordered,
    row_cluster=False,
    col_cluster=False,
    col_linkage=col_linkage,  
    row_linkage = row_linkage,
    cmap=discrete_cmap,
    norm = norm,
    cbar=False,
    annot=True,
    cbar_pos=None,
    linewidths=0.25,
    linecolor='black',
    annot_kws={"size": 18},
    square=True,
    fmt=".0f",
    figsize=(10, 10),
    dendrogram_ratio=(0.1, 0.1),
)
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=16,     
)
g.ax_heatmap.set_yticklabels(
    g.ax_heatmap.get_yticklabels(),
    rotation=0,
    fontsize=16,     
)


plt.show()





