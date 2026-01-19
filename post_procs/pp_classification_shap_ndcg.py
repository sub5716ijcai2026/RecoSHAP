import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import os
from tabpfn import  TabPFNRegressor
from sklearn.preprocessing import MinMaxScaler
from smt_explainability.shap import compute_shap_values,ShapDisplay
from sklearn.linear_model import Ridge as RidgeRegressor
from sklearn.linear_model import LogisticRegression,  SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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
from sklearn.metrics import mean_squared_error

import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, HuberRegressor, ElasticNet, LinearRegression, PoissonRegressor, TweedieRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import BayesianRidge
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

import openturns as ot  # ensure OpenTURNS is installed

import copy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
import openturns as ot
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier

# ================================
# Imports Boosting libs
# ================================
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")
# =======================
# 1. Load Dataset
# =======================
dataset="Abalone"
#dataset = "cmc"
#dfdata = pd.read_csv("Regression/"+dataset+".tsv", sep="\t")
dfdata = pd.read_csv("Regression/" + dataset + ".csv")
dfdata = dfdata.drop_duplicates(subset=dfdata.columns[:-1], keep='first')
feature_names =list(dfdata.columns.values)[:-1]

npdata = dfdata.to_numpy()

print("Numpy shape:", npdata.shape)

DATABASEx = npdata[:,:-1]
DATABASEy = npdata[:,-1]

scal= MinMaxScaler()
DATABASEx = scal.fit_transform(DATABASEx)
DATABASEy = scal.fit_transform(DATABASEy.reshape(-1, 1))

xt, X_test, yt, y_test = train_test_split(DATABASEx, DATABASEy, test_size=0.3, random_state=42) #,stratify=DATABASEy)
xval = DATABASEx
yval =DATABASEy

# Reproducibility seed
SEED = 42


  
# =======================
# 2. Define models
# =======================
class OpenTURNSPCERegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible, robust wrapper around OpenTURNS FunctionalChaosAlgorithm (PCE).

    Parameters
    ----------
    degree : int, default=2
        Maximum total degree (truncation) you want to test/tune.
        By default the wrapper sets OpenTURNS ResourceMap key
        "FunctionalChaosAlgorithm-MaximumTotalDegree" before building the PCE
        (this is the safest approach across OT versions).
    use_truncation_strategy : bool, default=False
        If True, the wrapper will attempt to build an explicit enumeration +
        FixedStrategy truncation and pass truncation/projection strategies to
        the FunctionalChaosAlgorithm constructor (this may fail on older/newer OT
        versions that don't expose the same overloads — the wrapper will try several overload forms).
    q_enum : float, default=1.0
        Hyperbolic enumeration q in (0,1] used if use_truncation_strategy=True.
    resource_map : dict or None
        Extra ResourceMap overrides (will be applied in fit). Keys must be strings.
        NOTE: the wrapper will NOT mutate the passed dict; it copies it before use.
    algorithm_kwargs : dict or None
        Extra kwargs to pass to the FunctionalChaosAlgorithm constructor (best-effort).
    input_distribution : openturns.Distribution or None
        If provided, the wrapper will attempt to call the constructor that accepts a distribution.
        If you pass a Distribution object, ensure it is the correct OT object.
    """

    def __init__(
        self,
        degree=2,
        use_truncation_strategy=False,
        q_enum=1.0,
        resource_map=None,
        algorithm_kwargs=None,
        input_distribution=None,
    ):
        # Constructor parameters only — do not create fitted attributes here
        self.degree = int(degree)
        self.use_truncation_strategy = bool(use_truncation_strategy)
        self.q_enum = float(q_enum)
        # store the dicts as is (do not mutate them later)
        self.resource_map = resource_map
        self.algorithm_kwargs = algorithm_kwargs
        self.input_distribution = input_distribution

    # --------------------
    # Helpers
    # --------------------
    def _apply_resource_map_copy(self):
        """Apply a copy of resource_map and also apply the MaximumTotalDegree key."""
        # We copy so we never mutate user-supplied dict.
        rm = {} if self.resource_map is None else dict(self.resource_map)
        # ensure degree is set (this is the safe approach)
        rm["FunctionalChaosAlgorithm-MaximumTotalDegree"] = int(self.degree)
        # Apply to OT ResourceMap
        for k, v in rm.items():
            try:
                if isinstance(v, bool):
                    ot.ResourceMap.SetAsBool(k, v)
                elif isinstance(v, int) and not isinstance(v, bool):
                    ot.ResourceMap.SetAsUnsignedInteger(k, int(v))
                elif isinstance(v, float):
                    ot.ResourceMap.SetAsScalar(k, float(v))
                else:
                    ot.ResourceMap.SetAsString(k, str(v))
            except Exception:
                # best-effort: continue if a specific key is not supported
                pass

    def _build_truncation_and_projection(self, n_features, X_ot, Y_ot):
        """Build enumeration, polynomial factory, truncation & projection strategies."""
        # Hyperbolic anisotropic enumeration
        enum_fn = ot.HyperbolicAnisotropicEnumerateFunction(n_features, float(self.q_enum))
        poly_factory = ot.OrthogonalProductPolynomialFactory(
            [ot.LegendreFactory()] * n_features, enum_fn
        )
        # Two ways to build truncation: either by degree or by index cardinal:
        try:
            # prefer a degree-based truncation if available
            trunc = ot.FixedStrategy(poly_factory, int(self.degree))
        except Exception:
            # fallback: compute the index cardinal for given degree
            try:
                idx = enum_fn.getStrataCumulatedCardinal(int(self.degree))
                trunc = ot.FixedStrategy(poly_factory, int(idx))
            except Exception:
                trunc = None

        proj = None
        try:
            proj = ot.LARSStrategy(X_ot, Y_ot)
        except Exception:
            proj = None

        return enum_fn, poly_factory, trunc, proj

    # --------------------
    # Main API
    # --------------------
    def fit(self, X, y):
        """
        Fit PCE metamodel.

        X: array-like shape (n_samples, n_features)
        y: array-like shape (n_samples,) or (n_samples, n_outputs)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_samples_y, output_dim = y.shape
        if n_samples != n_samples_y:
            raise ValueError("X and y must have same number of samples")

        # Save simple metadata
        self.n_features_in_ = n_features
        self.output_dim_ = output_dim

        # Apply resource map copy (sets maximum degree) — safest approach
        self._apply_resource_map_copy()

        # Convert to OpenTURNS Sample (use lists to be safe)
        xt_ot = ot.Sample(X.tolist())
        yt_ot = ot.Sample(y.tolist())

        # algorithm kwargs copy (don't mutate user dict)
        algo_kwargs = {} if self.algorithm_kwargs is None else dict(self.algorithm_kwargs)

        # If use_truncation_strategy: attempt to build explicit strategies and try several constructor overloads.
        if self.use_truncation_strategy:
            enum_fn, poly_factory, trunc, proj = self._build_truncation_and_projection(n_features, xt_ot, yt_ot)
            # Try a sequence of likely constructor signatures until one works
            tried_exceptions = []
            constructors_to_try = []

            # We'll try combinations in order of preference
            # 1) (xt, yt, input_dist, trunc, proj)
            if self.input_distribution is not None and trunc is not None and proj is not None:
                constructors_to_try.append(("xt_yt_dist_trunc_proj", (xt_ot, yt_ot, self.input_distribution, trunc, proj)))
            # 2) (xt, yt, trunc, proj)
            if trunc is not None and proj is not None:
                constructors_to_try.append(("xt_yt_trunc_proj", (xt_ot, yt_ot, trunc, proj)))
            # 3) (xt, yt, input_dist)
            if self.input_distribution is not None:
                constructors_to_try.append(("xt_yt_dist", (xt_ot, yt_ot, self.input_distribution)))
            # 4) (xt, yt)
            constructors_to_try.append(("xt_yt", (xt_ot, yt_ot)))

            algo = None
            for tag, args in constructors_to_try:
                try:
                    algo = ot.FunctionalChaosAlgorithm(*args, **algo_kwargs)
                    break
                except TypeError as e:
                    tried_exceptions.append((tag, str(e)))
                    algo = None
                except Exception as e:
                    # other exceptions (e.g., invalid args) — store and continue
                    tried_exceptions.append((tag, str(e)))
                    algo = None

            if algo is None:
                # if we failed to construct algorithm with truncation strategy, fallback to safest simple construction:
                # raise an informative error listing attempts
                msg = "Could not construct FunctionalChaosAlgorithm with any tried signature. Attempts:\n"
                for tag, err in tried_exceptions:
                    msg += f" - {tag}: {err}\n"
                msg += "As a fallback you can set use_truncation_strategy=False to use ResourceMap-based degree control."
                raise RuntimeError(msg)
            try:
                algo.setMaximumEvaluationNumber(int(1e6))
            except Exception:
                pass

            # If constructor accepted but we passed trunc/proj as separate args (in some overloads OT might not accept them),
            # we still try to set the strategies via constructor result if possible or continue.
            try:
                # run directly
                algo.run()
            except Exception:
                # If run fails, attempt to set strategies via setter names if they exist (best effort)
                try:
                    if trunc is not None and hasattr(algo, "setTruncationStrategy"):
                        algo.setTruncationStrategy(trunc)
                    if proj is not None and hasattr(algo, "setProjectionStrategy"):
                        algo.setProjectionStrategy(proj)
                    # some OT versions may use different method names; we don't assume more.
                    algo.run()
                except Exception as e:
                    raise RuntimeError(f"Failed to run FunctionalChaosAlgorithm after construction: {e}")

        else:
            # --------- SAFE SIMPLE FLOW ----------
            # Set maximum degree through ResourceMap already done above.
            # Try to construct with distribution if provided, otherwise simple form
            try:
                if self.input_distribution is not None:
                    algo = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot, self.input_distribution, **algo_kwargs)
                else:
                    algo = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot, **algo_kwargs)
                algo.run()
            except TypeError as e:
                # If constructor signature didn't accept distribution or kwargs, fallback to simplest form
                try:
                    algo = ot.FunctionalChaosAlgorithm(xt_ot, yt_ot)
                    algo.run()
                except Exception as e2:
                    raise RuntimeError(
                        "Failed to construct/run FunctionalChaosAlgorithm with the simple constructor. "
                        f"Original error: {e}; fallback error: {e2}"
                    )

        # Get result & metamodel
        try:
            result = algo.getResult()
        except Exception as e:
            raise RuntimeError(f"FunctionalChaosAlgorithm.run() succeeded but getResult() failed: {e}")

        if not hasattr(result, "getMetaModel"):
            raise RuntimeError("OpenTURNS returned a result without getMetaModel().")

        self.result_ = result
        self.metamodel_ = result.getMetaModel()
        return self

    def predict(self, X):
        if not hasattr(self, "metamodel_"):
            raise RuntimeError("Estimator not fitted. Call fit(X, y) first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        xt_ot = ot.Sample(X.tolist())
        y_ot = self.metamodel_(xt_ot)
        y_np = np.array(y_ot)
        y_np = np.where(np.isnan(y_np), np.array([6]), y_np)
        y_np = np.where(np.isinf(y_np), np.array([8]), y_np)
        if y_np.shape[1] == 1:
            return y_np[:, 0]
        return y_np

    def score(self, X, y, sample_weight=None):
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput="uniform_average")

    def get_params(self, deep=True):
        # Return exactly constructor params (so sklearn.clone works)
        return {
            "degree": self.degree,
            "use_truncation_strategy": self.use_truncation_strategy,
            "q_enum": self.q_enum,
            "resource_map": self.resource_map,
            "algorithm_kwargs": self.algorithm_kwargs,
            "input_distribution": self.input_distribution,
        }

    def set_params(self, **params):
        allowed = {
            "degree",
            "use_truncation_strategy",
            "q_enum",
            "resource_map",
            "algorithm_kwargs",
            "input_distribution",
        }
        for k, v in params.items():
            if k not in allowed:
                raise ValueError(f"Invalid parameter '{k}' for {self.__class__.__name__}")
            setattr(self, k, v)
        return self


class SMTWrapper(ScikitLearnAdapter):
    def __init__(self, model_cls,resource_map=None, algorithm_kwargs=None, n_comp=None, poly=None, n_start=None, random_state=None,
                 xlimits=None, n_inducing=None, print_global=False,corr=None,poly_degree=None,reg=None, p=None, kpls_dim=None,approx_order=None, **kwargs):
        # Save exposed hyperparameters
        self.resource_map = resource_map
        self.algorithm_kwargs = algorithm_kwargs
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
        self.approx_order = approx_order
        self.xlimits = xlimits

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
        if self.approx_order is not None : 
            merged["approx_order"] = self.approx_order
        if self.xlimits is not None : 
            merged["xlimits"] = self.xlimits
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
     #   return np.round(self.model_.predict_values(X).ravel())
        return (self.model_.predict_values(X).ravel())
    

import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from ciel.torch_mas.sequential.trainer import BaseTrainer as Trainer
from ciel.torch_mas.sequential.internal_model import LinearWithMemory
from ciel.torch_mas.sequential.activation_function import BaseActivation
from ciel.torch_mas.data import DataBuffer


class CIELRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper for CIEL (torch_mas) sequential trainer.
    Hyperparameters exposed: alpha, l1, memory_length, R, imprecise_th, bad_th, n_epochs, seed, device.
    """

    def __init__(self,
                 alpha=0.1,
                 l1=0.1,
                 memory_length=10,
                 R=0.1,
                 imprecise_th=0.01,
                 bad_th=0.1,
                 n_epochs=5,
                 seed=256,
                 device="cpu"):
        self.alpha = alpha
        self.l1 = l1
        self.memory_length = memory_length
        self.R = R
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.n_epochs = n_epochs
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32, device=self.device)

        dataset = DataBuffer(X_t, y_t)

        validity = BaseActivation(dataset.input_dim, dataset.output_dim, alpha=self.alpha)
        internal_model = LinearWithMemory(dataset.input_dim,
                                          dataset.output_dim,
                                          l1=self.l1,
                                          memory_length=self.memory_length)

        self.model_ = Trainer(validity,
                               internal_model,
                               R=self.R,
                               imprecise_th=self.imprecise_th,
                               bad_th=self.bad_th,
                               n_epochs=self.n_epochs,
                               device=self.device)

        self.model_.fit(dataset)
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Model not fitted — call fit first.")
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_pred = self.model_.predict(X_t)
        return y_pred.detach().cpu().numpy()

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            "l1": self.l1,
            "memory_length": self.memory_length,
            "R": self.R,
            "imprecise_th": self.imprecise_th,
            "bad_th": self.bad_th,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
            "device": self.device
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self

models = {
    # Linear & classifiers
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
    ),
    "RidgeClassifier": (
        RidgeClassifier(random_state=SEED),
    ),
    "SGDClassifier": (
        SGDClassifier(max_iter=1000, tol=1e-3, random_state=SEED),
    ),
    "LinearSVC": (
        LinearSVC(max_iter=2000, random_state=SEED),
    ),
# =============================================================================
#     "LeastSquares": (
#         SMTWrapper(LS, print_global=False),
#         {}
#     ),
# =============================================================================

    # Kernel / distance-based
    "SVC": (
        SVC(probability=True, random_state=SEED),
    ),
    "KNN": (
        KNeighborsClassifier(),
    ),
# =============================================================================
#     "RBF": (
#         SMTWrapper(RBF,  print_global=False),
#         {
#             "poly_degree": [-1, 0, 1],
#             "reg" : [1e-10,1e-9,1e-8,1e-7],
#         }
#     ),
# =============================================================================
# =============================================================================
#     "IDW": (
#         SMTWrapper(IDW,  print_global=False),
#         {
#             "p" : [1,2,3],
#         }
#     ),
# =============================================================================

    # Tree-based
    "DecisionTree": (
        DecisionTreeClassifier(random_state=SEED),
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=SEED),
    ),

    # Boosting
    "AdaBoost": (
        AdaBoostClassifier(random_state=SEED),
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=SEED),
    ),
# =============================================================================
#     "XGBoost": (
#         XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED),
#     ),
# =============================================================================
# =============================================================================
#     
#     # Neural Networks
#     "MLP": (
#         MLPRegressor(max_iter=500, random_state=SEED),
#      
#     ),
# =============================================================================
    
# =============================================================================
#     "LightGBM": (
#         LGBMClassifier(verbose=-1, random_state=SEED),
#     ),
# =============================================================================
# =============================================================================
#     "CatBoost": (
#         CatBoostClassifier(verbose=0, random_state=SEED),
#     ),
# =============================================================================

    # Neural Networks
    "MLP": (
            MLPClassifier(max_iter=500, random_state=SEED),
    ),

    # Probabilistic
# =============================================================================
#     "GaussianNB": (
#         GaussianNB(),
#     ),
# 
# =============================================================================
# =============================================================================
#     # Surrogate / SMT models
#     "KPLS": (
#         SMTWrapper(GPX, print_global=False, eval_noise=True, seed=SEED),
#         {
#             "corr" : ["abs_exp", "pow_exp"],
#             "poly" : ["constant", "linear"],
#             "kpls_dim" : [2,3],
#         }
#     ),
# =============================================================================
# =============================================================================
#     "QP": (
#         SMTWrapper(QP, print_global=False),
#         {}
#     ),
# =============================================================================
# =============================================================================
#     "SGP": (
#         SMTWrapper(SGP, print_global=False, seed=SEED),
#         {"n_inducing" : [20,30,50]},
#     ),
# =============================================================================
}


# =======================
# 5. Post-process
# ======================= 



    

datasets = ["pol","BNG_lowbwt","houses"]

#datasets = ["Abalone", "cpu_small","satellite_image","wind", "puma8NH"]

#datasets =[ "pm10", "Wine_Quality", "pollen"]

#datasets = ["LEV", "ESL", "analcatdata_apnea1", "ERA"]
datasets = ["LEV", "ESL", "analcatdata_apnea1", "ERA"]

datasets = ["iris"]

datasets = ["car"]
datasets = ["Breast-cancer"]
datasets = ["diabetes"]
datasets = ["cmc"]
datasets = ["wine"]
datasets = ["chess"]
datasets = ["hypothyroid"]
atasets = ["mushroom"]
datasets = ["splice"]   
datasets = ["vehicle"]
datasets = ["Customer_Churn"]
datasets = [
       "iris",
       "car",
 #   "cmc",
   # "wine",
    "splice",
    "vehicle",
  #  "shuttle",
    "Personal_Loan_Modelling"    ,
  #  "Adult",
    "Breast-cancer",
    "diabetes",
    "chess",
    "hypothyroid",
    "mushroom",
    "churn",
]
model_names = list(models.keys())
num_models = len(model_names)
num_datasets = len(datasets)

# --- Store all NDCG matrices per dataset ---
ndcg_all = np.zeros((num_datasets, num_models, num_models))

for d_idx, dataset in enumerate(datasets):
    shap_values_dict = {}
    print(dataset)
    # Load SHAP values for this dataset
    for name in model_names:
        shap_file = f"shap_outputs_classification_last/{dataset}/{name}_shap_values_{dataset}.csv"
        shap_df = pd.read_csv(shap_file)
        shap_values_dict[name] = shap_df.values

    # Compute pairwise NDCG between models
    ndcg_matrix = np.eye(num_models)  # start with identity (1 on diagonal)
    for i, j in combinations(range(num_models), 2):
        key_i = model_names[i]
        key_j = model_names[j]
        model_i = np.abs(shap_values_dict[key_i])
        model_j = np.abs(shap_values_dict[key_j])

        # Compute NDCG in both directions
        ndcg_ij = ndcg_score(model_i, model_j)
        ndcg_ji = ndcg_score(model_j, model_i)

        ndcg_matrix[i, j] = ndcg_ij
        ndcg_matrix[j, i] = ndcg_ji

    # Store this dataset’s results
    ndcg_all[d_idx] = ndcg_matrix

# --- Compute median across datasets ---
ndcg = np.mean(ndcg_all, axis=0)

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
    cbar_pos=None,
#    cbar=False,
    norm = norm,
    fmt=".0f",
    annot_kws={"size": 12},
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
    row_cluster=True,
    col_cluster=True,
    col_linkage=col_linkage,  
    row_linkage = row_linkage,
    cmap=discrete_cmap,
    norm = norm,
    cbar=False,
    annot=True,
    cbar_pos=None,
    linewidths=0.25,
    linecolor='black',
    annot_kws={"size": 12},
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






# Show the plot
plt.savefig("ndcg_all_v2_class.png", dpi=350, bbox_inches='tight')
plt.tight_layout()
plt.show()




