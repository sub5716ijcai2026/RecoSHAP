import pandas as pd
import ast
import re
import numpy as np

def split_outside_braces_quotes(line: str):
    """Split line on commas, ignoring commas inside quotes or braces."""
    parts = []
    cur = []
    brace_depth = 0
    in_quote = False
    quote_char = None

    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote:
            if ch == quote_char and i+1 < len(line) and line[i+1] == quote_char:
                cur.append(quote_char)
                i += 2
                continue
            elif ch == quote_char:
                in_quote = False
                cur.append(ch)
                i += 1
                continue
            else:
                cur.append(ch)
                i += 1
                continue
        else:
            if ch in ('"', "'"):
                in_quote = True
                quote_char = ch
                cur.append(ch)
            elif ch == '{':
                brace_depth += 1
                cur.append(ch)
            elif ch == '}':
                brace_depth -= 1
                cur.append(ch)
            elif ch == ',' and brace_depth == 0:
                parts.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
            i += 1
    parts.append(''.join(cur).strip())
    return parts

def parse_model_summary(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = None
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # header
        if idx == 0:
            header = [h.strip() for h in split_outside_braces_quotes(line)]
            continue
        # remove outer quotes if whole line is quoted
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].replace('""', '"')
        parts = split_outside_braces_quotes(line)

        # ensure exactly 5 columns
        if len(parts) < 5:
            parts += [np.nan]*(5 - len(parts))
        elif len(parts) > 5:
            # merge extra columns into best_params
            extra = parts[2:len(parts)-(5-3)]
            parts = parts[:2] + [','.join(extra)] + parts[-2:]

        # parse best_params
        try:
            parts[2] = ast.literal_eval(parts[2])
        except Exception:
            parts[2] = {}

        # parse numeric columns
        for i in [1, 3, 4]:
            try:
                parts[i] = float(parts[i])
            except:
                parts[i] = np.nan

        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    return df


import os

# folder containing model_summary CSVs
shap_folder = "./shap_outputs"

# dictionary to store loaded DataFrames
model_summary_dfs = {}

# loop through all files starting with 'model_summary' and ending with .csv
for fname in os.listdir(shap_folder):
    if fname.startswith("model_summary") and fname.endswith(".csv"):
        fullpath = os.path.join(shap_folder, fname)
        try:
            df = parse_model_summary(fullpath)
            key_name = os.path.splitext(fname)[0]  # e.g., 'model_summary_Abalone'
            model_summary_dfs[key_name] = df
            print(f"Loaded {fname} -> {df.shape} rows")
        except Exception as e:
            print(f"[ERROR] could not parse {fname}: {e}")

# Example: access the Abalone model_summary
df_abalone = model_summary_dfs.get("model_summary_Abalone")
if df_abalone is not None:
    print(df_abalone.head())


import pandas as pd
import numpy as np
from collections import defaultdict

def summary_by_model_with_time(model_summary_dfs: dict,
                               model_col='model',
                               acc_col='accuracy',
                               train_col='training_time',
                               shap_col='shap_time',
                               normalize_model_names: bool = False):
    """
    For each model (across datasets in model_summary_dfs), compute:
      - per-dataset mean accuracy, then aggregate across datasets -> mean/std/min/max (accuracy)
      - per-dataset total_time = mean(training_time) + mean(shap_time), then aggregate across datasets -> mean/std/min/max (total_time)

    Returns a DataFrame with columns:
      ['model', 'n_datasets',
       'mean_accuracy','std_accuracy','min_accuracy','max_accuracy',
       'mean_total_time','std_total_time','min_total_time','max_total_time']
    """
    acc_values = defaultdict(list)
    total_time_values = defaultdict(list)

    for ds_name, df in model_summary_dfs.items():
        if df is None or df.shape[0] == 0:
            continue
        # defensive column check
        if model_col not in df.columns:
            print(f"[WARN] dataset '{ds_name}' missing model column '{model_col}' -> skipping")
            continue

        tmp = df.copy()
        # normalize model name
        if normalize_model_names:
            tmp[model_col] = tmp[model_col].astype(str).str.strip().str.lower()
        else:
            tmp[model_col] = tmp[model_col].astype(str).str.strip()

        # coerce numeric columns
        if acc_col in tmp.columns:
            tmp[acc_col] = pd.to_numeric(tmp[acc_col], errors='coerce')
        else:
            tmp[acc_col] = np.nan

        if train_col in tmp.columns:
            tmp[train_col] = pd.to_numeric(tmp[train_col], errors='coerce')
        else:
            tmp[train_col] = np.nan

        if shap_col in tmp.columns:
            tmp[shap_col] = pd.to_numeric(tmp[shap_col], errors='coerce')
        else:
            tmp[shap_col] = np.nan

        # drop rows that have no useful info (no accuracy and no time)
        if tmp[[acc_col, train_col, shap_col]].isna().all(axis=1).all():
            continue

        # per-dataset mean per-model for accuracy and times
        # compute mean accuracy per model (ignore NaNs)
        per_ds_acc = tmp.groupby(model_col, as_index=False)[acc_col].mean()
        # compute mean train and shap per model
        per_ds_time = tmp.groupby(model_col, as_index=False)[[train_col, shap_col]].mean()

        # merge on model to ensure same set
        per_ds = pd.merge(per_ds_acc, per_ds_time, on=model_col, how='outer')

        # for each model in this dataset, record values if available
        for _, row in per_ds.iterrows():
            model_name = row[model_col]
            # accuracy
            acc_val = row.get(acc_col, np.nan)
            if not pd.isna(acc_val):
                acc_values[model_name].append(float(acc_val))
            # total_time: need at least one of train/shap not NaN (we'll treat NaN as 0? better to require numeric)
            train_v = row.get(train_col, np.nan)
            shap_v = row.get(shap_col, np.nan)
            # If both NaN -> skip total_time for this dataset/model
            if pd.isna(train_v) and pd.isna(shap_v):
                continue
            # replace NaN with 0 when summing? Decision: treat missing as 0 would bias; instead treat NaN as 0 only if other exists.
            # We'll treat NaN as 0 when the other exists (so total_time uses available info). If both NaN we skipped above.
            t = 0.0
            
    ##### TO HAVE TOTAL TIME : PUT BACK
       #     if not pd.isna(train_v):
        #        t += float(train_v)
     #######
            if not pd.isna(shap_v):
                t += float(shap_v)
            total_time_values[model_name].append(float(t))

    # Build result rows
    rows = []
    all_model_names = set(list(acc_values.keys()) + list(total_time_values.keys()))
    for m in sorted(all_model_names):
        # accuracy stats
        acc_arr = np.array(acc_values.get(m, []), dtype=float) if acc_values.get(m) else np.array([], dtype=float)
        if acc_arr.size > 0:
            mean_acc = float(acc_arr.mean())
            std_acc = float(acc_arr.std(ddof=0))
            min_acc = float(acc_arr.min())
            max_acc = float(acc_arr.max())
            n_acc = acc_arr.size
        else:
            mean_acc = std_acc = min_acc = max_acc = np.nan
            n_acc = 0

        # total_time stats
        t_arr = np.array(total_time_values.get(m, []), dtype=float) if total_time_values.get(m) else np.array([], dtype=float)
        if t_arr.size > 0:
            mean_t = float(t_arr.mean())
            std_t = float(t_arr.std(ddof=0))
            min_t = float(t_arr.min())
            max_t = float(t_arr.max())
            n_t = t_arr.size
        else:
            mean_t = std_t = min_t = max_t = np.nan
            n_t = 0

        # n_datasets: number of datasets where the model had either an accuracy or a total_time value
        n_datasets = len(set(
            [m] if False else []
        ))  # placeholder, we'll compute below more meaningfully

        rows.append({
            'model': m,
            'n_datasets_with_accuracy': n_acc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'n_datasets_with_time': n_t,
            'mean_total_time': mean_t,
            'std_total_time': std_t,
            'min_total_time': min_t,
            'max_total_time': max_t
        })

    result = pd.DataFrame(rows)
    # sort by mean_accuracy desc (NaNs go last)
    result = result.sort_values(by='mean_accuracy', ascending=False, na_position='last').reset_index(drop=True)

    # nice rounding
    for c in ['mean_accuracy','std_accuracy','min_accuracy','max_accuracy']:
        if c in result.columns:
            result[c] = result[c].round(6)
    for c in ['mean_total_time','std_total_time','min_total_time','max_total_time']:
        if c in result.columns:
            result[c] = result[c].round(6)

    return result

# ---------------- Usage ----------------
# assume model_summary_dfs is defined
summary = summary_by_model_with_time(model_summary_dfs, normalize_model_names=False)
print(summary.to_string(index=False))

# plot_models_rmse_vs_time.py
import os, ast, re, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def split_outside_braces_and_quotes(line: str):
    parts = []; cur = []
    in_quote = False; quote_char = None; brace_depth = 0; i = 0; L = len(line)
    while i < L:
        ch = line[i]
        if in_quote:
            if ch == quote_char and i+1 < L and line[i+1] == quote_char:
                cur.append(quote_char); i += 2; continue
            elif ch == quote_char:
                in_quote = False; cur.append(ch); i += 1; continue
            else:
                cur.append(ch); i += 1; continue
        else:
            if ch in ('"', "'"):
                in_quote = True; quote_char = ch; cur.append(ch); i += 1; continue
            if ch == '{':
                brace_depth += 1; cur.append(ch); i += 1; continue
            if ch == '}':
                if brace_depth > 0: brace_depth -= 1
                cur.append(ch); i += 1; continue
            if ch == ',' and brace_depth == 0 and not in_quote:
                parts.append(''.join(cur).strip()); cur = []; i += 1; continue
            cur.append(ch); i += 1
    parts.append(''.join(cur).strip())
    return parts

def robust_split_line(line: str, header_len: int, best_param_idx=2):
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1].replace('""', '"')
    line = line.replace('""', '"').replace("''", "'")
    line = line.lstrip('[').rstrip(']').strip()
    tokens = split_outside_braces_and_quotes(line)
    if len(tokens) == header_len:
        return tokens
    if len(tokens) < header_len:
        return tokens + [''] * (header_len - len(tokens))
    if len(tokens) > header_len:
        extra = tokens[best_param_idx: len(tokens) - (header_len - best_param_idx -1)]
        tokens_new = tokens[:best_param_idx] + [','.join(extra)] + tokens[-(header_len - best_param_idx -1):]
        return tokens_new
    return tokens

def _normalize_cell(cell):
    if cell is None:
        return np.nan
    s = str(cell).strip()
    if s == '' or s.lower() in ('nan','none','na','null'):
        return np.nan
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]; s = s.replace('""','"').replace("''","'").strip()
    try:
        val = ast.literal_eval(s)
        return val
    except Exception:
        pass
    try:
        if re.search(r'[0-9]', s):
            return float(s)
    except Exception:
        pass
    return s

def parse_model_summary_file(fullpath: str, has_header: bool = True):
    rows = []; header = None
    with open(fullpath, 'r', encoding='utf-8', newline='') as fh:
        raw_lines = fh.readlines()
    for i, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        if i == 0 and has_header:
            header = split_outside_braces_and_quotes(raw_line)
            header = [h.strip().strip('"').strip("'") for h in header]
            continue
        parts = robust_split_line(raw_line, header_len=len(header) if header else 5, best_param_idx=2)
        norm = [_normalize_cell(p) for p in parts]
        rows.append(norm)
    if not rows:
        return pd.DataFrame(columns=header if header else [])
    maxlen = max(len(r) for r in rows)
    padded = [r + [np.nan]*(maxlen - len(r)) for r in rows]
    if header and len(header) == maxlen:
        df = pd.DataFrame(padded, columns=header)
    else:
        df = pd.DataFrame(padded)
        if header:
            for idx, name in enumerate(header):
                if idx < df.shape[1]:
                    df.rename(columns={idx: str(name)}, inplace=True)
    df.columns = [str(c) for c in df.columns]
    return df

# --- load all model_summary*.csv in shap_outputs ---
shap_folder = "./shap_outputs_classification_last2"
if not os.path.isdir(shap_folder):
    raise FileNotFoundError(f"Folder '{shap_folder}' not found. Please run this script from the project root or adjust shap_folder path.")

model_summary_dfs = {}
for fname in sorted(os.listdir(shap_folder)):
    if fname.startswith("model_summary") and fname.lower().endswith(".csv"):
        full = os.path.join(shap_folder, fname)
        try:
            df = parse_model_summary_file(full, has_header=True)
            model_summary_dfs[os.path.splitext(fname)[0]] = df
            print(f"Loaded {fname} with shape {df.shape}")
        except Exception as e:
            print(f"[ERROR] parsing {fname}: {e}")

if not model_summary_dfs:
    raise RuntimeError("No model_summary CSV files were successfully parsed in shap_outputs.")

# --- aggregate: per-dataset mean per model, then mean across datasets ---
acc_vals = defaultdict(list)
time_vals = defaultdict(list)
for ds_name, df in model_summary_dfs.items():
    if ds_name in [
    "model_summary_iris",
    "model_summary_car",
    "model_summary_splice",
    "model_summary_vehicle",
    "model_summary_Personal_Loan_Modelling",
    "model_summary_Breast-cancer",
    "model_summary_diabetes",
    "model_summary_chess",
    "model_summary_hypothyroid",
    "model_summary_mushroom",
    "model_summary_churn",
    ] :    
        if df is None or df.shape[0] == 0:
            continue
        for c in ['model','accuracy','training_time','shap_time']:
            if c not in df.columns:
                df[c] = np.nan
        tmp = df.copy()
        tmp['model'] = tmp['model'].astype(str).str.strip()
        tmp['accuracy'] = pd.to_numeric(tmp['accuracy'], errors='coerce')
        tmp['training_time'] = pd.to_numeric(tmp['training_time'], errors='coerce')
        tmp['shap_time'] = pd.to_numeric(tmp['shap_time'], errors='coerce')
        per_ds_acc = tmp.groupby('model', as_index=False)['accuracy'].mean()
        per_ds_time = tmp.groupby('model', as_index=False)[['training_time','shap_time']].mean()
        per_ds = pd.merge(per_ds_acc, per_ds_time, on='model', how='outer')
        for _, row in per_ds.iterrows():
            m = row['model']
            if pd.notna(row['accuracy']):
                acc_vals[m].append(float(row['accuracy']))
            ttrain = row.get('training_time', np.nan)
            tshap = row.get('shap_time', np.nan)
            if pd.isna(ttrain) and pd.isna(tshap):
                continue
            total = 0.0
            if pd.notna(ttrain):
                total += float(ttrain)
            if pd.notna(tshap):
                total += float(tshap)
            time_vals[m].append(total)

models = sorted(set(list(acc_vals.keys()) + list(time_vals.keys())))
mean_acc = []; mean_time = []
for m in models:
    a = np.array(acc_vals.get(m, []), dtype=float)
    t = np.array(time_vals.get(m, []), dtype=float)
    mean_acc.append(np.nan if a.size==0 else a.mean())
    mean_time.append(np.nan if t.size==0 else t.mean())

summary_df = pd.DataFrame({'model': models, 'mean_rmse': mean_acc, 'mean_total_time': mean_time})
summary_df['mean_rmse'] =1-summary_df['mean_rmse']
summary_df = summary_df[~summary_df['model'].isin(
    ['GaussianNB','LightGBM','CatBoost','XGBoost']
)]

plot_df = summary_df.dropna(subset=['mean_rmse','mean_total_time']).reset_index(drop=True)
if plot_df.empty:
    raise RuntimeError("No models with both mean_rmse and mean_total_time available to plot.")

# Group 1: orange
group1 = [
    'SVC','LinearSVC','LogisticRegression','RidgeClassifier','SGDClassifier'
]

# Group 2/3: green
group2_3 = [
'XGBoost', 'DecisionTree','GradientBoosting','RandomForest','CatBoost', 'KNN', 'AdaBoost', 'MLP','ExtraTrees'
]

# Create a color column
summary_df['color'] = summary_df['model'].apply(
    lambda m: 'orange' if m in group1 else ('green' if m in group2_3 else 'gray')
)


# keep only models with 15 accuracy values
valid_models = [m for m, v in acc_vals.items()] # if len(v) == 7]
# filter plot_df
plot_df_filtered = plot_df[plot_df['model'].isin(valid_models)].reset_index(drop=True)


# --- scatter plot ---
fig, ax = plt.subplots(figsize=(11,8))

y = plot_df['mean_total_time'].values
x = plot_df['mean_rmse'].values
ax.scatter(x, y,  color='lightgray',alpha=0.75, linewidths=0.25, edgecolors="black", s=110)#, label='early stopped')




ax.set_yscale('log')  # <-- logarithmic scale for total time
ax.set_ylabel('Mean total time (training_time + shap_time) [seconds]')
ax.set_xlabel('Mean Accuracy Error')  # lower is better for RMSE
#ax.set_title('Models: Mean RMSE vs Mean Total Time (one point per model)')

# annotate models
from adjustText import adjust_text
label_df = plot_df.reset_index(drop=True)
# Create text objects at the point locations (they'll be moved by adjust_text)
texts = []
# =============================================================================
# 
# for i, m in enumerate(plot_df['model']):
#     xi = x[i]; yi = y[i]
#     xoff = 0.01 * (max(x) - min(x) if max(x)!=min(x) else 1.0)
#     yoff = 0.01 * (max(y) - min(y) if max(y)!=min(y) else 1.0)
#     if m not in ["AdaBoost", "CIEL","RMTS","Poisson","ElasticNet", "LinearSVR","Huber", "SGDRegressor", "LeastSquares","RidgeRegressor","BayesianRidge","Tweedie"] :
#        ax.text(xi , yi, m, fontsize=10.5, ha='left', va='bottom')
#     else : 
# =============================================================================
for _, row in label_df.iterrows():
    xt = row['mean_rmse']
    yt = row['mean_total_time']
 #   if row["model"] == m : 
        # initial placement: center of the point
    txt = ax.text(xt, yt, row['model'], fontsize=11.5, ha='center', va='center')
    texts.append(txt)

# adjust_text will move text labels to avoid overlaps and draw arrows back to points
adjust_text(
    texts,
    x=label_df['mean_rmse'].values,
    y=label_df['mean_total_time'].values,
    ax=ax,
    expand_text=(1.5, 2.0),    # push texts away from each other more (x and y)
    expand_axes= True,
    expand_points=(1.05, 1.05),
    force_text=(1.5, 2.0),
    force_points=(0.1, 0.1),
    precision=0.005,            # smaller = finer placement (slower)
    lim=2250,                  # allow more optimisation iterations (default is smaller)
    only_move={'points':'y', 'texts':'xy'},  # helps with log-scale y
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.8)
)
    

y = plot_df_filtered['mean_total_time'].values
x = plot_df_filtered['mean_rmse'].values
# Separate x, y by group
x_global = summary_df.loc[summary_df['color']=='orange', 'mean_rmse']
y_global = summary_df.loc[summary_df['color']=='orange', 'mean_total_time']

x_local = summary_df.loc[summary_df['color']=='green', 'mean_rmse']
y_local = summary_df.loc[summary_df['color']=='green', 'mean_total_time']

# Scatter for global models
ax.scatter(
    x_global, y_global,
    color='orange', alpha=1, edgecolors='black', s=100,
    label='Global models'
)

# Scatter for local models
ax.scatter(
    x_local, y_local,
    color='green', alpha=1, edgecolors='black', linewidths=0.25, s=100,
    label='Local models'
)

#ax.scatter(x, y,     color=summary_df['color'], alpha=0.75, linewidths=0.25, edgecolors="black", s=110,   label='all datasets')




ax.grid(True)
ax.legend(loc='upper right')
#♦plt.tight_layout()

# Save outputs (adjust paths if running on Windows; /mnt/data is available in many notebook runtimes)
#out_png = "/mnt/data/models_rmse_vs_time.png"
#out_csv = "/mnt/data/model_rmse_time_summary.csv"
#plt.savefig(out_png, dpi=150)
#plot_df.to_csv(out_csv, index=False)
#plt.show()



# Assume plot_df has 'mean_rmse' (x) and 'mean_total_time' (y)
x = plot_df_filtered['mean_rmse'].values
y = plot_df_filtered['mean_total_time'].values

# Find Pareto-optimal points (lower RMSE and lower time is better)
def pareto_frontier(rmse, time):
    # Sort by RMSE ascending
    sorted_idx = np.argsort(rmse)
    rmse_sorted = rmse[sorted_idx]
    time_sorted = time[sorted_idx]
    is_pareto = []
    min_time = np.inf
    for i in range(len(rmse_sorted)):
        if time_sorted[i] < min_time:
            is_pareto.append(sorted_idx[i])
            min_time = time_sorted[i]
    return np.array(is_pareto)

pareto_idx = pareto_frontier(x, y)
pareto_df = plot_df_filtered.iloc[pareto_idx]


# Pareto-optimal models highlightedhted
ax.scatter(pareto_df['mean_rmse'], pareto_df['mean_total_time'], 
           color='red',alpha=1, linewidths=1.5, edgecolors="red", facecolors = "None", s=100, label='Pareto-optimal')

ax.set_xlabel('Mean Accuracy Error', fontsize=16)
#ax.set_ylabel('Mean total time (training_time + shap_time) [seconds]')
ax.set_ylabel('Mean SHAP time  [seconds]', fontsize=16)

ax.set_yscale('log')  # log scale for total time
#ax.set_title('Models: Pareto-optimal RMSE vs Total Time')
ax.grid(True, which='both', linestyle='--', alpha=0.6)

# =============================================================================
# # Annotate Pareto-optimal models only
# for i, row in pareto_df.iterrows():
#     ax.text(row['mean_rmse']*1.001, row['mean_total_time']*1.01, 
#             row['model'], fontsize=9, ha='left', va='bottom')
# 
# =============================================================================
leg= ax.legend(fontsize=16,loc='upper right')
#leg.set_bbox_to_anchor((1.0, 0.94))

plt.tight_layout()
plt.savefig("out_pareto_all_class_2.png", dpi=150)
plt.show()




