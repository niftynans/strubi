import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def optimize_and_update_predictions(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)
    
    # --- FIX: Explicitly convert metric columns to numeric types ---
    df['sync_mech_ami'] = pd.to_numeric(df['sync_mech_ami'], errors='coerce')
    df['coverage'] = pd.to_numeric(df['coverage'], errors='coerce')
    # --------------------------------------------------------------

    # Identify the "Full" oracle to find thresholds
    if 'full' in df['oracle'].values:
        df_ref = df[df['oracle'] == 'full'].copy()
        print("Using 'full' oracle for threshold optimization.")
    else:
        df_ref = df.copy()
        print(f"Warning: 'full' oracle not found. Using all available data.")

    # 2. Find optimal t1 (AMI) to separate 'none' from any shift
    thresholds = np.linspace(0, 1, 101)
    best_t1, best_f1_t1 = 0.5, -1
    
    # Drop NaNs just for the threshold calculation to ensure clean math
    clean_ref = df_ref.dropna(subset=['sync_mech_ami', 'coverage'])
    
    y_true_bin = (clean_ref['gt_type'] != 'none').astype(int)
    for t in thresholds:
        y_pred_bin = (clean_ref['sync_mech_ami'] > t).astype(int)
        score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        if score > best_f1_t1:
            best_f1_t1, best_t1 = score, t
            
    # 3. Find optimal t2 (Coverage) and direction for Confounder vs Collider
    sub_shifts = clean_ref[clean_ref['gt_type'] != 'none']
    best_t2, best_f1_t2, conf_lower = 0.5, -1, True
    
    if not sub_shifts.empty:
        y_true_type = sub_shifts['gt_type']
        for t in thresholds:
            for direction in [True, False]:
                if direction:
                    preds = sub_shifts['coverage'].apply(lambda x: 'confounder' if x < t else 'collider')
                else:
                    preds = sub_shifts['coverage'].apply(lambda x: 'collider' if x < t else 'confounder')
                
                score = f1_score(y_true_type, preds, labels=['confounder', 'collider'], average='macro', zero_division=0)
                if score > best_f1_t2:
                    best_f1_t2, best_t2, conf_lower = score, t, direction

    print(f"Optimal Thresholds Found: AMI={best_t1:.2f}, Coverage={best_t2:.2f}")
    # 4. Apply changes INPLACE to predicted_gt_type
    def predict_logic(row):
        # Handle cases where AMI or Coverage might be NaN
        if pd.isna(row['sync_mech_ami']): return 'none'
        
        # Step 1: Presence check
        if row['sync_mech_ami'] <= best_t1:
            return 'none'
        # Step 2: Type check
        if conf_lower:
            return 'confounder' if row['coverage'] < best_t2 else 'collider'
        else:
            return 'collider' if row['coverage'] < best_t2 else 'confounder'

    # Filter for the specific oracles the user requested
    target_oracles = ['full', 'perturbed', 'TOPIC']
    mask = df['oracle'].str.lower().isin([o.lower() for o in target_oracles])
    
    df.loc[mask, 'predicted_gt_type'] = df[mask].apply(predict_logic, axis=1)
    # print(f"Accuracy: {accuracy_score((s := df[df['oracle'] == 'full'])['gt_type'], s['predicted_gt_type']):.4f}, Macro F1: {f1_score(s['gt_type'], s['predicted_gt_type'], labels=['none', 'confounder', 'collider'], average='macro', zero_division=0):.4f}")
    
    
    # 5. Save the result
    df.to_csv(file_path, index=False) # Saving back to the same file for "inplace" effect
    print(f"✓ Updated predictions in {file_path}")

def clean_to_set(val):
    if pd.isna(val) or str(val).strip().lower() in ["none", "[]", ""]: return set()
    try:
        s = str(val).replace("np.int64(", "").replace("np.float64(", "").replace(")", "")
        parsed = ast.literal_eval(s)
        flat, stack = [], [parsed]
        while stack:
            curr = stack.pop()
            if isinstance(curr, list): stack.extend(curr)
            elif curr is not None: flat.append(curr)
        out = set()
        for item in flat:
            t = str(item).strip().replace("'", "").replace('"', "")
            if not t: continue
            out.add(f"X{t}" if t.isdigit() else (t if t.startswith("X") else f"X{t}"))
        return out
    except Exception: return set()

def extract_nodes(val):
    if not val or str(val).strip() in ["None", "[]", ""]:
        return set()
    try:
        clean_val = str(val).replace("np.int64(", "").replace("np.float64(", "").replace(")", "")
        parsed = ast.literal_eval(clean_val)
        extracted = []
        stack = [parsed]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                stack.extend(list(current))
            elif current is not None:
                extracted.append(current)
        node_set = set()
        for n in extracted:
            item = str(n).strip().replace("'", "").replace('"', "")
            if not item: continue
            formatted = f"X{item}" if item.isdigit() else (item if item.startswith("X") else f"X{item}")
            node_set.add(formatted)
        return node_set
    except:
        return set()


def subset_f1_from_sets(pred_set, exp_set):
    if not exp_set: return np.nan
    tp, fp, fn = len(pred_set & exp_set), len(pred_set - exp_set), len(exp_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0



def process_any_ablation(input_csv, base_output_dir="csv_for_figs"):
    df = pd.read_csv(input_csv)
    axis_name = df['axis'].iloc[0].replace("-", "_")
    output_dir = os.path.join(base_output_dir, f"{axis_name}_ablations")
    os.makedirs(output_dir, exist_ok=True)
    
    df["subset_f1"] = df.apply(lambda row: subset_f1_from_sets(clean_to_set(row["subset_nodes"]), clean_to_set(row["expected_set"])), axis=1)

    if axis_name == "contexts":
        x_axis, series_col = 'axis_val', 'n_nodes'
    else:
        x_axis = 'n_nodes' if len(df['n_nodes'].unique()) > 1 else 'axis_val'
        series_col = 'axis_val' if x_axis == 'n_nodes' else 'oracle'

    unique_x = sorted(df[x_axis].unique())
    unique_series = sorted(df[series_col].unique())
    gt_types = ['none', 'confounder', 'collider']

    
    for m_name, func in {
        'f1': lambda sub: f1_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'accuracy': lambda sub: accuracy_score(sub['gt_type'], sub['predicted_gt_type']),
        'precision': lambda sub: precision_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'recall': lambda sub: recall_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'subset': lambda sub: sub[sub['gt_type'] != 'none']['subset_f1'].mean(),
        'runtime': lambda sub: pd.to_numeric(sub['runtime'], errors='coerce').mean()
    }.items():
        data = {"X": unique_x}
        for s in unique_series:
            means, stds = [], []
            for x in unique_x:
                sub = df[(df[x_axis] == x) & (df[series_col] == s)]
                if sub.empty:
                    means.append(np.nan); stds.append(0.0)
                    continue
                
                # 1. Global Mean (Exact original logic)
                means.append(func(sub))

                # 2. Standard Deviation calculation
                if m_name == 'f1':
                    sd = sub.groupby('seed').apply(lambda g: f1_score(
                        g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                        average="weighted", zero_division=0)).std()
                elif m_name == 'subset':
                    sd = sub[sub['gt_type'] != 'none']['subset_f1'].std()
                else:
                    sd = pd.to_numeric(sub['runtime'], errors='coerce').std()
                
                stds.append(sd if not pd.isna(sd) else 0.0)
            
            data[str(s)] = means
            data[str(s) + '_std'] = stds
        pd.DataFrame(data).to_csv(f"{output_dir}/{axis_name}_{m_name}.csv", index=False)

    # Loop 2: Information-theoretic signals (AMI, Coverage)
    for metric in ['sync_mech_ami', 'coverage']:
        short = 'ami' if 'ami' in metric else 'coverage'
        for s in unique_series:
            data = {"X": unique_x}
            for gt in gt_types:
                means, stds = [], []
                for x in unique_x:
                    sub_slice = df[(df[series_col] == s) & (df[x_axis] == x) & (df['gt_type'] == gt)][metric]
                    sub_numeric = pd.to_numeric(sub_slice, errors='coerce')
                    means.append(sub_numeric.mean())
                    stds.append(sub_numeric.std() if not pd.isna(sub_numeric.std()) else 0.0)
                
                data[gt] = means
                data[gt + '_std'] = stds
            pd.DataFrame(data).to_csv(f"{output_dir}/{axis_name}_{short}_{s}.csv", index=False)

    thresholds = np.linspace(0, 1, 101)
    for s in unique_series:
        sub_s = df[(df[series_col] == s) & (df['gt_type'] != 'none') & (df['oracle'] == 'full')]
        if sub_s.empty: continue
        rows = []
        for t in thresholds:
            preds = sub_s['coverage'].apply(lambda x: "confounder" if x < t else "collider")
            rows.append({
                'threshold': t,
                'f1_macro': f1_score(sub_s['gt_type'], preds, labels=["confounder", "collider"], average="macro", zero_division=0),
                'tpr_confounder': (preds[sub_s['gt_type'] == 'confounder'] == 'confounder').mean(),
                'tpr_collider': (preds[sub_s['gt_type'] == 'collider'] == 'collider').mean()
            })
        pd.DataFrame(rows).to_csv(f"{output_dir}/{axis_name}_threshold_sweep_{s}.csv", index=False)

    print(f"✓ Processed {axis_name}: files saved to {output_dir}")
    

def process_main_ablation_oracle_wise(input_csv, output_dir="csv_for_figs/main_ablations"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    df['n_nodes'] = pd.to_numeric(df['n_nodes'], errors='coerce')
    df = df.dropna(subset=['n_nodes'])
    df['n_nodes'] = df['n_nodes'].astype(int)
    
    for col in ['sync_mech_ami', 'coverage', 'runtime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["subset_f1"] = df.apply(lambda row: subset_f1_from_sets(
        clean_to_set(row["subset_nodes"]), clean_to_set(row["expected_set"])), axis=1)

    unique_x = sorted(df['n_nodes'].unique())
    unique_oracles = ['full', 'partitions', 'perturbed', 'TOPIC']
    gt_types = ['none', 'confounder', 'collider']

    perf_metrics = {
        'f1': lambda sub: f1_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'accuracy': lambda sub: accuracy_score(sub['gt_type'], sub['predicted_gt_type']),
        'precision': lambda sub: precision_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'recall': lambda sub: recall_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'subset': lambda sub: sub[sub['gt_type'] != 'none']['subset_f1'].mean(),
        'runtime': lambda sub: pd.to_numeric(sub['runtime'], errors='coerce').mean()
    }

    for m_name, func in perf_metrics.items():
        data = {"X": unique_x}
        for oracle in unique_oracles:
            data[oracle] = [func(df[(df['n_nodes'] == x) & (df['oracle'] == oracle)]) for x in unique_x]
            data[oracle + '_std'] = [
                (df[(df['n_nodes'] == x) & (df['oracle'] == oracle)]
                .groupby('seed')
                .apply(lambda g: f1_score(
                    g['gt_type'], g['predicted_gt_type'], 
                    labels=gt_types, average="weighted", zero_division=0
                )).std() 
                if m_name == 'f1' else 
                df[(df['n_nodes'] == x) & (df['oracle'] == oracle)]
                .groupby('seed')
                .apply(func)
                .std())
                for x in unique_x
            ]
            data[oracle + '_std'] = [0.0 if pd.isna(s) else s for s in data[oracle + '_std']]
            print(oracle, m_name)
            if m_name == 'runtime':
                print(data[oracle], data[oracle + '_std'])
        pd.DataFrame(data).to_csv(f"{output_dir}/main_{m_name}.csv", index=False)
        

    for metric in ['sync_mech_ami', 'coverage']:
        short = 'ami' if 'ami' in metric else 'coverage'
        data = {"X": unique_x}
        for gt in gt_types:
            for oracle in unique_oracles:
                col_name = f"{gt}_{oracle}"
                data[col_name] = [
                    df[(df['oracle'] == oracle) & (df['n_nodes'] == x) & (df['gt_type'] == gt)][metric].mean() 
                    for x in unique_x
                ]
                data[col_name + '_std'] = [
                    df[(df['oracle'] == oracle) & (df['n_nodes'] == x) & (df['gt_type'] == gt)][metric].std() 
                    for x in unique_x
                ]
                data[col_name + '_std'] = [0.0 if pd.isna(s) else s for s in data[col_name + '_std']]
        pd.DataFrame(data).to_csv(f"{output_dir}/main_{short}.csv", index=False)
        
    sub_full = df[(df['oracle'] == 'full') & (df['gt_type'] != 'none')].copy()
    if not sub_full.empty:
        thresholds = np.linspace(0, 1, 101)
        sweep_rows = []
        for t in thresholds:
            preds = sub_full['coverage'].apply(lambda x: "confounder" if x < t else "collider")
            sweep_rows.append({
                'threshold': t,
                'f1_macro': f1_score(sub_full['gt_type'], preds, labels=["confounder", "collider"], average="macro", zero_division=0),
                'tpr_confounder': (preds[sub_full['gt_type'] == 'confounder'] == 'confounder').mean(),
                'tpr_collider': (preds[sub_full['gt_type'] == 'collider'] == 'collider').mean()
            })
        pd.DataFrame(sweep_rows).to_csv(f"{output_dir}/main_threshold_sweep.csv", index=False)
    
    print(f"✓ All files saved to {output_dir}")
    
    
def extract_node_ids(val):
    if not val or str(val).strip() in ["None", "[]", ""]:
        return set()
    try:
        clean_val = str(val).replace("np.int64", "").replace("np.float64", "")
        parsed = ast.literal_eval(clean_val)
        nodes = []
        stack = [parsed]
        while stack:
            curr = stack.pop()
            if isinstance(curr, (list, tuple, set)):
                stack.extend(list(curr))
            elif curr is not None:
                nodes.append(curr)
        return {f"X{str(n).strip()}" for n in nodes if str(n).strip().isdigit()}
    except Exception:
        return set()
    
def process_main_with_baselines_old(main_csv, baselines_csv, output_dir="csv_for_figs/baseline_plots"):
    os.makedirs(output_dir, exist_ok=True)
    df_main = pd.read_csv(main_csv)
    df_base = pd.read_csv(baselines_csv)
    for d in [df_main, df_base]:
        d['n_nodes'] = pd.to_numeric(d['n_nodes'], errors='coerce')
        d['seed'] = pd.to_numeric(d['seed'], errors='coerce')
        d['func'] = d['func'].astype(str)
        d.dropna(subset=['n_nodes', 'seed'], inplace=True)
        d['n_nodes'] = d['n_nodes'].astype(int)
        d['seed'] = d['seed'].astype(int)

    ours = df_main[df_main['oracle'] == 'full'].copy()
    ours['method'] = 'full'
    ours['predicted_nodes'] = ours['subset_nodes'].apply(extract_nodes)
    ours['expected_nodes'] = ours['expected_set'].apply(extract_nodes)
    ours['subset_f1'] = ours.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    ref_gt = ours[['n_nodes', 'seed', 'func', 'gt_type', 'expected_set']].drop_duplicates()
    baselines = pd.merge(df_base, ref_gt, on=['n_nodes', 'seed', 'func'], how='inner')

    baselines['predicted_nodes'] = baselines['predicted_pairs'].apply(extract_node_ids)
    baselines['expected_nodes'] = baselines['expected_set'].apply(extract_nodes)
    baselines['predicted_gt_type'] = baselines['predicted_nodes'].apply(lambda s: 'confounder' if len(s) > 0 else 'none')
    baselines['subset_f1'] = baselines.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    print(baselines['predicted_nodes'].head(),  baselines['expected_nodes'].head())
    combined = pd.concat([ours, baselines], ignore_index=True)
    unique_x = sorted(combined['n_nodes'].unique())
    method_map = {'full': 'full', 'coco_full': 'coco', 'fci_jci': 'fci-jci', 'fci_pooled': 'fci-pooled'}
    gt_types = ['none', 'confounder', 'collider']

    metrics = {
        'f1': lambda sub: f1_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'accuracy': lambda sub: accuracy_score(sub['gt_type'], sub['predicted_gt_type']),
        'precision': lambda sub: precision_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'recall': lambda sub: recall_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'subset': lambda sub: sub[sub['gt_type'] != 'none']['subset_f1'].mean(),
        'f1_nc': lambda sub: f1_score(
            sub[sub['gt_type'].isin(['none', 'confounder'])]['gt_type'], 
            sub[sub['gt_type'].isin(['none', 'confounder'])]['predicted_gt_type'], 
            labels=['none', 'confounder'], average="macro", zero_division=0),
        'f1_colliders': lambda sub: f1_score(
            sub[sub['gt_type'].isin(['none', 'collider'])]['gt_type'], 
            sub[sub['gt_type'].isin(['none', 'collider'])]['predicted_gt_type'], 
            labels=['none', 'confounder'], average="macro", zero_division=0),
        'runtime': lambda sub: pd.to_numeric(sub['runtime'], errors='coerce').mean()
    }
    
    for m_name, func in metrics.items():
        data = {"X": unique_x}
        for internal_name, label in method_map.items():
            sub_method = combined[combined['method'] == internal_name]
            
            means, stds = [], []
            for x in unique_x:
                slice_df = sub_method[sub_method['n_nodes'] == x]
                
                if slice_df.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                
                # 1. MEAN: Remains EXACTLY as you defined it
                means.append(func(slice_df))

                # 2. STD: Corresponding logic for each baseline metric
                if m_name == 'f1':
                    sd = slice_df.groupby('seed').apply(lambda g: f1_score(
                        g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                        average="weighted", zero_division=0)).std()
                
                elif m_name == 'f1_nc':
                    sub_nc = slice_df[slice_df['gt_type'].isin(['none', 'confounder'])]
                    if sub_nc.empty:
                        sd = 0.0
                    else:
                        sd = sub_nc.groupby('seed').apply(lambda g: f1_score(
                            g['gt_type'], g['predicted_gt_type'], labels=['none', 'confounder'], 
                            average="weighted", zero_division=0)).std()
                
                elif m_name == 'subset':
                    # Logic matches your mean: only look at non-'none' cases
                    sd = slice_df[slice_df['gt_type'] != 'none']['subset_f1'].std()
                
                elif m_name == 'runtime':
                    sd = pd.to_numeric(slice_df['runtime'], errors='coerce').std()
                
                else:
                    sd = 0.0

                stds.append(sd if not pd.isna(sd) else 0.0)

            data[label] = means
            data[f"{label}_std"] = stds
            
        pd.DataFrame(data).to_csv(f"{output_dir}/baseline_{m_name}.csv", index=False)

def process_main_with_baselines(main_csv, baselines_csv, output_dir="csv_for_figs/baseline_plots"):
    os.makedirs(output_dir, exist_ok=True)
    df_main = pd.read_csv(main_csv)
    df_base = pd.read_csv(baselines_csv)
    df_selection_baseline_pooled = pd.read_csv('multivariate/results/baseline_latent_selection_pooled.csv')
    df_selection_baseline_context = pd.read_csv('multivariate/results/baseline_latent_selection_context.csv')
    
    # 1. Clean and Standardize all dataframes
    for d in [df_main, df_base, df_selection_baseline_pooled, df_selection_baseline_context]:
        d['n_nodes'] = pd.to_numeric(d['n_nodes'], errors='coerce')
        d['seed'] = pd.to_numeric(d['seed'], errors='coerce')
        d['func'] = d['func'].astype(str).str.lower().str.strip()
        d.dropna(subset=['n_nodes', 'seed'], inplace=True)
        d['n_nodes'] = d['n_nodes'].astype(int)
        d['seed'] = d['seed'].astype(int)
        
        # Ensure label columns are clean strings to prevent sklearn "unknown" errors
        for col in ['gt_type', 'predicted_gt_type']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.lower().str.strip().fillna('none')

    # 2. Process Our Method
    ours = df_main[df_main['oracle'] == 'full'].copy()
    ours['method'] = 'full'
    ours['predicted_nodes'] = ours['subset_nodes'].apply(extract_nodes)
    ours['expected_nodes'] = ours['expected_set'].apply(extract_nodes)
    ours['subset_f1'] = ours.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    ref_gt = ours[['n_nodes', 'seed', 'func', 'gt_type', 'expected_set']].drop_duplicates()
    
    # 3. Process Baselines (CoCo, FCI)
    baselines = pd.merge(df_base, ref_gt, on=['n_nodes', 'seed', 'func'], how='inner')
    baselines['method'] = baselines['method'] # Ensure method col exists
    baselines['predicted_nodes'] = baselines['predicted_pairs'].apply(extract_node_ids)
    baselines['expected_nodes'] = baselines['expected_set'].apply(extract_nodes)
    baselines['predicted_gt_type'] = baselines['predicted_nodes'].apply(lambda s: 'confounder' if len(s) > 0 else 'none')
    baselines['subset_f1'] = baselines.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)
    
    dai_baseline_pooled = pd.merge(df_selection_baseline_pooled, ref_gt, on=['n_nodes', 'seed', 'func', 'gt_type'], how='inner')
    dai_baseline_pooled['method'] = dai_baseline_pooled['method']
    dai_baseline_pooled['predicted_nodes'] = dai_baseline_pooled['subset_nodes'].apply(clean_to_set)
    dai_baseline_pooled['expected_nodes'] = dai_baseline_pooled['expected_set'].apply(clean_to_set)
    dai_baseline_pooled['subset_f1'] = dai_baseline_pooled.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    dai_baseline_context = pd.merge(df_selection_baseline_context, ref_gt, on=['n_nodes', 'seed', 'func', 'gt_type'], how='inner')
    dai_baseline_context['method'] = dai_baseline_context['method']
    dai_baseline_context['predicted_nodes'] = dai_baseline_context['subset_nodes'].apply(clean_to_set)
    dai_baseline_context['expected_nodes'] = dai_baseline_context['expected_set'].apply(clean_to_set)
    dai_baseline_context['subset_f1'] = dai_baseline_context.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)


    combined = pd.concat([ours, baselines, dai_baseline_pooled, dai_baseline_context], ignore_index=True)
    combined['gt_type'] = combined['gt_type'].fillna('none')
    combined['predicted_gt_type'] = combined['predicted_gt_type'].fillna('none')

    unique_x = sorted(combined['n_nodes'].unique())
    method_map = {'full': 'full', 'coco_full': 'coco', 'fci_jci': 'fci-jci', 'fci_pooled': 'fci-pooled', 'latent-selection-pooled': 'dai_pooled', 'latent-selection-per-context-majority': 'dai_context'}
    gt_types = ['none', 'confounder', 'collider']

    metrics = {
        'f1': lambda sub: f1_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'accuracy': lambda sub: accuracy_score(sub['gt_type'], sub['predicted_gt_type']),
        'precision': lambda sub: precision_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'recall': lambda sub: recall_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'subset': lambda sub: sub[sub['gt_type'] != 'none']['subset_f1'].mean(),
        'f1_nc': None, # Special handling below
        'runtime': lambda sub: pd.to_numeric(sub['runtime'], errors='coerce').mean()
    }
    
    for m_name, func in metrics.items():
        data = {"X": unique_x}
        for internal_name, label in method_map.items():
            sub_method = combined[combined['method'] == internal_name]
            means, stds = [], []
            for x in unique_x:
                slice_df = sub_method[sub_method['n_nodes'] == x]
                if slice_df.empty:
                    means.append(np.nan); stds.append(0.0)
                    continue
                
                # --- MEAN CALCULATION ---
                if m_name == 'f1_nc':
                    target_labels = ['none', 'collider'] if 'latent-selection' in internal_name  else ['none', 'confounder']
                    eval_sub = slice_df[slice_df['gt_type'].isin(target_labels)]
                    if eval_sub.empty:
                        m_val = np.nan
                    else:
                        m_val = f1_score(eval_sub['gt_type'], eval_sub['predicted_gt_type'], 
                                        labels=target_labels, average="macro", zero_division=0)
                    means.append(m_val)
                else:
                    means.append(func(slice_df))

                # --- STD CALCULATION ---
                if m_name == 'f1':
                    sd = slice_df.groupby('seed').apply(lambda g: f1_score(
                        g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                        average="weighted", zero_division=0)).std()
                    
                elif m_name == 'accuracy':
                    sd = slice_df.groupby('seed').apply(lambda g: accuracy_score(
                        g['gt_type'], g['predicted_gt_type'])).std()
                elif m_name == 'precision':
                    sd = slice_df.groupby('seed').apply(lambda g: precision_score(
                        g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                        average="weighted")).std()
                elif m_name == 'recall':
                    sd = slice_df.groupby('seed').apply(lambda g: recall_score(
                        g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                        average="weighted")).std()
                elif m_name == 'f1_nc':
                    target_labels = ['none', 'collider'] if internal_name == 'latent-selection' else ['none', 'confounder']
                    sub_nc = slice_df[slice_df['gt_type'].isin(target_labels)]
                    if sub_nc.empty:
                        sd = 0.0
                    else:
                        # Fix: Ensure weighted labels match the target slice
                        sd = sub_nc.groupby('seed').apply(lambda g: f1_score(
                            g['gt_type'], g['predicted_gt_type'], labels=target_labels, 
                            average="weighted", zero_division=0)).std()
                
                elif m_name == 'subset':
                    sd = slice_df[slice_df['gt_type'] != 'none']['subset_f1'].std()
                elif m_name == 'runtime':
                    sd = pd.to_numeric(slice_df['runtime'], errors='coerce').std()
                else:
                    sd = 0.0

                stds.append(sd if not pd.isna(sd) else 0.0)

            data[label] = means
            data[f"{label}_std"] = stds
            
        pd.DataFrame(data).to_csv(f"{output_dir}/baseline_{m_name}.csv", index=False)
        
        
        
# def extract_nodes_ls(s):
#     """Helper to convert string representations of sets/lists to python sets."""
#     if pd.isna(s) or s in ['set()', '', '[]']: return set()
#     s = str(s).strip()
#     for char in ["{", "}", "[", "]", "'", '"']:
#         s = s.replace(char, "")
#     if not s: return set()
#     return set(x.strip() for x in s.split(',') if x.strip())



def process_latent_selection_comparison(main_csv, output_dir="csv_for_figs/latent_comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    df_main = pd.read_csv(main_csv)
    df_selection_baseline_pooled = pd.read_csv('multivariate/results/baseline_latent_selection_pooled.csv')
    df_selection_baseline_context = pd.read_csv('multivariate/results/baseline_latent_selection_context.csv')
    
    df_selection_baseline_pooled = df_selection_baseline_pooled[
        df_selection_baseline_pooled['scenario'].str.contains('linear', case=False, na=False)
    ]
    df_selection_baseline_context = df_selection_baseline_context[
        df_selection_baseline_context['scenario'].str.contains('linear', case=False, na=False)
    ]

    # ours_full = df_main[df_main['oracle'] == 'full'].copy()
    # ours_full['method'] = 'full'
    
    topic = df_main[df_main['oracle'] == 'TOPIC'].copy()
    topic['method'] = 'topic'

    # 3. Clean and Standardize
    dfs_to_process = [topic, df_selection_baseline_pooled, df_selection_baseline_context]
    for d in dfs_to_process:
        d['n_nodes'] = pd.to_numeric(d['n_nodes'], errors='coerce')
        d['seed'] = pd.to_numeric(d['seed'], errors='coerce')
        d.dropna(subset=['n_nodes', 'seed'], inplace=True)
        d['n_nodes'] = d['n_nodes'].astype(int)
        d['seed'] = d['seed'].astype(int)
        if 'func' in d.columns:
            d['func'] = d['func'].astype(str).str.lower().str.strip()
        for col in ['gt_type', 'predicted_gt_type']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.lower().str.strip().fillna('none')

    # Reference for ground truth merge
    ref_gt = topic[['n_nodes', 'seed', 'func', 'gt_type', 'expected_set']].drop_duplicates()
    
    # Process Subset F1 for our methods
    for df in [topic]:
        df['predicted_nodes'] = df['subset_nodes'].apply(extract_nodes)
        df['expected_nodes'] = df['expected_set'].apply(extract_nodes)
        df['subset_f1'] = df.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    # Merge and Process Subset F1 for Dai baselines
    dai_pooled = pd.merge(df_selection_baseline_pooled, ref_gt, on=['n_nodes', 'seed', 'func', 'gt_type'], how='inner')
    dai_pooled['method'] = 'dai_pooled'
    
    dai_context = pd.merge(df_selection_baseline_context, ref_gt, on=['n_nodes', 'seed', 'func', 'gt_type'], how='inner')
    dai_context['method'] = 'dai_context'

    for df in [dai_pooled, dai_context]:
        df['predicted_nodes'] = df['subset_nodes'].apply(extract_nodes)
        df['expected_nodes'] = df['expected_set'].apply(extract_nodes)
        df['subset_f1'] = df.apply(lambda r: subset_f1_from_sets(r['predicted_nodes'], r['expected_nodes']), axis=1)

    # 4. Combine and Compute Metrics
    combined = pd.concat([topic, dai_pooled, dai_context], ignore_index=True)
    unique_x = sorted(combined['n_nodes'].unique())
    method_map = {
        'topic': 'topic',
        'dai_pooled': 'dai_pooled',
        'dai_context': 'dai_context'
    }
    gt_types = ['none', 'confounder', 'collider']

    metrics = {
        'f1': lambda sub: f1_score(sub['gt_type'], sub['predicted_gt_type'], labels=gt_types, average="macro", zero_division=0),
        'accuracy': lambda sub: accuracy_score(sub['gt_type'], sub['predicted_gt_type']),
        'subset': lambda sub: sub[sub['gt_type'] != 'none']['subset_f1'].mean(),
        'runtime': lambda sub: pd.to_numeric(sub['runtime'], errors='coerce').mean()
    }

    for m_name, func in metrics.items():
        data = {"X": unique_x}
        for internal_name, label in method_map.items():
            sub = combined[combined['method'] == internal_name]
            means, stds = [], []
            for x in unique_x:
                slice_df = sub[sub['n_nodes'] == x]
                if slice_df.empty:
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(func(slice_df))
                    if m_name == 'subset':
                        sd = slice_df[slice_df['gt_type'] != 'none']['subset_f1'].std()
                    elif m_name == 'f1':
                        sd = slice_df.groupby('seed').apply(lambda g: f1_score(
                            g['gt_type'], g['predicted_gt_type'], labels=gt_types, 
                            average="weighted")).std()
                    elif m_name == 'accuracy':
                        sd = slice_df.groupby('seed').apply(lambda g: accuracy_score(g['gt_type'], g['predicted_gt_type'])).std()
                    elif m_name == 'runtime':
                        sd = pd.to_numeric(slice_df['runtime'], errors='coerce').std()
                    else:
                        sd = 0.0
                    stds.append(sd if not pd.isna(sd) else 0.0)
            
            data[label] = means
            data[f"{label}_std"] = stds
        
        pd.DataFrame(data).to_csv(f"{output_dir}/latent_{m_name}.csv", index=False)
        
if __name__ == "__main__":
    for files in [ 'ablation_main.csv', 'ablation_affected_nodes.csv', 'ablation_contexts.csv', 
                'ablation_perturbed_graph.csv', 'ablation_samples.csv', 'ablation_shifts.csv', 
                'ablation_sparsity.csv']:

        if 'main' in files:
            optimize_and_update_predictions('multivariate/results/' + files)
            process_main_ablation_oracle_wise('multivariate/results/' +files)
            process_main_with_baselines(main_csv='multivariate/results/' +files, baselines_csv='multivariate/results/baselines.csv')
            
        else:
            process_any_ablation('multivariate/results/' +files)

    # For Dai. et al.
    process_latent_selection_comparison('multivariate/results/ablation_main.csv')