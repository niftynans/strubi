import numpy as np
import pandas as pd
import networkx as nx
import json
import os, sys
import random
import datetime
import time
import traceback
import warnings
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")

from coco.co_co import CoCo
from coco.co_test_types import CoCoTestType, CoShiftTestType, CoDAGType
from coco.mi_sampling import Sampler
from coco.fci import FCI_JCI
from experiments.results_coco import MethodType

import os
import re
import random
from pathlib import Path
from collections import defaultdict, Counter



def map_predictions_to_edges(retrieved_raw, lt):
    predicted_edges = []
    if not retrieved_raw:
        return predicted_edges
        
    if lt == 'confounder':
        for pair in retrieved_raw:
            if len(pair) == 2:
                u_idx, v_idx = pair
                predicted_edges.append(('Z0', f'X{u_idx}'))
                predicted_edges.append(('Z0', f'X{v_idx}'))
                
    elif lt == 'collider':
        for u_idx in retrieved_raw:
            predicted_edges.append((f'X{u_idx}', 'S0'))
            
    return predicted_edges






def plot_baseline_discovery(G_true_int, predicted_latent_edges, meta, save_path):
    n = meta['n']
    lc = meta['lc']

    mapping = {i: f"X{i}" for i in range(n)}
    for i in range(lc[0]): mapping[n + i] = f"Z{i}"
    for i in range(lc[1]): mapping[n + lc[0] + i] = f"S{i}"
    
    G_full = nx.relabel_nodes(G_true_int, mapping)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(G_full, seed=42) 
    
    node_colors = []
    node_shapes = {}
    for node in G_full.nodes():
        if str(node).startswith('Z'):
            node_shapes[node] = 's'
            node_colors.append('lightgreen')
        elif str(node).startswith('S'):
            node_shapes[node] = 'D'
            node_colors.append('salmon')
        else:
            node_shapes[node] = 'o'
            node_colors.append('lightblue')

    ax = axes[0]
    ax.set_title("Ground Truth (Reality)", fontsize=16, fontweight='bold')
    for shape in set(node_shapes.values()):
        nlist = [n_node for n_node in G_full.nodes() if node_shapes[n_node] == shape]
        colors = [c for n_node, c in zip(G_full.nodes(), node_colors) if node_shapes[n_node] == shape]
        nx.draw_networkx_nodes(G_full, pos, nodelist=nlist, node_shape=shape, node_color=colors, ax=ax, node_size=800, edgecolors='white')
    nx.draw_networkx_labels(G_full, pos, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G_full, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    ax = axes[1]
    ax.set_title("Refined Causal Model (Discovery)", fontsize=16, fontweight='bold')
    for shape in set(node_shapes.values()):
        nlist = [n_node for n_node in G_full.nodes() if node_shapes[n_node] == shape]
        colors = [c for n_node, c in zip(G_full.nodes(), node_colors) if node_shapes[n_node] == shape]
        nx.draw_networkx_nodes(G_full, pos, nodelist=nlist, node_shape=shape, node_color=colors, ax=ax, node_size=800, edgecolors='white')
    nx.draw_networkx_labels(G_full, pos, font_weight='bold', ax=ax)
    
    obs_edges = [(u, v) for u, v in G_full.edges() if not (str(u).startswith(('Z', 'S')) or str(v).startswith(('Z', 'S')))]
    nx.draw_networkx_edges(G_full, pos, edgelist=obs_edges, edge_color='gray', arrows=True, arrowsize=20, ax=ax, alpha=0.5)
    
    if predicted_latent_edges:
        nx.draw_networkx_edges(G_full, pos, edgelist=predicted_latent_edges, edge_color='red', style='dashed', width=2.5, arrows=True, arrowsize=25, ax=ax)
        
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # LEGEND
    handles = [
        mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=10, label='Observed (X)'),
        mlines.Line2D([], [], color='lightgreen', marker='s', linestyle='None', markersize=10, label='Latent (Z)'),
        mlines.Line2D([], [], color='salmon', marker='D', linestyle='None', markersize=10, label='Selection (S)')
    ]
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_scenario(scenario_id, dataset_dir="datasets"):
    print(f"  Loading scenario: {scenario_id}")
    out = f"{dataset_dir}/{scenario_id}"
    
    # 1. Load Data (Using PyArrow engine for faster CSV reading)
    adj   = np.load(f"{out}/adj.npy")
    adj_f = np.load(f"{out}/adj_full.npy")
    X     = pd.read_csv(f"{out}/X.csv", engine='pyarrow')
    with open(f"{out}/meta.json") as f:
        meta = json.load(f)

    n             = meta['n']
    lc            = meta['lc']          
    n_confounders = lc[0]
    n_colliders   = lc[1]
    start_conf    = n
    start_coll    = n + n_confounders

    if not np.allclose(adj, adj_f[:n, :n]):
        print(f"  [!] WARNING: adj.npy differs from adj_full[:n,:n] for {scenario_id}")

    # 2. Extract Latent Connections
    nodes_confounded = []
    for z_idx in range(start_conf, start_conf + n_confounders):
        children = list(np.where(adj_f[z_idx, :n] != 0)[0])
        if children: nodes_confounded.append(children)
    print(nodes_confounded)

    nodes_selection_parents = []
    for s_idx in range(start_coll, start_coll + n_colliders):
        parents = list(np.where(adj_f[:n, s_idx] != 0)[0])
        if parents: nodes_selection_parents.append(parents)
    print(nodes_selection_parents)

    # 3. Clean and Sort Contexts
    # Force numeric contexts to prevent alphabetical sorting errors
    X['Context'] = pd.to_numeric(X['Context'])
    
    # 4. Standardize Observed Columns (Crucial for CoCo's KCI tests)
    obs_cols = [f"X{i}" for i in range(n)]
    X[obs_cols] = (X[obs_cols] - X[obs_cols].mean()) / (X[obs_cols].std() + 1e-9)

    # 5. Fast Matrix Construction via GroupBy
    grouped = X.groupby('Context')
    
    # Find the smallest context size to determine our cap
    min_samples = grouped.size().min()
    cap = min(min_samples, 500)
    
    if cap < 200:
        print(f"  [!] WARNING: Cap is low ({cap}). Statistical tests may struggle.")

    # Build D in a single pass through the groups, grabbing the first `cap` rows
    # Sort=True in groupby (default) guarantees ascending context order
    D = np.array([group[obs_cols].values[:cap] for _, group in grouped])  

    # 6. Build Graph Objects
    G_obs = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    G_true = nx.from_numpy_array(adj_f, create_using=nx.DiGraph)
    
    class MinimalDAG: pass
    dag_obj = MinimalDAG()
    dag_obj.G = G_obs
    dag_obj.G_true = G_true
    dag_obj.nodes_confounded = nodes_confounded
    dag_obj.nodes_selection_parents = nodes_selection_parents
    dag_obj.maps_nodes = {node: np.zeros(len(grouped), dtype=int) for node in G_obs.nodes()}
    dag_obj.maps_nodes_star = {node: np.zeros(len(grouped), dtype=int) for node in G_true.nodes()}

    return D, G_obs, G_true, dag_obj, meta


def run_coco(D, G_obs, dag_obj):
    print("In run_coco")
    sampler = Sampler()
    results = {}
    for variant, dag_discovery in [('coco_full', CoDAGType.SKIP)]:#, ('coco_mec',  CoDAGType.MSS)]:
        try:
            t0   = time.time()
            coco = CoCo(D, list(G_obs.nodes()), co_test=CoCoTestType.MI_ZTEST,
                        shift_test=CoShiftTestType.PI_KCI, dag_discovery=dag_discovery,
                        sampler=sampler, n_components=None, dag=dag_obj)
            results[variant] = {'model': coco, 'runtime': time.time() - t0, 'error': None}
            print(f"  [{variant}] done in {results[variant]['runtime']:.1f}s")
        except Exception as e:
            results[variant] = {'model': None, 'runtime': None, 'error': str(e)}
    return results


def run_fci(D, G_obs, dag_obj):
    print("  Running FCI variants...")
    results = {}
    for variant, method in [('fci_jci', MethodType.ORACLE_DAG), ('fci_pooled', MethodType.FCI_POOLED)]: #('fci_jci_full', MethodType.FCI_JCI_FULL),  ('fci_jci', MethodType.FCI_JCI), ('fci_pooled', MethodType.FCI_POOLED),
        try:
            t0  = time.time()
            fci = FCI_JCI(D, dag_obj.G, dag_obj.G_true, dag_obj, independence_test='fisherz', method=method)
            results[variant] = {'model': fci, 'method': method, 'runtime': time.time() - t0, 'error': None}
            print(f"  [{variant}] done in {results[variant]['runtime']:.1f}s")
        except Exception as e:
            results[variant] = {'model': None, 'method': method, 'runtime': None, 'error': str(e)}
    return results


def make_row(scenario_id, variant, meta, runtime, error, tp, fp, tn, fn, f1, retrieved_edges=""):
    return {
        'scenario_id': scenario_id, 'method': variant, 'n_nodes': meta['n'],
        'seed': meta['seed'], 'func': meta['func'], 'lc': str(meta['lc']),
        'lt': meta['lt'], 'nc': meta['nc'], 'runtime': runtime, 'error': error,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'f1': f1, 'retrieved_edges': retrieved_edges
    }


def evaluate_coco(results, dag_obj, meta, scenario_id, G_true, plot_dir):
    rows = []
    for variant, res in results.items():
        tp = fp = tn = fn = f1 = tpr_val = fpr_val = None
        jacc = ari = ami = None
        tp_s = fp_s = tn_s = fn_s = f1_s = None
        tp_adj = fp_adj = tn_adj = fn_adj = f1_adj = None
        error = res['error']
        predicted_pairs = []

        if res['model'] is not None:
            try:
                tp, fp, tn, fn, f1, tpr_val, fpr_val = \
                    res['model'].eval_estimated_edges(dag_obj)
                f1 = round(f1, 4)

                jacc, ari, ami, \
                tp_s, fp_s, tn_s, fn_s, f1_s, \
                tp_adj, fp_adj, tn_adj, fn_adj, f1_adj = \
                    res['model'].eval_estimated_graph_cuts(dag_obj)

                for v in [jacc, ari, ami, f1_s, f1_adj]:
                    v = round(float(v), 4) if v is not None else None

                predicted_pairs = [
                    (ni, nj)
                    for ni in res['model'].sim_01
                    for nj in res['model'].sim_01[ni]
                    if res['model'].sim_01[ni][nj]
                ]

                if meta['lt'] in ['confounder', 'collider'] and plot_dir:
                    plot_path = f"{plot_dir}/{scenario_id}_{variant}.png"
                    predicted_latent_edges = [(f"X{ni}", f"X{nj}") for ni, nj in predicted_pairs]
                    plot_baseline_discovery(G_true, predicted_latent_edges, meta, plot_path)

            except Exception as e:
                traceback.print_exc()
                error = f"eval_failed: {e}"
                predicted_pairs = []

        row = {
            'scenario_id':   scenario_id,
            'method':        variant,
            'n_nodes':       meta['n'],
            'seed':          meta['seed'],
            'func':          meta['func'],
            'lc':            str(meta['lc']),
            'lt':            meta['lt'],
            'nc':            meta['nc'],
            'runtime':       res['runtime'],
            'error':         error,
            # pairwise
            # 'tp':            tp,
            # 'fp':            fp,
            # 'tn':            tn,
            # 'fn':            fn,
            # 'f1_pairwise':   f1,
            # 'tpr':           round(float(tpr_val), 4) if tpr_val is not None else None,
            # 'fpr':           round(float(fpr_val), 4) if fpr_val is not None else None,
            # set-level
            # 'f1_set':        round(float(f1_s),   4) if f1_s   is not None else None,
            # 'f1_adjusted':   round(float(f1_adj), 4) if f1_adj is not None else None,
            # 'jaccard':       round(float(jacc),   4) if jacc   is not None else None,
            # 'ari':           round(float(ari),    4) if ari    is not None else None,
            # 'ami_sets':      round(float(ami),    4) if ami    is not None else None,
            # predicted pairs
            'predicted_pairs': str(predicted_pairs),
        }
        rows.append(row)
        # print(f"  [{variant}] "
        #       f"F1_pair={row['f1_pairwise']} "
        #       f"F1_adj={row['f1_adjusted']} "
        #       f"F1_set={row['f1_set']} "
        #       f"TP={tp} FP={fp} FN={fn} "
        #       f"error={error}")

    return rows



def evaluate_fci(results, dag_obj, meta, scenario_id, G_true, plot_dir):
    rows = []
    for variant, res in results.items():
        # Pairwise metrics (Standard for FCI)
        tp = fp = tn = fn = f1_pairwise = tpr_val = fpr_val = None
        # Set-level metrics (Usually None for FCI, but kept for column alignment)
        jacc = ari = ami = f1_set = f1_adj = None
        
        error = res['error']
        predicted_pairs = []

        if res['model'] is not None:
            try:
                # Standard FCI evaluation for confounded pairs
                # Assuming this returns (tp, fp, tn, fn, f1)
                eval_res = res['model'].eval_confounded(dag_obj, res['method'])
                tp, fp, tn, fn, f1_pairwise = eval_res
                
                if f1_pairwise is not None:
                    f1_pairwise = round(float(f1_pairwise), 4)

                # Extract predicted edges for plotting and storage
                if hasattr(res['model'], 'estimated_confounders'):
                    # Ensure format is a list of tuples: [(i, j), ...]
                    predicted_pairs = res['model'].estimated_confounders
                
                # Plotting logic mirrored from COCO
                if meta['lt'] in ['confounder', 'collider'] and plot_dir:
                    plot_path = f"{plot_dir}/{scenario_id}_{variant}.png"
                    # Format for plotter: list of strings like [("X0", "X2"), ...]
                    plot_edges = [(f"X{ni}", f"X{nj}") for ni, nj in predicted_pairs]
                    plot_baseline_discovery(G_true, plot_edges, meta, plot_path)

            except Exception as e:
                traceback.print_exc()
                error = f"eval_failed: {e}"
                predicted_pairs = []

        # Mimic the exact row structure of evaluate_coco
        row = {
            'scenario_id':   scenario_id,
            'method':        variant,
            'n_nodes':       meta['n'],
            'seed':          meta['seed'],
            'func':          meta['func'],
            'lc':            str(meta['lc']),
            'lt':            meta['lt'],
            'nc':            meta['nc'],
            'runtime':       res['runtime'],
            'error':         error,
            # pairwise (FCI's primary output)
            # 'tp':            tp,
            # 'fp':            fp,
            # 'tn':            tn,
            # 'fn':            fn,
            # 'f1_pairwise':   f1_pairwise,
            # 'tpr':           round(float(tpr_val), 4) if tpr_val is not None else None,
            # 'fpr':           round(float(fpr_val), 4) if fpr_val is not None else None,
            # set-level (mirrored placeholders)
            # 'f1_set':        round(float(f1_set),   4) if f1_set   is not None else None,
            # 'f1_adjusted':   round(float(f1_adj), 4) if f1_adj is not None else None,
            # 'jaccard':       round(float(jacc),   4) if jacc   is not None else None,
            # 'ari':           round(float(ari),    4) if ari    is not None else None,
            # 'ami_sets':      round(float(ami),    4) if ami    is not None else None,
            # predicted pairs
            'predicted_pairs': str(predicted_pairs),
        }
        
        rows.append(row)
        
        # Consistent console logging
        # print(f"  [{variant}] "
        #       f"F1_pair={row['f1_pairwise']} "
        #       f"TP={tp} FP={fp} FN={fn} "
        #       f"error={error}")

    return rows


def run_coco_light(X_biased, G_obs, alpha=0.05):
    node_names = [c for c in X_biased.columns if c != 'Context']
    grouped = X_biased.groupby('Context')
    
    min_samples = grouped.size().min()
    cap = min(min_samples, 2000)
    
    Dc = np.array([group[node_names].values[:cap] for _, group in grouped])
    
    print(f"  CoCo Data: {Dc.shape[0]} contexts, {Dc.shape[1]} samples, {Dc.shape[2]} nodes")

    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    DAG_SEARCH = CoDAGType.SKIP 

    mapping = {name: i for i, name in enumerate(node_names)}
    G_int = nx.relabel_nodes(G_obs, mapping)
    
    # class DagWrapper:
    #     def __init__(self, G): self.G = G
    
    class DagWrapper:
        def __init__(self, G, n_contexts):
            self.G = G
            # CoCo expects these dictionaries to store results per node
            self.maps_nodes = {
                i: np.zeros(n_contexts, dtype=int) for i in G.nodes()
            }
            # If your CoCo version checks for an oracle/true graph:
            self.maps_nodes_star = {
                i: np.zeros(n_contexts, dtype=int) for i in G.nodes()
            }
    dg = DagWrapper(G_int, Dc.shape[0])
    coco = CoCo(
        Dc, 
        G_int, 
        Sampler(), 
        CONFOUNDING_TEST, 
        SHIFT_TEST, 
        DAG_SEARCH,
        n_components=1, 
        dag=dg, 
        node_nms=node_names, 
        alpha_shift_test=alpha
    )

    coco._estimated_graph_cuts_n(1)
    return coco, node_names

def get_trimmed_grid_from_files(directory):
    file_pattern = re.compile(r"n(?P<n>\d+)_seed(?P<seed>\d+)_func(?P<func>.+)_lc(?P<lc>\d{2})")
    experimental_grid = []
    path = Path(directory)

    if not path.exists():
        print(f"Error: Directory '{directory}' not found.")
        return []

    # 1. Discover and parse all files
    for file_path in path.iterdir():
        match = file_pattern.match(file_path.name)
        if match:
            # Extract data using the named groups in the regex
            data = match.groupdict()
            
            experimental_grid.append({
                'n': int(data['n']),
                'seed': int(data['seed']),
                'func': data['func'],
                # Convert "10" -> (1, 0)
                'lc': (int(data['lc'][0]), int(data['lc'][1])),
                'filename': file_path.name,
                'full_path': str(file_path)
            })

    # 2. Group scenarios by number of nodes (n)
    by_n = defaultdict(list)
    for cfg in experimental_grid:
        by_n[cfg['n']].append(cfg)

    # 3. Apply the allocation (Sampling)
    trimmed_grid = []
    for n, k in allocation.items():
        if n in by_n:
            # Sample k items, or fewer if total available is less than k
            sample_size = min(k, len(by_n[n]))
            trimmed_grid.extend(random.sample(by_n[n], sample_size))
        # else:
        #     print(f"Warning: No matching files found in '{directory}' for n={n}")

    return trimmed_grid

if __name__ == "__main__":
    synthetic = True
    real_world = False
    
    if synthetic:
        # DATASET_DIR = "datasets"
        RESULTS_DIR = "results"
        PLOT_DIR = "results/plots"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)

        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # csv_out   = f"{RESULTS_DIR}/baselines_{timestamp}.csv"
        csv_out   = f"{RESULTS_DIR}/baselines.csv"


        DATASET_DIR = "datasets"
        allocation = {3: 25, 5: 25, 7: 25, 9: 25, 11: 25, 13: 25, 15: 25}
        random.seed(6)
        
        all_rows = []
        trimmed_grid = get_trimmed_grid_from_files(DATASET_DIR)
    
        print(len(trimmed_grid))
        print(f"Total scenarios found and sampled: {len(trimmed_grid)}")
        print("Distribution by node size:", Counter(cfg['n'] for cfg in trimmed_grid))

        for i, scenario in enumerate(reversed(trimmed_grid)):
            seed = scenario['seed']; lc = scenario['lc']
            func = scenario['func']; n = scenario['n']
            scenario_id = f"n{n}_seed{seed}_func{func}_lc{lc[0]}{lc[1]}"
        
            print(f"\n[{i+1}/{len(trimmed_grid)}] {scenario_id}")
            if not os.path.exists(f"{DATASET_DIR}/{scenario_id}/X.csv"):
                print(f"  SKIPPED — dataset not found")
                continue
            D, G_obs, G_true, dag_obj, meta = load_scenario(scenario_id, DATASET_DIR)

            # print("  --- CoCo ---")
            start_time = time.time()
            coco_results = run_coco(D, G_obs, dag_obj)
            end_time = time.time()
            print(f"  CoCo variants completed in {end_time - start_time:.1f}s")
            rows = evaluate_coco(coco_results, dag_obj, meta, scenario_id, G_true, PLOT_DIR)
            all_rows.extend(rows)

            # print("  --- FCI ---")
            fci_results = run_fci(D, G_obs, dag_obj)
            rows = evaluate_fci(fci_results, dag_obj, meta, scenario_id, G_true, PLOT_DIR)
            all_rows.extend(rows)

            df_inc = pd.DataFrame(all_rows[-3:])
            df_inc.to_csv(csv_out, mode='a', header=not os.path.exists(csv_out), index=False)
            break

        print(f"\nDone. Results saved to: {csv_out}")
        df_all = pd.DataFrame(all_rows)
        print(df_all[['scenario_id', 'method', 'lt', 'f1', 'error']].to_string())
        
    if real_world:
        nodes = ['PKC'] #'Raf', 'Mek', 'Erk', 'PKA', 'Akt', 'P38', 'Jnk', 'Plcg', 'PIP2', 'PIP3'
        scenarios = []
        for node in nodes:
            scenarios.extend(
                [('cytometry', 'collider',  node),
                ('cytometry', 'confounder', node),
                ('cytometry', '', node)]
            )
        #     scenarios = [
        #     # ('light_tunnel', 'collider', 'probe'),
        #     ('light_tunnel', 'confounder', 'red'),
        #     ('light_tunnel', '', 'current')
        # ]
        
        for source, lt, target in scenarios:
            X_raw = pd.read_csv('sachs_processed.csv')
            
            # with open(f"G_obs.json", 'r') as f:
            #     G_obs_dict = json.load(f)
            # G_full = nx.from_dict_of_lists(G_obs_dict, create_using=nx.DiGraph())
            edges = [('Plcg', 'PIP2'), ('Plcg', 'PKA'), ('PIP2', 'PKC'), ('PKC', 'Raf'), ('PKC', 'PKA'), 
                    ('PKC', 'P38'), ('PKC', 'Jnk'), ('PKA', 'Erk'), ('Raf', 'Mek'), ('Mek', 'Erk'), 
                    ('PIP3', 'Akt'), ('PIP3', 'Mek'), ('PIP3', 'P38'), ('PIP3', 'PKA'), ('PIP3', 'Jnk'), 
                    ('Erk', 'P38'), ('Erk', 'Akt'), ('Erk', 'PIP2')]
            
            G_full = nx.DiGraph()
            G_full.add_nodes_from(nodes)
            G_full.add_edges_from(edges)
            
            
            print(X_raw.head())
            print(G_full.edges())
            
            if target not in X_raw.columns:
                continue
            if lt == 'confounder':
                latent_label = 'Z' 
            elif lt == 'collider':
                latent_label = 'S'
            else:
                latent_label = str(target)
            G_true = nx.relabel_nodes(G_full, {target: latent_label}, copy=True)        
            G_obs = G_full.copy()
            if lt == '':
                print("Unbiased, hence not dropping the node")
            else:
                G_obs.remove_node(target)
            if lt == 'confounder':
                X_biased = X_raw.drop(columns=[target])
            elif lt == 'collider':
                clean_target = X_raw[target].replace([np.inf, -np.inf], np.nan).dropna()
                cutoff_val = 0.80
                thresh = clean_target.quantile(cutoff_val)
                X_biased = X_raw[X_raw[target] > thresh].copy()
            else:
                X_biased = X_raw.copy()


            # X_biased, G_obs
            variances = X_biased.std()
            nodes_to_skip = variances[variances == 0].index.tolist()
            
            if nodes_to_skip:
                print(f"  [Skipping] Zero-variance nodes: {nodes_to_skip}")
                X_biased = X_biased.drop(columns=nodes_to_skip)
                G_obs.remove_nodes_from(nodes_to_skip)

            if G_obs.number_of_nodes() == 0:
                print("  [Warning] All nodes were skipped due to zero variance.")
                continue
                
            print(f"  Starting CoCo Discovery on {len(X_biased.columns)-1} active nodes...")
            coco_result, current_nodes = run_coco_light(X_biased, G_obs)

            print(f"\nRESULTS for {source} | {lt}:")
            for i, node_name in enumerate(current_nodes):
                if np.any(coco_result.maps_estimated[i] != 0):
                    print(f"  [SHIFT] {node_name} mechanism changed across contexts.")

            if len(coco_result.estimated_cuts):
                for cluster in coco_result.estimated_cuts:
                    discovered_nodes = [current_nodes[j] for j in cluster]
                    print(f"  [CUT] Discovered Bias Cluster: {discovered_nodes}")
            else:
                print("  [CLEAN] No significant structural bias subsets found.")