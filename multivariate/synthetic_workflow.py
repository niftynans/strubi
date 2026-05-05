import os
import sys

import datetime
import argparse
import time

import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter


import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from info import run_multivariate_kci, draw_causal_comparison, visualize_mdl_graph
from mv_dgp_v4 import DAGConfoundedWithSelection

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import kruskal
from itertools import product

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag


repo_path = "struct_bias/topic"
src_path = os.path.join(repo_path, "src")
if os.path.exists(src_path):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
else:
    print(f"package error")
try:
    from topic.topic import Topic
except ImportError as e:
    print(f"package error")
    if os.path.exists(src_path):
        print(f"Contents of src: {os.listdir(src_path)}")

class LoggerWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)      
    def flush(self):
        self.terminal.flush()
        self.log.flush()





def _enumerate_dags_in_mec(cpdag_adj, n):
    undirected = []
    fixed_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if cpdag_adj[i, j] != 0 and cpdag_adj[j, i] != 0:
                undirected.append((i, j))
            elif cpdag_adj[i, j] != 0:
                fixed_edges.append((i, j))
            elif cpdag_adj[j, i] != 0:
                fixed_edges.append((j, i))
    valid_adjs = []
    for orientations in product([0, 1], repeat=len(undirected)):
        adj = np.zeros((n, n))
        for i, j in fixed_edges: adj[i, j] = 1
        for idx, orient in enumerate(orientations):
            u, v = undirected[idx]
            if orient == 0: adj[u, v] = 1
            else: adj[v, u] = 1        
        if nx.is_directed_acyclic_graph(nx.DiGraph(adj)):
            if _check_v_structures_preserved(nx.DiGraph(adj), cpdag_adj):
                valid_adjs.append(adj)
    return valid_adjs






def _check_v_structures_preserved(dag, cpdag_adj):
    for n in dag.nodes():
        parents = list(dag.predecessors(n))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p1, p2 = parents[i], parents[j]
                if cpdag_adj[p1, p2] == 0 and cpdag_adj[p2, p1] == 0:
                    if not (cpdag_adj[p1, n] != 0 and cpdag_adj[n, p1] == 0):
                        return False
    return True





def _is_mechanism_shifting(df, target_idx, parent_indices, contexts, alpha=0.05):
    y = df.iloc[:, target_idx].values
    if len(parent_indices) > 0:
        X_vals = df.iloc[:, parent_indices].values
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X_vals, y)
        residuals = y - model.predict(X_vals)
    else:
        residuals = y - np.mean(y)
    unique_ctx = np.unique(contexts)
    groups = [residuals[contexts == c] for c in unique_ctx if len(residuals[contexts == c]) > 5]
    if len(groups) < 2: return False
    _, p = kruskal(*groups)
    return p < alpha






def mss_refine_from_mec(X, true_adj, n_nodes):
    nodes = [GraphNode(f'X{i}') for i in range(n_nodes)]
    dag_obj = Dag(nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if true_adj[i, j] == 1:
                dag_obj.add_directed_edge(nodes[i], nodes[j])
    
    cpdag_obj = dag2cpdag(dag_obj)
    cpdag_matrix = cpdag_obj.graph 

    candidate_adjs = _enumerate_dags_in_mec(cpdag_matrix, n_nodes)
    
    # --- SAFETY HATCH ---
    if not candidate_adjs:
        print("WARNING: No candidates found in MEC. Falling back to True DAG.")
        # Return the true_adj as the "best" and only guess
        G_o = nx.from_numpy_array(true_adj, create_using=nx.DiGraph)
        mapping = {i: f'X{i}' for i in range(n_nodes)}
        return nx.relabel_nodes(G_o, mapping)

    best_adj = None
    min_mss_score = float('inf')
    contexts = X['Context'].values
    data_df = X.drop(columns=['Context'], errors='ignore')
    
    print(f"Calculating Mechanism Shift Score (MSS) for {len(candidate_adjs)} candidates...")
    for adj_cand in candidate_adjs:
        current_mss = 0
        for i in range(n_nodes):
            parents = np.where(adj_cand[:, i] == 1)[0]
            if _is_mechanism_shifting(data_df, i, parents, contexts):
                current_mss += 1
        
        if current_mss < min_mss_score:
            min_mss_score = current_mss
            best_adj = adj_cand
        
        if min_mss_score == 0: 
            break

    # Final check just in case best_adj is still None
    if best_adj is None:
        best_adj = true_adj

    G_o = nx.from_numpy_array(best_adj, create_using=nx.DiGraph)
    mapping = {i: f'X{i}' for i in range(n_nodes)}
    return nx.relabel_nodes(G_o, mapping)


        
def print_graph_in_terminal(adj_matrix, node_names, start_latent, is_confounder_mode=True):
    label = "[L]" if is_confounder_mode else "[S]"
    print(f"\n--- Causal Graph Structure ({'Confounding' if is_confounder_mode else 'Selection'}) ---")
    n = len(node_names)
    for i in range(n):
        node_type = label if i >= start_latent else "[O]"
        children = [node_names[j] for j in range(n) if adj_matrix[i, j] != 0]
        if children:
            child_str = ", ".join(children)
            print(f"{node_type} {node_names[i]:<3}  -->  {child_str}")
        else:
            print(f"{node_type} {node_names[i]:<3}  (Leaf Node)")
    print("------------------------------\n")




def plot_context_data(df, func_name, n_nodes, lt, highlight_nodes=None, active_contexts=None, example=0):
    obs_cols = [f"X{i}" for i in range(n_nodes)]
    highlight_nodes = highlight_nodes or []
    active_contexts = active_contexts or []
    
    n_contexts = 3  * n_nodes
    n_cols = n_nodes
    n_rows = 3

    df_plot = df.copy()
    df_plot['Context'] = df_plot['Context'].astype(str)

    for col in obs_cols:
        df_plot[col] = df_plot[col].clip(-5, 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 12), sharex=True, sharey=True)
    fig.suptitle(f"Type: {lt} | Function: {func_name} | Nodes: {n_nodes} (Clipped +/- 5)", fontsize=22, y=0.98)
    
    axes = axes.flatten()
    
    for c in range(n_contexts):
        ax = axes[c]
        context_data = df_plot[df_plot['Context'] == str(c)]
        
        if c in active_contexts:
            ax.set_facecolor('#FFF9C4') # Active Tint
            title_color = 'red'
            title_suffix = " (ACTIVE)"
        else:
            title_color = 'black'
            title_suffix = ""

        if context_data.empty:
            ax.text(0.5, 0.5, "EMPTY", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Ctx {c}", fontsize=10)
            continue

        melted = context_data.melt(value_vars=obs_cols, var_name='Node', value_name='Value')        
        palette = {node: "#FF7043" if node in highlight_nodes else "#4FC3F7" for node in obs_cols}
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)

        sns.violinplot(
            data=melted, x='Node', y='Value', ax=ax, 
            inner="quart", palette=palette, hue='Node', legend=False
        )
        
        ax.set_title(f"Ctx {c}{title_suffix}", color=title_color, fontweight='bold', fontsize=12)
        ax.set_ylim(-6, 6)
        
        if c % n_cols != 0:
            ax.set_ylabel("")
        ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])    
    save_path = f"gen_data_plots/viz_{example}_{n_nodes}n_{lt}_{func_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    


def compute_shd(g_true, g_pred):
    shd = 0
    nodes = set(g_true.nodes()) | set(g_pred.nodes())
    nodes = sorted(list(nodes))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            true_edge = 0 
            if g_true.has_edge(u, v): true_edge = 1 
            elif g_true.has_edge(v, u): true_edge = 2 
            pred_edge = 0
            if g_pred.has_edge(u, v): pred_edge = 1
            elif g_pred.has_edge(v, u): pred_edge = 2
            if true_edge != pred_edge:
                shd += 1            
    print('The SHD is ', shd)




def get_oracle_partitions(dag, n, lt, expected_set, all_contexts):
    partitions = {f'X{i}': [all_contexts] for i in range(n)}
    
    for i in range(n):
        if i in dag.maps_nodes_star:
            arr = dag.maps_nodes_star[i]
            indices = [idx for idx, val in enumerate(arr) if val == 1]
            if indices:
                s_str = sorted([str(x) for x in indices])
                o_str = sorted([x for x in all_contexts if x not in s_str])
                partitions[f'X{i}'] = [s_str, o_str]

    shift_indices = []
    if lt == 'collider' and dag.maps_colliders:
        arr = list(dag.maps_colliders.values())[0]
        shift_indices = [idx for idx, val in enumerate(arr) if val == 1]
    elif lt == 'confounder' and dag.maps_confounders:
        arr = list(dag.maps_confounders.values())[0]
        shift_indices = [idx for idx, val in enumerate(arr) if val == 1]

    if shift_indices:
        s_str = sorted([str(x) for x in shift_indices])
        o_str = sorted([x for x in all_contexts if x not in s_str])
        latent_partition = [s_str, o_str]
        
        for group in expected_set:
            for node_idx in group:
                node_name = f'X{node_idx}'
                if node_name in partitions:
                    partitions[node_name] = latent_partition

    return {k: {"marginal_shifts": v, "mechanism_shifts": v} for k, v in partitions.items()}



def perturb_graph(adj, n_flips=1, seed=None):
    """
    Perturb a DAG adjacency matrix by randomly reversing k edges.
    Guarantees the result is still a DAG (skips flips that create cycles).
    
    adj     : (n, n) numpy array, true DAG adjacency
    n_flips : number of edge reversals to attempt
    seed    : for reproducibility
    """
    import random
    rng = random.Random(seed)
    
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    edges = list(G.edges())
    
    if not edges:
        return adj
    
    flipped = 0
    attempted = set()
    
    while flipped < n_flips and len(attempted) < len(edges):
        edge = rng.choice(edges)
        if edge in attempted:
            continue
        attempted.add(edge)
        
        u, v = edge
        G.remove_edge(u, v)
        G.add_edge(v, u)
        
        # Only keep the flip if it doesn't create a cycle
        if nx.is_directed_acyclic_graph(G):
            flipped += 1
        else:
            # Revert
            G.remove_edge(v, u)
            G.add_edge(u, v)
    
    if flipped < n_flips:
        print(f"  [perturbed] Warning: only {flipped}/{n_flips} flips possible without cycles")
    
    return nx.to_numpy_array(G, dtype=int)




if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    str_d = f"plots_{timestamp}"
    
    for d in ["logs", "results", str_d, "datasets"]:
        if not os.path.exists(d): os.makedirs(d)
    sys.stdout = LoggerWriter(os.path.join("logs", f"run_{timestamp}.txt"))
    
    csv_master = f"results/synthetic_results.csv"
    example = 0
    n_samples = 500 
    
    run_full = False
    save_data = True
    
    
    n_nodes_list = [3, 5, 7, 9, 11, 13, 15]
    seeds = [50155, 83413, 85688, 28498, 94345, 75815, 44054, 62408, 72071, 61069]
    lat_combs = [(1,0), (0,1), (0,0)]      
    funcs = ["tanh", "linear", "x^2", "x^3", "sinc"]
    oracles = ['partitions', 'full', 'perturbed', 'TOPIC']

    experimental_grid = []
    for seed in seeds:
        for n in n_nodes_list:
            for func in funcs:
                for lc in lat_combs:
                    experimental_grid.append({
                        'seed': seed,
                        'lc': lc,
                        'func': func,
                        'n': n
                    })
    allocation = {
    3: 25, 5: 25, 7: 25, 9: 25, 11: 25, 
    13: 25, 15: 25
}
    random.seed(6)

    by_n = defaultdict(list)
    for cfg in experimental_grid:
        by_n[cfg['n']].append(cfg)

    trimmed_grid = []
    for n, k in allocation.items():
        trimmed_grid.extend(random.sample(by_n[n], k))

    print(Counter(cfg['n'] for cfg in trimmed_grid))

    for scenario in trimmed_grid:
        seed = scenario['seed']
        lc = scenario['lc']
        func = scenario['func']
        n = scenario['n']
        lt = 'confounder' if lc[0] > 0 else ('collider' if lc[1] > 0 else 'none')
        for method in oracles:
            start_time = time.time()
            if n == 3:
                nc = int(n*3)
            else:
                nc = int(n*2)  
            print(f"\n\nEXAMPLE {example} | Seed: {seed} | Nodes: {n} | Latent Comb: {lc} | Function: {func} | Latent Type: {lt}")
            dag = DAGConfoundedWithSelection(seed, nc, n, lc[0], lc[1], func, lt)
            X, adj, adj_f = dag.gen_data(seed, 5000)
                                
            if 'Context' in X.columns:
                X['Context'] = X['Context'].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x))
                
            scenario_id = f"n{n}_seed{seed}_func{func}_lc{lc[0]}{lc[1]}"
            dataset_dir = f"datasets/{scenario_id}"
            os.makedirs(dataset_dir, exist_ok=True)

            # Save data
            if save_data:
                X.to_csv(f"{dataset_dir}/X.csv", index=False)

                # Save adjacency matrices
                np.save(f"{dataset_dir}/adj.npy", adj)
                np.save(f"{dataset_dir}/adj_full.npy", adj_f)

                # Save metadata
                meta = {
                    'seed': seed, 'n': n, 'func': func,
                    'lc': lc, 'lt': lt, 'nc': nc
                }
                pd.Series(meta).to_json(f"{dataset_dir}/meta.json")
                scenario_id = f"n{n}_seed{seed}_func{func}_lc{lc[0]}{lc[1]}"
                dataset_dir = f"datasets/{scenario_id}"

                X   = pd.read_csv(f"{dataset_dir}/X.csv")
                adj = np.load(f"{dataset_dir}/adj.npy")
                adj_f = np.load(f"{dataset_dir}/adj_full.npy")
                meta = pd.read_json(f"{dataset_dir}/meta.json", typ='series').to_dict()
                
            if run_full:
                m = {i: f'X{i}' for i in range(n)}
                G_o = nx.from_numpy_array(adj, create_using=nx.DiGraph)
                if lt == 'confounder':
                    expected_set = [sorted([int(x) for x in sublist]) for sublist in dag.nodes_confounded]
                
                if lt == 'none':
                    expected_set = []
                    
                if lt == 'collider':
                    all_selection_parents = set()
                    for parents in dag.nodes_selection_parents:
                        all_selection_parents.update(int(x) for x in parents)
                    
                    if all_selection_parents:
                        downstream_nodes = []
                        for node in all_selection_parents:
                            descendants = nx.descendants(G_o, node)
                            if not (descendants & all_selection_parents):
                                downstream_nodes.append(node)
                        
                        all_ancestors = set()
                        for node in downstream_nodes:
                            all_ancestors.update(nx.ancestors(G_o, node))
                            all_ancestors.add(node) 
                            
                        expected_set = [sorted(list(all_ancestors))]
                    else:
                        expected_set = []
                G_o = nx.relabel_nodes(G_o, m)
                mf = {**m}
                for i in range(lc[0]): mf[n+i] = f'Z{i}'
                for i in range(lc[1]): mf[n+lc[0]+i] = f'S{i}'
                G_t = nx.relabel_nodes(nx.from_numpy_array(adj_f, create_using=nx.DiGraph), mf)
                start_confounders = n
                start_colliders = n + lc[0]
                node_names = list(mf.values())
                print_graph_in_terminal(adj_f, node_names, start_confounders, start_colliders)

                G_t_eff = nx.DiGraph()
                obs_list = [f'X{i}' for i in range(n)]
                G_t_eff.add_nodes_from(obs_list)
                
                if lt == 'collider':
                    s_eff = "S0"; G_t_eff.add_node(s_eff)
                    all_s_parents = set()
                    for s in [node for node in G_t.nodes() if str(node).startswith('S')]:
                        all_s_parents.update(G_t.predecessors(s))
                    for p in sorted(list(all_s_parents), reverse=True)[:2]: 
                        G_t_eff.add_edge(p, s_eff)
                elif lt == 'confounder':
                    z_node = "Z0"; G_t_eff.add_node(z_node)
                    for child in G_t.successors(z_node): 
                        G_t_eff.add_edge(z_node, child)
                

                if method == 'partitions':
                    all_contexts = sorted(X['Context'].unique().astype(str).tolist())
                    print(n, lt, expected_set, all_contexts)    
                    partitions = get_oracle_partitions(dag, n, lt, expected_set, all_contexts)
                    
                    
                elif method == 'perturbed':
                    flip_fraction = 0.25
                    n_true_edges = int(adj.sum())
                    n_flips = max(1, round(n_true_edges * flip_fraction))
                    
                    print(f"  [perturbed] True edges: {n_true_edges}, flipping {n_flips}")
                    
                    perturbed_adj = perturb_graph(adj, n_flips=n_flips, seed=seed)
                    G_o = nx.relabel_nodes(
                        nx.from_numpy_array(perturbed_adj, create_using=nx.DiGraph), m
                    )
                    partitions = None
                
                elif method == 'TOPIC':
                    try:
                        from topic.topic import Topic
                        from topic.scoring.fitting import DataType, ScoreType
                    except ImportError as e:
                        print(f"  [TOPIC] Import failed: {e}. Skipping.")
                        partitions = None
                        continue

                    if 'Context' in X.columns:
                        X_features = X.drop(columns=['Context']).values.astype(float)
                        context_labels = X['Context'].values
                        unique_contexts = np.unique(context_labels)

                        X_dict = {
                            i: X_features[context_labels == ctx]
                            for i, ctx in enumerate(unique_contexts)
                        }
                        data_type = DataType.CONT_MCONTEXT
                    else:
                        X_dict = X.values.astype(float)
                        data_type = DataType.CONTINUOUS

                    topic_model = Topic(
                        data_type=data_type,
                        score_type=ScoreType.GAM,
                        extra_refinement=True,
                        vb=0
                    )

                    topic_graph, top_order = topic_model.fit(X_dict)
                    print(f"  [TOPIC] Topological order: {top_order}")

                    learned_adj = nx.to_numpy_array(topic_graph, nodelist=list(range(n)), dtype=int)
                    G_o = nx.relabel_nodes(
                        nx.from_numpy_array(learned_adj, create_using=nx.DiGraph), m
                    )

                    G_true_obs = nx.relabel_nodes(
                        nx.from_numpy_array(adj, create_using=nx.DiGraph), m
                    )
                    true_edges = set(G_true_obs.edges())
                    pred_edges = set(G_o.edges())
                    reversals = {(v, u) for (u, v) in (true_edges - pred_edges)} & (pred_edges - true_edges)
                    shd = (len(true_edges - pred_edges - {(v, u) for (u, v) in reversals})
                        + len(pred_edges - true_edges - reversals)
                        + len(reversals))
                    print(f"  [TOPIC] SHD: {shd} | True edges: {len(true_edges)} | Pred edges: {len(pred_edges)}")

                    partitions = None
                                
                else:
                    partitions = None

                parts, Gr, k_labs, marg_labs, ab_l, M_graph = run_multivariate_kci(
                X, G_o, samples=n_samples, dataset_partitions=partitions)
                print()
                print("MECHANISMS")
                current_mechanisms = {}
                for node, shifts in parts.items():
                    mech_parts = shifts['mechanism_shifts']
                    current_mechanisms[node] = [sorted(list(group)) for group in mech_parts]
                    total = len(mech_parts)
                    merged = sum(1 for p in mech_parts if len(p) > 1)
                    print(f"\nNODE {node: <4} | {total} Groups ({merged} Merged)")
                    print("-" * 30)
                    for i, group in enumerate(mech_parts):
                        marker = "[*]" if len(group) > 1 else "[ ]"
                        print(f"  {marker} Group {i+1: >2}: {sorted(group)}")
                print()

                current_preds = [r['type'].lower() for r in ab_l]
                diag_clusters = [r['nodes'] for r in ab_l]
                synch_mechs = [r['sync_mech_ami'] for r in ab_l]
                synch_margs = [r['sync_marg_ami'] for r in ab_l]
                spectral_subsets = [r['spectral_subsets'] for r in ab_l]
                coverage = [r['coverage'] for r in ab_l]
                
                end_time = time.time()
                lapsed_time = end_time - start_time
                if ab_l:
                    report_row = {
                        'oracle':method, 'n_nodes': n, 'seed': seed, 'func': func, 'nc': nc, 'gt_type': lt,
                        'predicted_gt_type': current_preds[0] if current_preds else "none",
                        'subset_nodes': str(diag_clusters),
                        'sync_mech_ami': synch_mechs,
                        'sync_marg_ami': synch_margs,
                        'mechanism_groups': str(current_mechanisms),
                        'expected_set': str(expected_set) if lt in ['confounder', 'collider'] else "none",
                        'spectral_subsets': spectral_subsets,
                        'coverage': coverage,
                        'runtime': lapsed_time
                    }
                else:
                    report_row = {'oracle':method, 'n_nodes': n, 'seed': seed, 'func': func, 'nc': nc, 'gt_type': lt,
                        'predicted_gt_type': "none",
                        'subset_nodes': "",
                        'sync_mech_ami': "",
                        'sync_marg_ami': "",
                        'mechanism_groups': "",
                        'expected_set': "",
                        'spectral_subsets': "",
                        'coverage': "",
                        'runtime': lapsed_time}
                
                df_row = pd.DataFrame([report_row])
                df_row.to_csv(csv_master, mode='a', header=not os.path.exists(csv_master), index=False)

                p_name = f"{str_d}/{example}_plot.png"
                pm_name = f"{str_d}/{example}_AMI_Graph.png"
                draw_causal_comparison(G_t, Gr, plot_filename=p_name, seed=seed)
                visualize_mdl_graph(M_graph, plot_filename=pm_name)
                
                example += 1
                print('\n\n')
        print(f"\nExperiment Complete. Results: {csv_master}")