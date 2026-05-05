import os, sys
import re

from math import log2, lgamma
import itertools
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from scipy.stats import ks_2samp
from causallearn.utils.KCI.KCI import KCI_CInd
from scipy import stats


from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigvalsh
from sklearn.metrics import adjusted_mutual_info_score



def get_spectral_subsets(M_graph, strength_threshold=0.05):
    nodes = list(M_graph.nodes())
    if not nodes:
        return []

    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for u, v, d in M_graph.edges(data=True):
        weight = d['weight']
        if weight > strength_threshold:
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = adj_matrix[j, i] = weight

    if np.sum(adj_matrix) == 0:
        return []

    G_temp = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                G_temp.add_edge(nodes[i], nodes[j])
    
    initial_subsets = [list(c) for c in nx.connected_components(G_temp) if len(c) > 1]
    n_clusters = len(initial_subsets)
    
    if n_clusters == 0:
        return []

    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        assign_labels='discretize',
        random_state=42
    )
    labels = sc.fit_predict(adj_matrix)

    spectral_subsets = []
    for cluster_id in range(n_clusters):
        subset = [nodes[i] for i, l in enumerate(labels) if l == cluster_id]
        if len(subset) > 1:
            spectral_subsets.append(subset)
            
    return spectral_subsets
    
    
    
def visualize_mdl_graph(M_graph, plot_filename):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(M_graph, weight='weight', k=1.5, iterations=50)
    
    edges = M_graph.edges(data=True)
    weights = np.array([d['weight'] for u, v, d in edges])
    
    try:
        v_min, v_max = weights.min(), weights.max()
    except:
        v_min, v_max = 0.0 , 0.0
    norm_weights = (weights - v_min) / (v_max - v_min + 1e-9)

    nx.draw_networkx_nodes(M_graph, pos, node_size=800, node_color='#a1c9f4')
    nx.draw_networkx_labels(M_graph, pos, font_size=12, font_weight='bold')

    nx.draw_networkx_edges(
        M_graph, pos, 
        width=norm_weights * 6, 
        edge_color=weights, 
        edge_cmap=plt.cm.Blues,
        alpha=0.7
    )

    edge_labels = { (u, v): f"{d['weight']:.2f}" for u, v, d in edges }
    nx.draw_networkx_edge_labels(M_graph, pos, edge_labels=edge_labels, font_size=9)

    plt.title("MDL Gain Network: Identifying Latent Cliques", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(plot_filename)
    # print(f"Plot saved to {plot_filename}")
    plt.close()
    



def multinomial_mdl(counts, n_total):
    if n_total == 0: return 0
    data_term = 0
    for ni in counts:
        if ni > 0:
            data_term -= ni * log2(ni / n_total)
    k = len(counts)
    if k <= 1: return data_term
    complexity_term = ((k - 1) / 2) * log2(n_total / (2 * np.pi))
    return data_term + complexity_term










def get_mdl_gain(labels1, labels2, total_contexts):
    _, counts1 = np.unique(labels1, return_counts=True)
    _, counts2 = np.unique(labels2, return_counts=True)
    joint_labels = [f"{a}_{b}" for a, b in zip(labels1, labels2)]
    _, counts_joint = np.unique(joint_labels, return_counts=True)
    l1 = multinomial_mdl(counts1, total_contexts)
    l2 = multinomial_mdl(counts2, total_contexts)
    l_joint = multinomial_mdl(counts_joint, total_contexts)
    return max(0, (l1 + l2) - l_joint)













def get_shifted_set_old(G_part, unique_contexts):
    components = [sorted(list(c), key=lambda x: int(x)) for c in nx.connected_components(G_part)]    
    nodes_in_components = set().union(*components) if components else set()
    for c in unique_contexts:
        if c not in nodes_in_components:
            components.append([c])
    return sorted(components, key=lambda x: int(x[0]))



def get_shifted_set(G_conflict, unique_contexts):
    for c in unique_contexts:
        if c not in G_conflict.nodes():
            G_conflict.add_node(c)
    coloring = nx.coloring.greedy_color(G_conflict, strategy='largest_first')
    groups = {}
    for node, color in coloring.items():
        groups.setdefault(color, []).append(node)
    
    result = [sorted(g, key=lambda x: int(x)) for g in groups.values()]
    return sorted(result, key=lambda x: int(x[0]))




def get_partitions_by_count(counts, unique_contexts, real_world=False):
    if not counts:
        return []

    vals = list(counts.values())
    max_c = max(vals)
    min_c = min(vals)
    noise_floor = max(5, int(len(unique_contexts) * 0.15))
    # if real_world:
    #     noise_floor = 0.1
    if max_c < noise_floor:
        return [sorted([str(c) for c in unique_contexts], key=int)]
    
    threshold = max(noise_floor, (max_c + min_c) / 2)
    
    baseline_group = []
    shifted_group = []
    
    for ctx in unique_contexts:
        c_str = str(ctx)
        if counts.get(c_str, 0) >= threshold:
            shifted_group.append(c_str)
        else:
            baseline_group.append(c_str)
            
    result = []
    if baseline_group:
        result.append(sorted(baseline_group, key=int))
    if shifted_group:
        result.append(sorted(shifted_group, key=int))
        
    return result



def test_pair(c, c_prime, df, col_name, parent_cols, alpha, kci_test, samples=None):
    df_c = df[df['Context'] == c]
    df_c_prime = df[df['Context'] == c_prime]
    
    d_c_flat = df_c[col_name].values
    d_cp_flat = df_c_prime[col_name].values
    _, p_val_marg = ks_2samp(d_c_flat, d_cp_flat)
    is_shifted_marg = p_val_marg < alpha
    
    MAX_SAMPLES = 2000 if samples is None else samples
    if len(df_c) + len(df_c_prime) > MAX_SAMPLES:
        n_sub = MAX_SAMPLES // 2
        if len(df_c) > n_sub: df_c = df_c.sample(n=n_sub)
        if len(df_c_prime) > n_sub: df_c_prime = df_c_prime.sample(n=n_sub)
    
    data_node = np.concatenate([df_c[col_name].values, df_c_prime[col_name].values]).reshape(-1, 1)
    context_label = np.concatenate([np.zeros(len(df_c)), np.ones(len(df_c_prime))]).reshape(-1, 1)
    
    if not parent_cols:
        p_val_mech = p_val_marg
    else:
        data_parents = np.concatenate([df_c[parent_cols].values, df_c_prime[parent_cols].values])
        if data_parents.ndim == 1: 
            data_parents = data_parents.reshape(-1, 1)
            
        p_val_mech, _ = kci_test.compute_pvalue(data_node, context_label, data_parents)

    is_shifted_mech = p_val_mech < alpha

    status = "SHIFT DETECTED" if is_shifted_mech else "INVARIANT"
    # if is_shifted_mech:
    # print(f"Node {col_name} | {c} vs {c_prime} | Mech p-val: {p_val_mech:.6f} | {status}")
    return (c, c_prime, is_shifted_marg, is_shifted_mech)




def run_multivariate_kci(df, G_known, alpha=0.0001, kci_test=KCI_CInd(), samples=None, real_world=False, dataset_partitions=None):
    unique_contexts = sorted(df['Context'].unique())
    sorted_nodes = sorted(list(G_known.nodes()))
    total_contexts = len(unique_contexts)
    if not dataset_partitions:
        dataset_partitions = {}
        
        for node_idx in tqdm(sorted_nodes, desc="Nodes"):
            col_name = str(node_idx)
            p_cols = list(G_known.predecessors(node_idx))
            
            results = Parallel(n_jobs=-1)(delayed(test_pair)(
                c, cp, df, col_name, p_cols, alpha, kci_test, samples
            ) for c, cp in itertools.combinations(unique_contexts, 2))

            mech_counts = {str(c): 0 for c in unique_contexts}
            marg_counts = {str(c): 0 for c in unique_contexts}

            for c, cp, is_shifted_marg, is_shifted_mech in results:
                if is_shifted_marg:
                    marg_counts[str(c)] += 1
                    marg_counts[str(cp)] += 1
                if is_shifted_mech:
                    mech_counts[str(c)] += 1
                    mech_counts[str(cp)] += 1
            
            dataset_partitions[col_name] = { 
                "marginal_shifts": get_partitions_by_count(marg_counts, unique_contexts, real_world),
                "mechanism_shifts": get_partitions_by_count(mech_counts, unique_contexts, real_world)
            }
    
    node_labels, marg_labels = {}, {}
    for node in sorted_nodes:
        node_str = str(node)
        m_labs = np.zeros(total_contexts, dtype=int)
        # if real_world:
            
        #     context_ids = [int(re.search(r'\d+', c).group()) for c in X_kci['Context'].unique()]
        #     max_id = max(context_ids)
        #     m_labs = np.zeros(max_id + 1)
        for idx, comp in enumerate(dataset_partitions[node_str]['mechanism_shifts']):
            print(idx, comp)
            for ctx in comp:
                m_labs[int(re.search(r'\d+', ctx).group())] = idx + 1
        node_labels[node_str] = m_labs
        
        ma_labs = np.zeros(total_contexts, dtype=int)
        for idx, comp in enumerate(dataset_partitions[node_str]['marginal_shifts']):
            for ctx in comp:
                ma_labs[int(re.search(r'\d+', ctx).group())] = idx + 1
        marg_labels[node_str] = ma_labs

    nodes_list = sorted(list(node_labels.keys()))
    M_graph = nx.Graph()
    if real_world:
        strength_threshold = 0.25
    else:
        strength_threshold = 0.09
    
    node_signatures = {}
    for node in nodes_list:
        node_str = str(node)
        shifts = dataset_partitions[node_str]['mechanism_shifts']
        if real_world:
            if len(shifts) > 1:
                node_signatures[node_str] = set(min(shifts, key=len))
            else:
                node_signatures[node_str] = set()
            
        else:
                    
            if len(shifts) > 1:
                node_signatures[node_str] = set(shifts[1])
            else:
                node_signatures[node_str] = set()

    M_graph = nx.Graph()
    unique_contexts = [int(x) for x in unique_contexts]
    int_signatures = {node: {int(ctx) for ctx in sig} for node, sig in node_signatures.items()}
    
    for u, v in itertools.combinations(nodes_list, 2):
        print(u, v)
        sig_u = int_signatures[u]
        sig_v = int_signatures[v]
        
        if sig_u and sig_v:
            labels_u = [1 if ctx in sig_u else 0 for ctx in unique_contexts]
            labels_v = [1 if ctx in sig_v else 0 for ctx in unique_contexts]
            
            gain = adjusted_mutual_info_score(labels_u, labels_v)
            gain = max(0, gain)
            print(gain)
                
        else:
            gain = 0.0
        
        if gain > strength_threshold:
            M_graph.add_edge(u, v, weight=gain)

    all_weights = [d['weight'] for _, _, d in M_graph.edges(data=True)]

    if not all_weights:
        subsets = []
    else:
        G_strong = nx.Graph()
        for u, v, d in M_graph.edges(data=True):
            print(u, v, d['weight'])
            if d['weight'] > strength_threshold:
                G_strong.add_edge(u, v)
                
        # plt.figure(figsize=(10, 8))    
        # pos = nx.spring_layout(G_strong, k=0.5, seed=42)
        # nx.draw_networkx_nodes(G_strong, pos, node_size=700, node_color='skyblue', edgecolors='white')
        # nx.draw_networkx_labels(G_strong, pos, font_size=10, font_family='sans-serif', font_weight='bold')
        # nx.draw_networkx_edges(G_strong, pos, width=2, edge_color='gray', alpha=0.6)
        # plt.title("Strong Mechanism-Shift Connections", fontsize=15)
        # plt.axis('off')
        # plt.show()
        
        # if real_world:
        #     subsets = [list(c) for c in nx.find_cliques(G_strong) if len(c) > 1]
        # else:
        subsets = [list(c) for c in nx.connected_components(G_strong) if len(c) > 1]
    spectral_subsets = get_spectral_subsets(M_graph, strength_threshold=strength_threshold)
    G_refined = G_known.copy()
    ablation_results = []
    print("Subsets : ", subsets)

    for idx, X_S in enumerate(subsets):
        X_S_list = sorted(list(X_S))
        
        internal_sync = np.mean([adjusted_mutual_info_score(node_labels[a], node_labels[b]) 
                                for a, b in itertools.combinations(X_S_list, 2)])
        marg_sync = np.mean([adjusted_mutual_info_score(marg_labels[a], marg_labels[b]) 
                            for a, b in itertools.combinations(X_S_list, 2)])
        print(internal_sync)
        if real_world:
            internal_thresh = 0.1 # 0.50 for light_tunnel
        else:
            internal_thresh = 0.1
        
        if internal_sync < internal_thresh:
            continue
        
        node_a = None
        max_ancestors = -1
        print("_"*80)
        print("Subset : ", X_S_list)
        for node in X_S_list:
            print(node)
            ancestors = nx.ancestors(G_known, node)
            print(ancestors)
            overlap = len(set(ancestors).intersection(set(X_S_list)))
            print(overlap)
            if overlap > max_ancestors:
                max_ancestors = overlap
                node_a = node
        print("FINALLY")
        print(node_a)
        s_prime = nx.ancestors(G_known, node_a)
        print(s_prime)
        if len(s_prime) > 0:
            coverage = len(set(X_S_list).intersection(s_prime)) / len(s_prime)
        else:
            coverage = 0
        print("Coverage : ", coverage)
        if real_world:
            if coverage > 0.80:
                is_selection = True
                type_label = "collider"
                lid = f"S{idx}"
            else:
                is_selection = False
                type_label = "confounder"
                lid = f"Z{idx}"
        else:
            if coverage > 0.80:
                is_selection = True
                type_label = "collider"
                lid = f"S{idx}"
            else:
                is_selection = False
                type_label = "confounder"
                lid = f"Z{idx}"

        G_refined.add_node(lid, latent=True)
        for node in X_S_list:
            if is_selection:
                G_refined.add_edge(node, lid, style='dashed', color='red')
            else:
                G_refined.add_edge(lid, node, style='dashed', color='blue')
        
        ablation_results.append({
            'nodes': X_S_list, 
            'type': type_label,
            'is_selection': is_selection, 
            'sync_mech_ami': internal_sync,
            'sync_marg_ami': marg_sync,
            'spectral_subsets': str(spectral_subsets),
            'coverage': coverage
        })

    return dataset_partitions, G_refined, node_labels, marg_labels, ablation_results, M_graph



def draw_refined_graph(G):
    if not G.nodes():
        return
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    obs_nodes = [n for n in G.nodes if str(n).startswith('X')]
    lat_nodes = [n for n in G.nodes if not str(n).startswith('X')]
    nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes, node_color='skyblue', 
                        node_size=1500, label='Observed')
    nx.draw_networkx_nodes(G, pos, nodelist=lat_nodes, node_color='orange', 
                        node_size=1200, label='Discovered Latents')
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') != 'dashed']
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2, 
                        arrowsize=20, edge_color='gray')
    for u, v in dashed_edges:
        color = G[u][v].get('color', 'red')
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style='dashed', 
                            edge_color=color, width=2.5, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    plt.title("Refined Causal Structure (MDL Discovery)")
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.show()
    
    
    
    





        
def plot_ablation_results(df_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))    
    sns.boxplot(ax=axes[0], data=df_results, x='gt_type', y='saturation', hue='function')
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='0.5 Threshold')
    axes[0].axhline(0.6, color='green', linestyle='-', alpha=0.8, label='0.6 Threshold')
    axes[0].set_title("Saturation Ratio: Internal Sync / Avg Entropy")
    axes[0].set_ylabel("Saturation Ratio")
    axes[0].legend()
    sns.boxplot(ax=axes[1], data=df_results, x='gt_type', y='leakage', hue='function')
    axes[1].axhline(0.4, color='red', linestyle='--', alpha=0.6, label='0.4 Threshold')
    axes[1].set_title("Leakage Ratio: Max Anc Gain / Internal Sync")
    axes[1].set_ylabel("Leakage Ratio")
    axes[1].legend()
    plt.suptitle("Causal Latent Diagnosis Ablation Study", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()










def calculate_total_correlation(subset, node_labels, total_contexts):
    if not subset: return 0
    individual_sum = 0
    for node in subset:
        _, counts = np.unique(node_labels[node], return_counts=True)
        individual_sum += multinomial_mdl(counts, total_contexts)
    joint_strings = ["_".join(str(node_labels[node][t]) for node in subset) 
                    for t in range(total_contexts)]
    _, joint_counts = np.unique(joint_strings, return_counts=True)
    joint_mdl = multinomial_mdl(joint_counts, total_contexts)
    return individual_sum - joint_mdl





def refine_and_build_graph(subset, diagnosis, G_known, latent_idx=0):
    G_refined = G_known.copy()
    subset_list = sorted(list(subset))
    refinement_info = {'original_subset': subset_list}

    # Handle empty subsets safely
    if not subset_list:
        return G_refined, refinement_info

    # --- Case 1: Selection Bias (Collider) ---
    # Structure: Nodes -> S (Selection Node)
    if diagnosis == 'S':
        s_node = f"S{latent_idx}"
        G_refined.add_node(s_node, label='Selection')
        for node in subset_list: 
            G_refined.add_edge(node, s_node)
            
        refinement_info.update({
            'added_node': s_node,
            'direction': 'inward',
            'explanation': f"Selection bias: nodes {subset_list} point to {s_node}"
        })

    # --- Case 2: Confounding (Shared Latent) ---
    # Structure: Z -> Nodes (Latent Cause)
    else:
        z_node = f"Z{latent_idx}"
        G_refined.add_node(z_node, label='Latent')
        for node in subset_list: 
            G_refined.add_edge(z_node, node)
        
        refinement_info.update({
            'added_node': z_node, 
            'direction': 'outward',
            'explanation': f"Confounding: {z_node} points to nodes {subset_list}"
        })

    return G_refined, refinement_info



# THINK ABOUT TOTAL CORRELATION LATER -- CHALLENGING BECAUSE NOW BACKDOORS
def refine_and_build_graph_v1(subset, diagnosis, G_known, node_labels, total_contexts, latent_idx=0):
    G_refined = G_known.copy()
    subset_list = sorted(list(subset))
    refinement_info = {'original_subset': subset_list}

    tc_score = calculate_total_correlation(subset_list, node_labels, total_contexts)
    avg_ent = np.mean([multinomial_mdl(np.unique(node_labels[n], return_counts=True)[1], total_contexts) for n in subset_list])
    
    threshold_factor = 0.4
    threshold_val = (len(subset_list) - 1) * threshold_factor * avg_ent
    is_compressible = tc_score > threshold_val

    if diagnosis == 'S':
        s_node = f"S{latent_idx}"
        G_refined.add_node(s_node, label='Selection')
        for parent in subset_list: 
            G_refined.add_edge(parent, s_node)
            
        refinement_info.update({
            'nodes_added': subset_list,
            'explanation': f"Full joint selection cluster: {subset_list}"
        })
        return G_refined, refinement_info

    if is_compressible and len(subset_list) > 1:
        z_node = f"Z{latent_idx}"
        G_refined.add_node(z_node, label='Latent')
        for target in subset_list: 
            G_refined.add_edge(z_node, target)
        
        refinement_info.update({
            'nodes_added': subset_list, 
            'explanation': f"Full joint confounding cluster: {subset_list}"
        })

    # elif is_compressible and len(subset_list) == 1:
    #     z_node = f"Z{latent_idx}"
    #     G_refined.add_node(z_node, label='Latent')
    #     G_refined.add_edge(z_node, subset_list[0])
        
    # else:
    #     for sub_node in subset_list:
    #         lz = f"Z{sub_node}"
    #         G_refined.add_node(lz, label='Local Latent')
    #         G_refined.add_edge(lz, sub_node)

    return G_refined, refinement_info



def draw_causal_comparison(G_gt, G_refined, plot_filename=None, seed=42):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    all_nodes = set(G_gt.nodes()) | set(G_refined.nodes())
    union_G = nx.Graph()
    union_G.add_nodes_from(all_nodes)
    union_G.add_edges_from(G_gt.edges())
    union_G.add_edges_from(G_refined.edges())
    
    pos = nx.spring_layout(union_G, seed=seed, k=1.5)

    NODE_SIZE = 1000
    ARROW_SIZE = 35
    MARGIN = 35  

    ax1.set_title("Ground Truth (Reality)", fontsize=16, fontweight='bold')
    
    gt_obs = [n for n in G_gt.nodes() if str(n).startswith('X')]
    gt_lat = [n for n in G_gt.nodes() if str(n).startswith('Z')]
    gt_sel = [n for n in G_gt.nodes() if str(n).startswith('S')]

    nx.draw_networkx_nodes(G_gt, pos, nodelist=gt_obs, ax=ax1, node_color='lightblue', node_size=NODE_SIZE)
    nx.draw_networkx_nodes(G_gt, pos, nodelist=gt_lat, ax=ax1, node_color='lightgreen', node_size=NODE_SIZE, node_shape='s')
    nx.draw_networkx_nodes(G_gt, pos, nodelist=gt_sel, ax=ax1, node_color='salmon', node_size=NODE_SIZE, node_shape='d')    
    nx.draw_networkx_labels(G_gt, pos, ax=ax1, font_size=12, font_weight='bold')
    
    nx.draw_networkx_edges(G_gt, pos, ax=ax1, edge_color='gray', alpha=0.6, arrows=True, arrowsize=ARROW_SIZE, min_target_margin=MARGIN)

    ax2.set_title("Refined Causal Model (Discovery)", fontsize=16, fontweight='bold')
    
    obs_nodes = [n for n in G_refined.nodes() if str(n).startswith('X')]
    lat_nodes = [n for n in G_refined.nodes() if str(n).startswith('Z')]
    sel_nodes = [n for n in G_refined.nodes() if str(n).startswith('S')]
    
    nx.draw_networkx_nodes(G_refined, pos, nodelist=obs_nodes, ax=ax2, node_color='lightblue', node_size=NODE_SIZE)
    nx.draw_networkx_nodes(G_refined, pos, nodelist=lat_nodes, ax=ax2, node_color='lightgreen', node_size=NODE_SIZE, node_shape='s')
    nx.draw_networkx_nodes(G_refined, pos, nodelist=sel_nodes, ax=ax2, node_color='salmon', node_size=NODE_SIZE, node_shape='d')
    
    nx.draw_networkx_labels(G_refined, pos, ax=ax2, font_size=12, font_weight='bold')
    
    obs_edges = [(u, v) for u, v in G_refined.edges() if u in obs_nodes and v in obs_nodes]
    lat_edges = [(u, v) for u, v in G_refined.edges() if str(u).startswith('Z')]
    sel_edges = [(u, v) for u, v in G_refined.edges() if str(v).startswith('S')]
    
    nx.draw_networkx_edges(G_refined, pos, edgelist=obs_edges, ax=ax2, edge_color='gray', 
                        arrows=True, arrowsize=ARROW_SIZE, min_target_margin=MARGIN)
    
    nx.draw_networkx_edges(G_refined, pos, edgelist=lat_edges, ax=ax2, 
                        edge_color='green', style='dotted', width=2.5, 
                        arrows=True, arrowsize=ARROW_SIZE, min_target_margin=MARGIN)
    
    nx.draw_networkx_edges(G_refined, pos, edgelist=sel_edges, ax=ax2, 
                        edge_color='red', style='dashed', width=2.5, 
                        arrows=True, arrowsize=ARROW_SIZE, min_target_margin=MARGIN)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Observed (X)', markerfacecolor='lightblue', markersize=15),
        Line2D([0], [0], marker='s', color='w', label='Latent (Z)', markerfacecolor='lightgreen', markersize=15),
        Line2D([0], [0], marker='d', color='w', label='Selection (S)', markerfacecolor='salmon', markersize=15)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(plot_filename)
    # print(f"Plot saved to {plot_filename}")
    plt.close()
    
    
    
    
    
    
    
    
    
    
def calculate_metrics(G_refined, dag, n_nodes):
    disc_latents = [n for n in G_refined.nodes() if str(n).startswith('Z')]
    disc_selections = [n for n in G_refined.nodes() if str(n).startswith('S')]
    
    gt_z_neighborhoods = []
    for z_idx in range(dag.n_confounders):
        neighbors = set(f'X{i}' for i in range(n_nodes) if dag.adj_full[n_nodes + z_idx, i] != 0)
        if neighbors: gt_z_neighborhoods.append(neighbors)
        
    gt_s_neighborhoods = []
    for s_idx in range(dag.n_colliders):
        start_idx = n_nodes + dag.n_confounders + s_idx
        neighbors = set(f'X{i}' for i in range(n_nodes) if dag.adj_full[i, start_idx] != 0)
        if neighbors: gt_s_neighborhoods.append(neighbors)

    tp, fp, fn = 0, 0, 0
    matched_gt_z = set()
    for dz in disc_latents:
        dz_neighbors = set(G_refined.successors(dz))
        found_match = False
        for i, gt_neigh in enumerate(gt_z_neighborhoods):
            intersection = dz_neighbors.intersection(gt_neigh)
            if len(intersection) / len(dz_neighbors.union(gt_neigh)) > 0.5:
                tp += 1
                matched_gt_z.add(i)
                found_match = True
                break
        if not found_match:
            fp += 1 
    fn += (len(gt_z_neighborhoods) - len(matched_gt_z))

    matched_gt_s = set()
    for ds in disc_selections:
        ds_neighbors = set(G_refined.predecessors(ds))
        found_match = False
        for i, gt_neigh in enumerate(gt_s_neighborhoods):
            intersection = ds_neighbors.intersection(gt_neigh)
            if len(intersection) / len(ds_neighbors.union(gt_neigh)) > 0.3: 
                tp += 1
                matched_gt_s.add(i)
                found_match = True
                break
        if not found_match:
            fp += 1
    fn += (len(gt_s_neighborhoods) - len(matched_gt_s))
    return {"TP": tp, "TN": "N/A", "FP": fp, "FN": fn}