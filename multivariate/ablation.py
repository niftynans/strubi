import os
import time
import json
import random
import datetime
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

from info import run_multivariate_kci, draw_causal_comparison, visualize_mdl_graph
from mv_dgp_v4 import DAGConfoundedWithSelection
from synthetic_workflow import get_oracle_partitions, perturb_graph
import sys

SEEDS = [50155, 83413, 85688, 28498, 94345, 75815, 44054, 62408, 72071, 61069]
LAT_COMBS = [(1,0), (0,1), (0,0)] 
FUNCS = ["tanh", "linear", "x^2", "x^3", "sinc"]

def get_expected_set(dag, lt, G_o):
    """Calculates ground truth nodes for confounders and colliders."""
    if lt == 'confounder':
        return [sorted([int(x) for x in sublist]) for sublist in dag.nodes_confounded]
    elif lt == 'collider':
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
            return [sorted(list(all_ancestors))]
    return []

def run_ablation_matrix(specs):
    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for axis, spec in specs.items():
        print(f"\n{'='*40}\nAXIS: {axis.upper()}\n{'='*40}")
        # csv_out = f"results/ablation_{axis}_{timestamp}.csv"
        csv_out = f"multivariate/results/ablation_{axis}.csv"
        print(csv_out)

        situation_grid = list(itertools.product(
            spec["n_nodes_list"], SEEDS, FUNCS, LAT_COMBS,
            spec["random_pool"] if isinstance(spec["random_pool"], list) else [spec["random_pool"]]
        ))

        random.seed(6)
        by_n = defaultdict(list)
        for sit in situation_grid: by_n[sit[0]].append(sit)
        sampled_situations = []
        for n_val in spec["n_nodes_list"]:
            num_to_sample = min(len(by_n[n_val]), 25)
            sampled_situations.extend(random.sample(by_n[n_val], num_to_sample))
        
        if axis in ["samples", "affected_nodes", "main"]:        
            sampled_situations = sampled_situations[:len(sampled_situations)-25]
        
        print(sampled_situations)
        for n, seed, func, lc, is_rand in sampled_situations:
            lt = 'confounder' if lc[0] > 0 else ('collider' if lc[1] > 0 else 'none')
            
            sweep_key = {
                "samples": "n_samples", "shifts": "r", "sparsity": "dense", 
                "contexts": "nc_factor", "perturbed_graph": "flip_fractions"
            }.get(axis, "n_samples") if axis != "affected_nodes" else "n_samples"
            
            sweep_values = spec.get(sweep_key, [None])

            for val in sweep_values:
                start_time = time.time()
                
                samp = val if axis == "samples" else spec["n_samples"][0]
                r_val = val if axis == "shifts" else spec["r"][0]
                nc = int(n * (val if axis == "contexts" else spec["nc_factor"][0]))
                dense_val = val if axis == "sparsity" else spec["dense"][0]
                k_val = spec.get("fixed_k", [None])[0]

                dag = DAGConfoundedWithSelection(
                    seed, nc, n, lc[0], lc[1], func=func, 
                    latent_type=lt,
                    fixed_latent_size=k_val, obs_shift_ratio=r_val, dense_val=dense_val 
                )
                X, adj, adj_f = dag.gen_data(seed, samp)
                save_data = True
                if save_data:
                    scenario_id = f"n{n}_seed{seed}_func{func}_lc{lc[0]}{lc[1]}"
                    dataset_dir = f"datasets/{scenario_id}"
                    os.makedirs(dataset_dir, exist_ok=True)
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

                    
                run_ahead = True
                if run_ahead:
                    print("here")
                    for oracle in spec["oracles"]:
                        G_o = nx.from_numpy_array(adj, create_using=nx.DiGraph)
                        expected_set = get_expected_set(dag, lt, G_o)
                        partitions = None
                        
                        if oracle == 'partitions':
                            partitions = get_oracle_partitions(dag, n, lt, expected_set, X['Context'].unique().astype(str))
                        elif oracle == 'perturbed':
                            ff = val if axis == "perturbed_graph" else spec.get("flip_fractions", [0.25])[0]
                            n_flips = max(1, round(int(adj.sum()) * ff))
                            G_o = nx.from_numpy_array(perturb_graph(adj, n_flips, seed), create_using=nx.DiGraph)
                        elif oracle == 'TOPIC':
                            try:
                                from topic.topic import Topic
                                X_dict = {i: X[X['Context']==ctx].drop(columns=['Context']).values for i, ctx in enumerate(X['Context'].unique())}
                                topic_model = Topic(data_type=3, score_type=1, vb=0) 
                                G_o = nx.DiGraph(topic_model.fit(X_dict)[0])
                            except Exception: print('hello')

                        try:
                            G_o = nx.relabel_nodes(G_o, {i: f'X{i}' for i in range(n)})
                            parts, Gr, _, _, ab_l, M_graph = run_multivariate_kci(X, G_o, samples=samp, dataset_partitions=partitions)
                            
                            pred_type = ab_l[0]['type'].lower() if ab_l else "none"
                            subset_nodes = ab_l[0]['nodes'] if ab_l else []
                            coverage = ab_l[0]['coverage'] if ab_l else 0.0
                            ami = ab_l[0]['sync_mech_ami'] if ab_l else 0.0
                        except Exception as e:
                            print(f"Error in {axis} | {oracle}: {e}")
                            continue

                        report = {
                            'axis': axis, 'axis_val': val, 'n_nodes': n, 'seed': seed, 'func': func, 
                            'oracle': oracle, 'gt_type': lt, 'predicted_gt_type': pred_type, 
                            'subset_nodes': str(subset_nodes), 'expected_set': str(expected_set),
                            'sync_mech_ami': ami, 'coverage': coverage, 'runtime': time.time() - start_time
                        }
                        pd.DataFrame([report]).to_csv(csv_out, mode='a', header=not os.path.exists(csv_out), index=False)
                        print(f"[{axis.upper()}] n={n} | val={val} | Oracle={oracle} | PRED={pred_type}")

ABLATION_SPECS = {
    "samples": {
        "n_samples": [100, 200, 500, 1000], "n_nodes_list": [3, 5, 7, 9, 11, 13, 15],
        "oracles": ["full"], "r": [0.25], "nc_factor": [2.0], "random_pool": [True], "dense": [0.48]
    },
    "shifts": {
        "n_samples": [500], "n_nodes_list": [5], "oracles": ["full"], 
        "r": [0, 0.25, 0.5, 1.0, 1.5], "nc_factor": [2.0], "random_pool": [True], "dense": [0.48]
    },
    "sparsity": {
        "n_samples": [500], "n_nodes_list": [5], "oracles": ["full"], 
        "r": [0.25], "nc_factor": [2.0], "random_pool": [True], "dense": [0.2, 0.48, 0.6, 0.8] 
    },
    "contexts": {
        "n_samples": [500], "n_nodes_list": [5], "oracles": ["full"], 
        "r": [0.25], "nc_factor": [0.5, 1.0, 2.0, 3.0], "random_pool": [True], "dense": [0.48]
    },
    "affected_nodes": {
        "n_samples": [500], "n_nodes_list": [3, 5, 7, 9, 11, 13, 15], "oracles": ["full"], 
        "r": [0.25], "nc_factor": [2.0], "random_pool": [False], "dense": [0.48], "fixed_k": [2]
    },
    "perturbed_graph": {
        "n_samples": [500], "n_nodes_list": [5], "oracles": ["perturbed"], 
        "flip_fractions": [0.1, 0.25, 0.4, 0.5], "r": [0.25], "nc_factor": [2.0], "random_pool": [True], "dense": [0.48]
    },
    "main": {
        "n_samples": [500], "n_nodes_list": [3, 5, 7, 9, 11, 13, 15], 
        "oracles": ["partitions", "full", "perturbed", "TOPIC"],
        "r": [0.25], "nc_factor": [2.0], "random_pool": [True], "dense": [0.48], "flip_fractions": [0.25]
    }
}

if __name__ == "__main__":
    os.makedirs("multivariate/results", exist_ok=True)
    run_ablation_matrix(ABLATION_SPECS)