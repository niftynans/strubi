import os
import json
import pandas as pd
import numpy as np
import time
from itertools import chain
from utils.CCA_tools import Chi2RankTest
from utils.FCI_tools import fci

def run_experiment_suite(root_path, output_file="experiment_results.csv"):
    results = []
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    for folder in sorted(folders):
        start_time = time.time()
        f_path = os.path.join(root_path, folder)
        
        # 1. Load Metadata
        with open(os.path.join(f_path, "meta.json"), 'r') as f:
            meta = json.load(f)
            
        # 2. Load Data
        df = pd.read_csv(os.path.join(f_path, "X.csv"))
        cols = [c for c in df.columns if c.startswith('X')]
        X_data = df[cols].values
        X_data = (X_data - np.mean(X_data, axis=0)) / (np.std(X_data, axis=0) + 1e-9)
        
        # 3. Load Prior Adjacency
        prior_adj = np.load(os.path.join(f_path, "adj.npy"))
        
        # 4. Define Rank Test Logic
        rank_tester = Chi2RankTest(X_data, N_scaling=1)
        
        def ci_tester(i, j, Z=None):
            if Z is None: Z = []
            # Respect prior observed graph
            if prior_adj[i, j] != 0 or prior_adj[j, i] != 0:
                return False
            
            # Rank Test Partitioning
            Z_meas = [[z] for z in Z]
            Z1 = list(chain.from_iterable([xz[:len(xz)//2 + 1] for xz in Z_meas]))
            Z2 = list(chain.from_iterable([xz[len(xz)//2 + 1:] for xz in Z_meas]))
            
            val, _, _, _ = rank_tester.test([i] + Z1, [j] + Z2, len(Z), 0.05)
            return bool(val)

        # 5. Run Discovery
        node_ids = list(range(len(cols)))
        pag_edges = fci(node_ids, ci_tester)
        
        # 6. Categorize Resulting Structure
        # Check for undirected edges (signature of selection bias/collider conditioning)
        has_undirected = any("-" in str(edge) and "<" not in str(edge) for edge in pag_edges)
        predicted_type = "collider" if has_undirected else "confounder/none"
        
        runtime = time.time() - start_time
        
        # 7. Append Row
        results.append({
            "n_nodes": meta.get("nc", len(cols)),
            "func": meta.get("func", "unknown"),
            "gt_type": meta.get("lt", "unknown"),
            "predicted_gt_type": predicted_type,
            "subset_nodes": [cols],
            "expected_set": [list(range(len(cols)))],
            "runtime": round(runtime, 6)
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results_df

if __name__ == "__main__":
    # Update this path to your data root
    run_experiment_suite("datasets")