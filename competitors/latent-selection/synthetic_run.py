import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
from main import latent_selection_discovery
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from collections import Counter

def run_benchmark_per_context(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    results = []
    folders = [f for f in dataset_root.iterdir() if f.is_dir() and "seed" in f.name]
    
    for folder in tqdm(folders):
        try:
            with open(folder / "meta.json", 'r') as f:
                meta = json.load(f)
            
            df = pd.read_csv(folder / "X.csv")
            X_cols = [c for c in df.columns if c.startswith('X')]
            
            # --- START CONTEXT-WISE LOGIC ---
            context_nodes_found = []
            context_pred_types = []
            
            start_time = time.time()
            
            # Iterate through each unique context in the file
            unique_contexts = df['Context'].unique()
            for ctx in unique_contexts:
                ctx_data = df[df['Context'] == ctx][X_cols].to_numpy()
                
                try:
                    res = latent_selection_discovery(ctx_data, alpha=0.01)
                    
                    # Extract nodes for this specific context
                    mapping = res.get('estimated_Lid_to_Xids', {})
                    for observed_indices in mapping.values():
                        for idx in observed_indices:
                            context_nodes_found.append(f"X{idx}")
                    
                    # Extract prediction type for this specific context
                    pag_edges = res.get('estimated_L_PAG_edges', [])
                    edge_str = str(pag_edges).lower()
                    if 'undirected' in edge_str or '-' in edge_str:
                        context_pred_types.append('collider')
                    elif 'bidirected' in edge_str or '<->' in edge_str:
                        context_pred_types.append('confounder')
                    else:
                        context_pred_types.append('none')
                        
                except Exception as e:
                    continue # Skip failing contexts

            runtime = time.time() - start_time
            
            # --- MAJORITY VOTE AGGREGATION ---
            # 1. Nodes: Keep nodes that appeared in at least 50% of contexts
            node_counts = Counter(context_nodes_found)
            threshold = len(unique_contexts) / 2
            final_nodes = sorted([node for node, count in node_counts.items() if count >= threshold])
            
            # 2. Type: Most frequent prediction type across contexts
            if context_pred_types:
                final_pred_type = Counter(context_pred_types).most_common(1)[0][0]
            else:
                final_pred_type = 'none'
            # --- END CONTEXT-WISE LOGIC ---

            n = meta.get('n')
            seed = meta.get('seed')
            func = meta.get('func')
            lc = meta.get('lc', meta.get('nc', '0'))
            scenario_str = f"n{n}_seed{seed}_func{func}_lc{lc}"

            results.append({
                "scenario": scenario_str,
                "n_nodes": n,
                "seed": seed,
                "func": func,
                "gt_type": meta.get('lt'),
                "predicted_gt_type": final_pred_type,
                "subset_nodes": str(final_nodes),
                "runtime": round(runtime, 4)
            })

        except Exception as e:
            print(f"Error processing {folder.name}: {e}")

    return pd.DataFrame(results)


def run_benchmark(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    results = []
    folders = [f for f in dataset_root.iterdir() if f.is_dir() and "seed" in f.name]
    
    for folder in tqdm(folders):
        try:
            # 1. Load Data & Metadata
            with open(folder / "meta.json", 'r') as f:
                meta = json.load(f)
            
            df = pd.read_csv(folder / "X.csv")
            X_cols = [c for c in df.columns if c.startswith('X')]
            X_data = df[X_cols].to_numpy()

            # 2. Run Discovery
            start_time = time.time()
            discovered_nodes_list = []
            try:
                res = latent_selection_discovery(X_data, alpha=0.01)
                pag_edges = res.get('estimated_L_PAG_edges', [])
                
                # Extract nodes under selection/confounding influence
                # Flatten the mapping of Latent IDs to Observed IDs
                mapping = res.get('estimated_Lid_to_Xids', {})
                unique_nodes = set()
                for observed_indices in mapping.values():
                    for idx in observed_indices:
                        unique_nodes.add(f"X{idx}")
                discovered_nodes_list = sorted(list(unique_nodes))
                
            except Exception as e:
                # print(f"Discovery failed for {folder.name}: {e}")
                pag_edges = []
                
            runtime = time.time() - start_time
            
            # 3. Process Result Type
            pred_type = 'none'
            edge_str = str(pag_edges).lower()
            if 'undirected' in edge_str or '-' in edge_str:
                pred_type = 'collider'
            elif 'bidirected' in edge_str or '<->' in edge_str:
                pred_type = 'confounder'

            # 4. Construct Scenario String and Append
            # Note: Using 'lc' from meta; defaults to 'nc' or '?' if missing
            n = meta.get('n')
            seed = meta.get('seed')
            func = meta.get('func')
            lc = meta.get('lc', meta.get('nc', '0'))
            
            scenario_str = f"n{n}_seed{seed}_func{func}_lc{lc}"

            results.append({
                "scenario": scenario_str,
                "n_nodes": n,
                "seed": seed,
                "func": func,
                "gt_type": meta.get('lt'),
                "predicted_gt_type": pred_type,
                "subset_nodes": str(discovered_nodes_list),
                "runtime": round(runtime, 4)
            })

        except Exception as e:
            print(f"Error processing {folder.name}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    # results_df = run_benchmark("datasets")
    # results_df['method'] = 'latent-selection-pooled'    
    # cols = ['scenario', 'n_nodes', 'seed', 'func', 'gt_type', 'predicted_gt_type', 'subset_nodes', 'runtime', 'method']
    # results_df = results_df[cols]
    # results_df.to_csv("benchmark_results.csv", index=False)
    
    print("Running Pooled Baseline...")
    df_pooled = run_benchmark("datasets")
    df_pooled['method'] = 'latent-selection-pooled'
    df_pooled.to_csv("baseline_latent_selection_pooled.csv", index=False)

    
    print("Running Per-Context Majority Vote Baseline...")
    df_context = run_benchmark_per_context("datasets")
    df_context['method'] = 'latent-selection-per-context-majority'
    df_context.to_csv("baseline_latent_selection_context.csv", index=False)
    # combined_results = pd.concat([df_pooled, df_context], ignore_index=True)
    
    # cols = ['scenario', 'method', 'n_nodes', 'seed', 'func', 'gt_type', 'predicted_gt_type', 'subset_nodes', 'runtime']
    # combined_results = combined_results[cols]
    
    # combined_results.to_csv("benchmark_pooled.csv", index=False)
