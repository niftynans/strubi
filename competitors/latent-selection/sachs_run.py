import pandas as pd
import numpy as np
import time
from collections import Counter
from pathlib import Path

# Assuming your discovery function is in main.py
from main import latent_selection_discovery

def get_discovery_results(data, alpha=0.01):
    """Wrapper to extract nodes and prediction types from the discovery result."""
    try:
        res = latent_selection_discovery(data, alpha=alpha)
        
        # Extract identified nodes
        mapping = res.get('estimated_Lid_to_Xids', {})
        found_nodes = []
        for observed_indices in mapping.values():
            for idx in observed_indices:
                found_nodes.append(idx)
        
        # Determine prediction type based on PAG edges
        pag_edges = str(res.get('estimated_L_PAG_edges', [])).lower()
        pred_type = 'none'
        if 'undirected' in pag_edges or '-' in pag_edges:
            pred_type = 'collider' # Selection
        elif 'bidirected' in pag_edges or '<->' in pag_edges:
            pred_type = 'confounder' # Latent Confounding
            
        return sorted(list(set(found_nodes))), pred_type
    except Exception as e:
        print(f"Discovery error: {e}")
        return [], 'error'

def run_sachs_scenarios(csv_path):
    # 1. Load and Prepare Data
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Define protein columns (all except Context)
    all_proteins = [c for c in df.columns if c != 'Context']
    protein_to_idx = {name: i for i, name in enumerate(all_proteins)}
    idx_to_protein = {i: name for name, i in protein_to_idx.items()}

    results = []

    # --- SCENARIO SETUP ---
    # Scenario A: Hidden PKC (Drop PKC entirely)
    df_hidden = df.drop(columns=['PKC'])
    proteins_hidden = [c for c in all_proteins if c != 'PKC']
    
    # Scenario B: Selected PKC (Top 20% per context, then drop PKC)
    # We filter for high PKC activity, then hide PKC to check for selection bias detection
    df_selected_list = []
    for ctx in df['Context'].unique():
        ctx_data = df[df['Context'] == ctx].copy()
        threshold = np.percentile(ctx_data['PKC'], 80)
        df_selected_list.append(ctx_data[ctx_data['PKC'] >= threshold])
    
    df_selected = pd.concat(df_selected_list).drop(columns=['PKC'])
    proteins_selected = proteins_hidden # Same remaining columns

    scenarios = [
        ("Hidden PKC", df_hidden, proteins_hidden),
        ("Selected PKC (Top 20%)", df_selected, proteins_selected)
    ]

    for label, data_df, current_proteins in scenarios:
        print(f"\n>>> Running Scenario: {label}")
        
        # --- METHOD 1: POOLED ---
        start_t = time.time()
        X_pooled = data_df[current_proteins].to_numpy()
        found_indices, p_type = get_discovery_results(X_pooled)
        runtime_pooled = time.time() - start_t
        
        results.append({
            "Scenario": label,
            "Method": "Pooled",
            "Discovered Nodes": [current_proteins[i] for i in found_indices],
            "Type": p_type,
            "Runtime": round(runtime_pooled, 4)
        })

        # --- METHOD 2: PER-CONTEXT MAJORITY VOTE ---
        start_t = time.time()
        ctx_nodes = []
        ctx_types = []
        unique_contexts = data_df['Context'].unique()

        for ctx in unique_contexts:
            X_ctx = data_df[data_df['Context'] == ctx][current_proteins].to_numpy()
            f_idx, t = get_discovery_results(X_ctx)
            ctx_nodes.extend([current_proteins[i] for i in f_idx])
            ctx_types.append(t)
        
        # Aggregate: Node must appear in > 50% of contexts
        threshold = len(unique_contexts) / 2
        node_counts = Counter(ctx_nodes)
        final_nodes = [node for node, count in node_counts.items() if count >= threshold]
        
        # Aggregate: Majority type
        final_type = Counter(ctx_types).most_common(1)[0][0] if ctx_types else 'none'
        runtime_ctx = time.time() - start_t

        results.append({
            "Scenario": label,
            "Method": "Per-Context Majority",
            "Discovered Nodes": sorted(final_nodes),
            "Type": final_type,
            "Runtime": round(runtime_ctx, 4)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    final_results = run_sachs_scenarios("sachs_processed.csv")
    print("\n--- FINAL RESULTS ---")
    print(final_results)
    final_results.to_csv("sachs_scenario_results.csv", index=False)