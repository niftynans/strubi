import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from main import latent_selection_discovery
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

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

            # 2. Run Discovery (Auto-estimates clusters)
            start_time = time.time()
            try:
                # This estimates clusters AND finds the PAG
                res = latent_selection_discovery(X_data, alpha=0.01)
                pag_edges = res['estimated_L_PAG_edges']
                n_discovered = len(res['estimated_Lid_to_Xids'])
            except Exception:
                pag_edges, n_discovered = [], 0
                
            runtime = time.time() - start_time
            
            # 3. Process Result
            pred_type = 'none'
            edge_str = str(pag_edges).lower()
            if 'undirected' in edge_str or '-' in edge_str:
                pred_type = 'collider'
            elif 'bidirected' in edge_str or '<->' in edge_str:
                pred_type = 'confounder'

            results.append({
                "n_nodes": meta.get('n'),
                "seed": meta.get('seed'),
                "func": meta.get('func'),
                "gt_type": meta.get('lt'),
                "predicted_gt_type": pred_type, # if n_discovered >= 2 else 'unidentifiable',
                "runtime": round(runtime, 4)
            })

        except Exception as e:
            print(f"Error in {folder.name}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_benchmark("latent-selection/datasets")
    df['method'] = 'latent-selection'
    df.to_csv("benchmark_results.csv", index=False)
    # df = df[df['func'] == 'linear']

    # Define ground truth and predictions for the linear subset
    y_true_lin = df['gt_type']
    y_pred_lin = df['predicted_gt_type']

    # Calculate Metrics
    acc_lin = accuracy_score(y_true_lin, y_pred_lin)
    p_lin, r_lin, f1_lin, _ = precision_recall_fscore_support(
        y_true_lin, y_pred_lin, average='macro', zero_division=0
    )
    report_lin = classification_report(y_true_lin, y_pred_lin, zero_division=0)

    print(f"--- Metrics for LINEAR Functional Cases ({len(df)} samples) ---")
    print(f"Accuracy:  {acc_lin:.4f}")
    print(f"Macro Precision: {p_lin:.4f}")
    print(f"Macro Recall:    {r_lin:.4f}")
    print(f"Macro F1 Score:  {f1_lin:.4f}")
    print(f"\n--- Detailed Class Report for Linear cases ---\n")
    print(report_lin)