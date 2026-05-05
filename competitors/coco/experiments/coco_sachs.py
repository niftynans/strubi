import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coco.co_co import CoCo
from coco.co_test_types import CoShiftTestType, CoCoTestType, CoDAGType
from coco.mi_sampling import Sampler

def reproduce_full_confounder_test(path="../experiments/data_cytometry"):
    # 1. Configuration
    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    DAG_SEARCH = CoDAGType.SKIP
    ALPHA_SHIFT_TEST = 0.05

    # 2. Define the Ground Truth DAG (Sachs)
    dag = np.zeros((11, 11))
    dag[8, np.asarray([10, 7, 0, 9])] = 1
    dag[2, np.asarray([3, 7])] = 1
    dag[7, 5] = 1
    dag[0, 1] = 1
    dag[3, 8] = 1
    dag[4, np.asarray([10, 7, 9, 1, 6])] = 1
    dag[1, 5] = 1
    dag[5, np.asarray([9, 6, 3])] = 1

    nodes_true = list(range(len(dag)))
    
    # 3. Load Data
    Dc_raw = [pd.read_csv(f'{path}/dataset_{i}.csv') for i in range(1, 10)]
    nms_all = Dc_raw[0].columns
    # Log-transform and truncate to keep contexts balanced (707 samples each)
    Dc_all = np.array([np.log(np.array(X[:707])) for X in Dc_raw])

    results = {}

    # 4. Iterate through every node as the hidden confounder
    for confounder_idx in nodes_true:
        conf_name = nms_all[confounder_idx]
        print(f"\n--- TESTING NODE AS HIDDEN: {confounder_idx} ({conf_name}) ---")

        # Define observed nodes
        obs_indices = [n for n in nodes_true if n != confounder_idx]
        nms_observed = nms_all[obs_indices]
        
        # Prepare observed data (dropping the confounder column)
        Dobs = Dc_all[:, :, obs_indices]

        # Build the observed Graph
        # We must exclude any edge connected to the hidden confounder
        G_observed = nx.DiGraph()
        for i in obs_indices:
            for j in obs_indices:
                if dag[i, j] == 1:
                    G_observed.add_edge(i, j)
        
        # Relabel nodes to be 0 to N-1 to match Dobs column indices
        mapping = {old_node: new_idx for new_idx, old_node in enumerate(obs_indices)}
        G_relabeled = nx.relabel_nodes(G_observed, mapping)

        # Full Graph (for CoCo's internal validation)
        G_true = nx.DiGraph(dag)

        # Dummy class for internal CoCo metadata
        class CoCoMetadata:
            def __init__(self, G, n_contexts):
                self.G = G
                self.G_true = G
                self.maps_nodes = {n: np.zeros(n_contexts, dtype=int) for n in G.nodes()}
                self.maps_nodes_star = self.maps_nodes
                self.nodes_confounded = []
                self.nodes_selection_parents = []

        meta = CoCoMetadata(G_true, n_contexts=len(Dc_all))

        # 5. Initialize and Run CoCo
        try:
            coco = CoCo(
                Dobs, 
                G_relabeled, 
                Sampler(), 
                CONFOUNDING_TEST, 
                SHIFT_TEST, 
                DAG_SEARCH,
                n_components=1, 
                dag=meta, 
                node_nms=nms_observed, 
                alpha_shift_test=ALPHA_SHIFT_TEST
            )

            coco._estimated_graph_cuts_n(1)
            results[conf_name] = coco

            if len(coco.estimated_cuts):
                # Map the relabeled indices back to original names for clarity
                detected_indices = coco.estimated_cuts[0]
                detected_names = [nms_observed[j] for j in detected_indices]
                print(f"  [!] SUCCESS: Hidden Confounder '{conf_name}' detected via cuts: {detected_names}")
            else:
                print(f"  [-] No cuts detected for hidden node {conf_name}")

        except Exception as e:
            print(f"  [X] Error processing {conf_name}: {e}")

    return results

if __name__ == "__main__":
    res = reproduce_full_confounder_test()