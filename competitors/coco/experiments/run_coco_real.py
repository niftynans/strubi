import numpy as np
import pandas as pd
import networkx as nx
from coco.co_co import CoCo
from coco.co_test_types import CoShiftTestType, CoCoTestType, CoDAGType
from coco.mi_sampling import Sampler

from coco.fci import FCI_JCI
from experiments.results_coco import MethodType

def reproduce_fig3(path="../experiments/data_cytometry", mode="confounder"):
    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    DAG_SEARCH = CoDAGType.SKIP
    ALPHA_SHIFT_TEST = 0.05

    dag = np.zeros((11, 11))
    dag[8, np.asarray([10, 7, 0, 9])] = 1
    dag[2, np.asarray([3, 7])] = 1
    dag[7, 5] = 1
    dag[0, 1] = 1
    dag[3, 8] = 1
    dag[4, np.asarray([10, 7, 9, 1, 6])] = 1
    dag[1, 5] = 1
    dag[5, np.asarray([9, 6, 3])] = 1 

    
    Dc_raw = [np.log(pd.read_csv(f'{path}/dataset_{i}.csv'))[:707] for i in range(1, 10)]
    nms = Dc_raw[0].columns
    node_pkc = 8
    
    
    processed_contexts = []
    for df in Dc_raw:
        if mode == "selection":
            threshold = df.iloc[:, node_pkc].quantile(0.9)
            df_filtered = df[df.iloc[:, node_pkc] >= threshold].copy()
            processed_contexts.append(df_filtered)
        else:
            processed_contexts.append(df)

    cap = min(len(ctx) for ctx in processed_contexts)
    if mode == "":
        nodes_observed = [n for n in range(len(dag))] 
    else:
        nodes_observed = [n for n in range(len(dag)) if n != node_pkc]
    
    Dc_processed = [np.array(ctx)[:cap, nodes_observed] for ctx in processed_contexts]
    Dobs = np.array(Dc_processed)
    nms_observed = nms[nodes_observed]

    G_observed = nx.DiGraph()
    G_observed.add_nodes_from(range(len(nodes_observed)))
    
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_observed)}
    edges_observed = []
    for i in range(len(dag)):
        for j in range(len(dag)):
            if dag[i, j] == 1 and i != node_pkc and j != node_pkc:
                edges_observed.append((mapping[i], mapping[j]))
    G_observed.add_edges_from(edges_observed)

    # Setup Truth Class
    class TruthDAG:
        def __init__(self, G, n_contexts):
            self.G = G
            self.G_true = G
            self.maps_nodes = {n: np.zeros(n_contexts, dtype=int) for n in G.nodes()}
            self.maps_nodes_star = self.maps_nodes
            self.nodes_confounded = []
            self.nodes_selection_parents = []

    dg = TruthDAG(G_observed, len(Dobs))

    print(f"\n{'='*20} Mode: {mode.upper()} {'='*20}")
    print(f"Samples per context: {cap}")

    # --- 1. Run CoCo ---
    print("\n[Running CoCo...]")
    coco = CoCo(Dobs, G_observed, Sampler(), CONFOUNDING_TEST, SHIFT_TEST, DAG_SEARCH,
                n_components=1, dag=dg, node_nms=nms_observed, alpha_shift_test=ALPHA_SHIFT_TEST)
    coco._estimated_graph_cuts_n(1)

    if len(coco.estimated_cuts):
        print(f"\tCoCo Identified Cuts: {coco.estimated_cuts} "
              f"{[nms_observed[j] for j in coco.estimated_cuts[0]]}")

    # # --- 2. Run JCI-FCI Full ---
    # print("\n[Running JCI-FCI Full...]")
    # try:
    #     fci = FCI_JCI(Dobs, G_observed, G_observed, dg, 
    #                   independence_test='fisherz', 
    #                   method=MethodType.FCI_JCI_FULL)
        
    #     # JCI-FCI identifies confounded pairs (bidirected edges)
    #     tp, fp, tn, fn, f1 = fci.eval_confounded(dg, MethodType.FCI_JCI_FULL)
        
    #     # Extract estimated confounded pairs from the PAG
    #     retrieved_confounders = []
    #     if hasattr(fci, 'estimated_confounders'):
    #         retrieved_confounders = fci.estimated_confounders
            
    #     print(f"\tJCI-FCI F1 (Confounded Pairs): {f1}")
    #     if retrieved_confounders:
    #         readable_pairs = [(nms_observed[u], nms_observed[v]) for u, v in retrieved_confounders]
    #         print(f"\tJCI-FCI Confounded Pairs: {readable_pairs}")
    #     else:
    #         print("\tJCI-FCI: No confounded pairs identified.")
            
    # except Exception as e:
    #     print(f"\tJCI-FCI Failed: {e}")

# Run both scenarios
reproduce_fig3(mode="confounder")
reproduce_fig3(mode="selection")
reproduce_fig3(mode="")