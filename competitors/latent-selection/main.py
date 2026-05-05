import random
import numpy as np
import networkx as nx
from itertools import chain
from utils.CCA_tools import Chi2RankTest
from utils.FCI_tools import fci, MAG
from utils.find_measurement_clusters import estimate_Lid_to_Xids
random.seed(42)
np.random.seed(42)

def estimate_latent_pag(
    X_data,
    Lid_to_Xids,
    N_scaling_for_CCA=1,
    alpha_for_CCA=0.05,
):
    '''
    We assume the data generating process is a MAG among latent variables L,
    and each Li has at least two pure measurements in X.
    We use the fact that Li _||_ Lj | LZ <==> rank(Xi+XZ^(1), Xj+XZ^(2)) == |Z|  (otherwise > |Z|)

    :param X_data: observed data; a numpy array of shape (sample_size, num_measurements)
    :param Lid_to_Xids: a dictionary mapping latent variable id to a list of measurement indices
                       if Lid_to_Xids is unknown, please refer to find_measurement_clusters.py to find it from X_data first
                       sometimes Lid_to_Xids is given as domain knowledge; TODO: however, checks may be needed to ensure they are true "clusters"
    :param N_scaling_for_CCA: scaling factor for the sample size in CCA
    :param alpha_for_CCA: significance level for the CCA test
    :return: PAG edges among L
    '''
    assert all(len(v) >= 2 for v in Lid_to_Xids.values()), ""
    all_x_ids = list(chain.from_iterable(Lid_to_Xids.values()))
    assert len(all_x_ids) == len(set(all_x_ids)), "Each measurement can only belong to one latent variable"
    assert set(all_x_ids) == set(range(X_data.shape[1])), "Measurement indices must match that of X_data"
    all_L_ids = list(Lid_to_Xids.keys())

    rank_tester = Chi2RankTest(X_data, N_scaling=N_scaling_for_CCA)
    def ci_tester_among_L(i, j, Z=None):
        if Z is None: Z = []
        assert i != j and i not in Z and j not in Z
        assert {i, j} | set(Z) <= set(all_L_ids)
        Z_measurements = [Lid_to_Xids[l] for l in Z]
        Z_one_half = list(chain.from_iterable([xz[:len(xz)//2] for xz in Z_measurements]))
        Z_another_half = list(chain.from_iterable([xz[len(xz)//2:] for xz in Z_measurements]))
        i_measurements, j_measurements = Lid_to_Xids[i], Lid_to_Xids[j]
        if_fail_to_reject, p, testStat, criticalValue = rank_tester.test(i_measurements + Z_one_half, j_measurements + Z_another_half, len(Z), alpha_for_CCA)
        return bool(if_fail_to_reject)

    pag_edges_among_L = fci(all_L_ids, ci_tester_among_L)
    return pag_edges_among_L


def latent_selection_discovery(X_data, alpha=0.01, oracle_Lid_to_Xids=None):
    estimated_Lid_to_Xids = estimate_Lid_to_Xids(X_data, N_scaling_for_CCA=1, alpha_for_CCA=alpha) if oracle_Lid_to_Xids is None else oracle_Lid_to_Xids
    estimated_L_PAG_edges = estimate_latent_pag(X_data, estimated_Lid_to_Xids, N_scaling_for_CCA=1, alpha_for_CCA=alpha)
    params_est = {
        'estimated_Lid_to_Xids': estimated_Lid_to_Xids,
        'estimated_L_PAG_edges': estimated_L_PAG_edges
    }
    return params_est


if __name__ == '__main__':
    init_sample_size = 100000

    ############ Step1: Generate original data from latents to observed; before selection ############
    # For example, we use a classical example among four latents with selection bias:
    #           L0 → L2
    #            ↓   ↓
    #           L1   L3
    #             ↘  ↙
    #              S
    # The PAG among L is a square with four undirected edges (i.e., FCI tells for sure there's selection bias):
    #           LO - L2
    #            |   |
    #           L1 - L3
    L0 = np.random.normal(0, 1, init_sample_size)
    L1 = L0 + np.random.normal(0, 1, init_sample_size)
    L2 = L0 + np.random.normal(0, 1, init_sample_size)
    L3 = L2 + np.random.normal(0, 1, init_sample_size)
    LS = L1 + L3 + np.random.normal(0, 1, init_sample_size)
    oracle_DAG_among_L = nx.DiGraph()
    oracle_DAG_among_L.add_nodes_from([0, 1, 2, 3, 'LS'])
    oracle_DAG_among_L.add_edges_from([(0, 1), (0, 2), (2, 3), (1, 'LS'), (3, 'LS')])
    oracle_MAG_among_L = MAG(oracle_DAG_among_L, observed_nodes=[0, 1, 2, 3], latent_nodes=[], selected_nodes=['LS'])
    oracle_PAG_among_L = oracle_MAG_among_L.get_PAG_edges()
    L_data = np.stack([L0, L1, L2, L3], axis=1)
    Lid_to_Xids = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
    X_data = np.empty((init_sample_size, sum(len(v) for v in Lid_to_Xids.values())))
    for lid, xids in Lid_to_Xids.items():
        X_data[:, xids] = L_data[:, lid][:, None] + np.random.normal(0, 1, (init_sample_size, len(xids)))

    ############ Step2: Apply selection ############
    selected_mask = np.logical_and.reduce([                                       # there could be more selection variables
        np.logical_and(LS > np.percentile(LS, 30), LS < np.percentile(LS, 70)),   # can be a range; not necessarily point selection.
    ])
    X_data = X_data[selected_mask]

    ############ The following step is for running the `latent_selection_discovery` on your own data  ############
    ############ Step3: Estimate the latents-to-pure-measurements correspondence, and the PAG among L ############
    results_est = latent_selection_discovery(
        X_data,                          # in shape (num_samples, num_measurements)
        alpha=0.01,
        # oracle_Lid_to_Xids=Lid_to_Xids   # sometimes you may already have domain knowledge about measurement clusters (e.g., from questionnaires).
        #                                  # if so, you may uncomment this line to specify it with `oracle_Lid_to_Xids`;
        #                                  # otherwise, the Lid_to_Xids will be estimated from X data first.
    )
    estimated_Lid_to_Xids, estimated_L_PAG_edges = results_est['estimated_Lid_to_Xids'], results_est['estimated_L_PAG_edges']
    print(f'Truth Lid_to_Xids:     {Lid_to_Xids}')
    print(f'Estimated Lid_to_Xids: {estimated_Lid_to_Xids} (subject to Lid permutation)\n')
    print(f'Truth PAG among L:     {oracle_PAG_among_L}')
    print(f'Estimated PAG among L: {estimated_L_PAG_edges}')

