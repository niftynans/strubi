#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Haoyue
@file: find_measurement_clusters.py
@time: 1/21/25 03:21
@desc: 
"""
import networkx as nx
from itertools import combinations

from networkx.algorithms.components import connected_components

from utils.CCA_tools import Chi2RankTest

def estimate_Lid_to_Xids(
        X_data,
        N_scaling_for_CCA=1,
        alpha_for_CCA=0.05,
):
    '''
    We assume each Li has at least two pure measurements in X.
    We use the fact that: two measurements Xi and Xj belong to the same latent variable if and only if
                          rank(XiXj, all_remaining_X) <= 1 (otherwise == 2)
    So we first find for each pair of Xi and Xj, whether such low rank is satisfied,
                and then use union set to merge these pairs into clusters.
                TODO (1): however, empirically it's possible that rank(X1X2,..)==1, rank(X2X3,..)==1, but rank(X1X3,..)==2; how to deal with this?

    TODO (2): also, this code may be used for real data, to check if clusters given by domain knowledge are accurate.

    :param X_data: observed data; a numpy array of shape (sample_size, num_measurements)
    :param N_scaling_for_CCA: scaling factor for the sample size in CCA
    :param alpha_for_CCA: significance level for the CCA test

    :return: Lid_to_Xid: a dictionary mapping latent variable id to a list of measurement indices
                         since we do not know the latent IDs, there is permutation indeterminacies.
                         we sort Lid_to_Xid by the smallest index in each cluster.
    '''
    nx_union_set = nx.Graph()
    nx_union_set.add_nodes_from(range(X_data.shape[1]))
    rank_tester = Chi2RankTest(X_data, N_scaling=N_scaling_for_CCA)

    for i, j in combinations(range(X_data.shape[1]), 2):
        if_fail_to_reject, p, testStat, criticalValue = rank_tester.test(
            [i, j], list(set(range(X_data.shape[1])) - {i, j}), 1, alpha_for_CCA)
        if if_fail_to_reject:
            nx_union_set.add_edge(i, j)

    clusters = list(nx.connected_components(nx_union_set))
    clusters = sorted(clusters, key=lambda x: min(x))
    Lid_to_Xid = {i: list(cluster) for i, cluster in enumerate(clusters)}
    return Lid_to_Xid