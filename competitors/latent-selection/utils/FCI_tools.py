#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pdb
import networkx as nx
from itertools import chain, combinations, permutations, product


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

AROW, DASH, CIRC = 'AROW', 'DASH', 'CIRC'
LEFT, RIGHT = 'LEFT', 'RIGHT'


def fci(nodelist, CI_tester):
    # this is a simpliest wrapper for FCI without prior knowledge; for more complex cases, refer to the two inner functions
    skeleton_edges, sepsets, dependencies = get_skeleton_and_sepsets(nodelist, CI_tester)
    edgemark_dict, update_reasons = get_PAG_from_skeleton_and_sepsets(nodelist, skeleton_edges, sepsets)
    pag_edges = translate_edgemarks_to_readable(edgemark_dict)
    return pag_edges


def translate_edgemarks_to_readable(CURREDGES):
    pag_edges = {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
    for (node1, node2), (type1, type2) in CURREDGES.items():
        if type1 == DASH and type2 == AROW:
            pag_edges['->'].add((node1, node2))
        elif type1 == AROW and type2 == AROW:
            pag_edges['<->'].add((node1, node2))  # has symmetric repeats
        elif type1 == DASH and type2 == DASH:
            pag_edges['--'].add((node1, node2))  # has symmetric repeats
        elif type1 == CIRC and type2 == DASH:
            pag_edges['⚬--'].add((node1, node2))
        elif type1 == CIRC and type2 == AROW:
            pag_edges['⚬->'].add((node1, node2))
        elif type1 == CIRC and type2 == CIRC:
            pag_edges['⚬-⚬'].add((node1, node2))  # has symmetric repeats
    return pag_edges


def translate_probs_to_readable(CURREDGES, is_PAG=True):
    pag_edges = {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
    for (node1, node2), edge_type_dict in CURREDGES.items():
        node1 = int(node1)
        node2 = int(node2)

        # find the corresponding edge type
        edge_type = max(edge_type_dict, key=edge_type_dict.get)
        translate_dict = {'-->': (DASH, AROW), '<->': (AROW, AROW), '---': (DASH, DASH), 'o--': (CIRC, DASH), 'o->': (CIRC, AROW), 'o-o': (CIRC, CIRC),
                          '<--': (AROW, DASH), '--o': (DASH, CIRC), '<-o': (AROW, CIRC)}
        (type1, type2) = translate_dict[edge_type]
        if is_PAG == False: # output is a CPDAG
            # Turn the tails into circles
            if type1 == DASH:
                type1 = CIRC
            if type2 == DASH:
                type2 = CIRC
        if type1 == DASH and type2 == AROW:
            pag_edges['->'].add((node1, node2))
        elif type1 == AROW and type2 == AROW:
            pag_edges['<->'].add((node1, node2))  # has symmetric repeats
        elif type1 == DASH and type2 == DASH:
            pag_edges['--'].add((node1, node2))  # has symmetric repeats
        elif type1 == CIRC and type2 == DASH:
            pag_edges['⚬--'].add((node1, node2))
        elif type1 == CIRC and type2 == AROW:
            pag_edges['⚬->'].add((node1, node2))
        elif type1 == CIRC and type2 == CIRC:
            pag_edges['⚬-⚬'].add((node1, node2))  # has symmetric repeats
    return pag_edges


def get_PAG_from_skeleton_and_sepsets(
    nodelist,
    skeleton_edges,
    sepsets,
    background_knowledge_edges=None,
    sure_no_latents=False,
    sure_no_selections=False,
    rules_to_use=None,
):
    '''
    :param nodelist: enumerate of nodes
    :param skeleton_edges: enumerate of tuples of nodes; no need to be symmetric
    :param sepsets: dict{(i, j): S}; assert all (i, j) not in skeleton_edges should have a sepset
    :param background_knowledge_edges: a dictionary like {(i, j): (DASH, AROW)}
    :param sure_no_latents: boolean; if True, all (CIRC, AROW) is oriented as (DASH, AROW)
    :param sure_no_selections: boolean;
    :param rules_to_use: None, of subset of [1, .., 10]
    :return:
    '''
    assert set().union(*skeleton_edges) <= set(nodelist)
    assert all(set(k) | set(v) <= set(nodelist) for k, v in sepsets.items())
    ALLNODES = set(nodelist)
    CURREDGES, SEPSETS = {}, {}
    UPDATEREASONS = {}
    for x, y in skeleton_edges: CURREDGES[(x, y)] = CURREDGES[(y, x)] = (CIRC, CIRC)
    for (node1, node2), Z in sepsets.items(): SEPSETS[(node1, node2)] = SEPSETS[(node2, node1)] = set(Z)
    assert len(set(CURREDGES.keys()) & set(SEPSETS.keys())) == 0
    assert set(CURREDGES.keys()) | set(SEPSETS.keys()) == {(x, y) for x, y in product(nodelist, nodelist) if x != y}
    if background_knowledge_edges is not None:
        assert set(background_knowledge_edges.keys()) <= set(CURREDGES.keys())
    UNSHIELDED_TRIPLES = set()
    UNSHIELDED_TRIPLE_EDGES = set()
    for x, y in combinations(nodelist, 2):
        for z in ALLNODES - {x, y}:
            if (x, y) not in CURREDGES.keys() and {(x, z), (y, z)} <= set(CURREDGES.keys()):
                UNSHIELDED_TRIPLES |= {(x, y, z), (y, x, z)}
                UNSHIELDED_TRIPLE_EDGES |= {(x, z), (y, z), (z, x), (z, y)}

    def get_curr_edge_type(node1, node2, end=LEFT):
        if (node1, node2) not in CURREDGES: return False
        if end == LEFT:
            return CURREDGES[(node1, node2)][0]
        elif end == RIGHT:
            return CURREDGES[(node1, node2)][1]
        assert False

    def update_edge(node1, node2, type1, type2, reason=None):
        # print("update_edge", node1, node2, type1, type2, reason)
        curr_type_1, curr_type_2 = CURREDGES[(node1, node2)]
        if type1 is None: type1 = curr_type_1
        if type2 is None: type2 = curr_type_2
        if curr_type_1 != CIRC and curr_type_1 != type1:
            print(f"[WARNING] [INNER FCI] Conflict detected: Attempt to change '{curr_type_1}' to '{type1}' was not successful.")
            type1 = curr_type_1
        if curr_type_2 != CIRC and curr_type_2 != type2:
            print(f"[WARNING] [INNER FCI] Conflict detected: Attempt to change '{curr_type_2}' to '{type2}' was not successful.")
            type2 = curr_type_2
        CURREDGES[(node1, node2)] = (type1, type2)
        CURREDGES[(node2, node1)] = (type2, type1)
        UPDATEREASONS[(node1, node2, type1, type2)] = UPDATEREASONS[(node2, node1, type2, type1)] = reason

    def _RO():
        # print('RO...')
        for alpha, beta in combinations(ALLNODES, 2):  # safe here; it's symmetric
            if (alpha, beta) in CURREDGES: continue
            for gamma in ALLNODES - {alpha, beta}:
                if (alpha, gamma) in CURREDGES and (beta, gamma) in CURREDGES:
                    gamma_not_in_sepset = gamma not in SEPSETS[(alpha, beta)]
                    if gamma_not_in_sepset:
                        update_edge(alpha, gamma, None, AROW, reason='R0')
                        update_edge(beta, gamma, None, AROW, reason='R0')

    def _R1():
        # print('R1...')
        # If α∗→ β◦−−∗ γ, and α and γ are not adjacent, then orient the triple as α∗→ β →γ
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if get_curr_edge_type(alpha, bt, RIGHT) == AROW]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, LEFT) == CIRC and (alpha, gm) not in CURREDGES]:
                    update_edge(beta, gamma, DASH, AROW, reason='R1')
                    changed_something = True
        return changed_something

    def _R2():
        # print('R2...')
        # If α→β∗→ γ or α∗→ β →γ, and α ∗−◦ γ,then orient α ∗−◦ γ as α∗→γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, RIGHT) == AROW]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, RIGHT) == AROW]:
                    if get_curr_edge_type(alpha, gamma, RIGHT) == CIRC:
                        if get_curr_edge_type(alpha, beta, LEFT) == DASH or \
                           get_curr_edge_type(beta, gamma, LEFT) == DASH:
                            changed_something = True
                            update_edge(alpha, gamma, None, AROW, reason='R2')
        return changed_something

    def _R3():
        # print('R3...')
        # If α∗→ β ←∗γ, α ∗−◦ θ ◦−∗ γ, α and γ are not adjacent, and θ ∗−◦ β, then orient θ ∗−◦ β as θ∗→ β.
        changed_something = False
        for alpha, gamma in combinations(ALLNODES, 2):  # safe here; it's symmetric
            if (alpha, gamma) in CURREDGES: continue
            for beta in [bt for bt in ALLNODES - {alpha, gamma} if
                         get_curr_edge_type(alpha, bt, RIGHT) == AROW and \
                         get_curr_edge_type(bt, gamma, LEFT) == AROW]:
                for theta in [th for th in ALLNODES - {alpha, beta, gamma} if
                         get_curr_edge_type(alpha, th, RIGHT) == CIRC and \
                         get_curr_edge_type(th, gamma, LEFT) == CIRC]:
                    if get_curr_edge_type(theta, beta, RIGHT) == CIRC:
                        changed_something = True
                        update_edge(theta, beta, None, AROW, reason='R3')
        return changed_something

    def _R4():
        # print('R4...')
        # If u = <θ, ...,α,β,γ> is a discriminating path between θ and γ for β, and β◦−−∗γ;
        # then if β ∈ Sepset(θ,γ), orient β◦−−∗ γ as β →γ; otherwise orient the triple <α,β,γ> as α ↔β ↔γ.
        changed_something = False
        for theta in ALLNODES:
            for gamma in ALLNODES - {theta}:
                if (theta, gamma) in CURREDGES: continue
                for beta in {bt for bt in ALLNODES - {theta, gamma} if
                             get_curr_edge_type(bt, gamma, LEFT) == CIRC}:
                    gamma_parents = {af for af in ALLNODES - {theta, gamma, beta} if \
                                     get_curr_edge_type(af, gamma, LEFT) == DASH and
                                     get_curr_edge_type(af, gamma, RIGHT) == AROW}
                    if len(gamma_parents) < 1: continue
                    # to prevent from nx.all_simple_paths(self.mag_undirected_graph, ..) (too slow) we use subgraph to only allow paths through gamma_parents
                    subgraph = nx.Graph()   # undirected
                    subgraph.add_nodes_from(gamma_parents | {theta, beta})
                    subgraph.add_edges_from([(x, y) for x, y in combinations(gamma_parents | {theta, beta}, 2) if (x, y) in CURREDGES])
                    for theta_beta_path in nx.all_simple_paths(subgraph, theta, beta):
                        if len(theta_beta_path) < 3: continue
                        path = theta_beta_path + [gamma]
                        if all(
                                get_curr_edge_type(path[i - 1], path[i], RIGHT) == AROW and
                                get_curr_edge_type(path[i], path[i + 1], LEFT) == AROW
                                for i in range(1, len(path) - 2)
                        ):
                            changed_something = True
                            # print('    R4 applied!')
                            if beta in SEPSETS[(theta, gamma)]:
                                update_edge(beta, gamma, DASH, AROW, reason='R4')
                            else:
                                # in this faithful single intervention twin graph, it won't happen
                                update_edge(path[-3], beta, AROW, AROW, reason='R4')
                                update_edge(beta, gamma, AROW, AROW, reason='R4')
        return changed_something

    def _R5():
        # print('R5...')
        # For every (remaining) α◦−−◦β, if there is an uncovered circle path p =?α,γ,...,θ,β? between α and β s.t. α,θ are
        # not adjacent and β,γ are not adjacent, then orient α◦−−◦β and every edge on p as undirected edges (--)
        #  i.e., to ensure the graph remains chordal, no colliders allowed.
        changed_something = False
        current_circ_circ_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if types12 == (CIRC, CIRC) and x < y} & UNSHIELDED_TRIPLE_EDGES
        subgraph = nx.Graph()  # undirected
        subgraph.add_nodes_from(set().union(*current_circ_circ_edges_that_also_belongs_to_UTs))
        subgraph.add_edges_from(current_circ_circ_edges_that_also_belongs_to_UTs)
        for cycle in nx.cycle_basis(subgraph):
            if len(cycle) < 4: continue
            cycle_is_uncovered = all((cycle[nid - 1], cycle[((nid + 1) % len(cycle))]) not in CURREDGES for nid in range(len(cycle)))
            if cycle_is_uncovered:
                changed_something = True
                for nid in range(len(cycle)):
                    update_edge(cycle[nid - 1], cycle[nid], DASH, DASH, reason='R5')
        return changed_something

    def _R6():
        # print('R6...')
        # If α—β◦−−∗ γ (α and γ may or may not be adjacent), then orient β◦−−∗ γ as β −−∗ γ.
        #  (this is because for any α—-β, both α and β must be ancestors of S)
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) == DASH]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              get_curr_edge_type(beta, gm, LEFT) == CIRC]:
                    changed_something = True
                    update_edge(beta, gamma, DASH, None, reason='R6')
        return changed_something

    def _R7():
        # print('R7...')
        # If α −−◦ β◦−−∗ γ, and α, γ are not adjacent, then orient β◦−−∗ γ as β −−∗ γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) == CIRC]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                              (alpha, gm) not in CURREDGES and
                              get_curr_edge_type(beta, gm, LEFT) == CIRC]:
                    changed_something = True
                    update_edge(beta, gamma, DASH, None, reason='R7')
        return changed_something

    def _R8():
        # print('R8...')
        # If α→β →γ or α−−◦β →γ, and α◦→γ,orient α◦→γ as α→γ.
        changed_something = False
        for alpha in ALLNODES:
            for beta in [bt for bt in ALLNODES - {alpha} if
                         get_curr_edge_type(alpha, bt, LEFT) == DASH and get_curr_edge_type(alpha, bt, RIGHT) in [AROW, CIRC]]:
                for gamma in [gm for gm in ALLNODES - {alpha, beta} if
                        get_curr_edge_type(beta, gm, LEFT) == DASH and get_curr_edge_type(beta, gm, RIGHT) == AROW]:
                    if get_curr_edge_type(alpha, gamma, LEFT) == CIRC and get_curr_edge_type(alpha, gamma, RIGHT) == AROW:
                        changed_something = True
                        update_edge(alpha, gamma, DASH, None, reason='R8')
        return changed_something

    def _R9():
        # print('R9...')
        # If α→β →γ or α−−◦β →γ, and α◦−−∗γ,orient α◦−−∗γ as α−−∗γ.
        changed_something = False
        circ_arrow_edges_exists = any(types == (CIRC, AROW) for types in CURREDGES.values())
        if not circ_arrow_edges_exists: return False
        current_semi_directed_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if
             types12 not in [(AROW, AROW), (DASH, DASH)] and x < y} & UNSHIELDED_TRIPLE_EDGES
        pretend_directed_edges = []
        for x, y in current_semi_directed_edges_that_also_belongs_to_UTs:
            type1, type2 = CURREDGES[(x, y)]
            if type1 == DASH or type2 == AROW:
                pretend_directed_edges.append((x, y))
            elif type1 == AROW or type2 == DASH:
                pretend_directed_edges.append((y, x))
            else:
                assert type1 == CIRC and type2 == CIRC
                pretend_directed_edges.extend([(x, y), (y, x)])
        sub_directed_graph = nx.DiGraph()
        sub_directed_graph.add_edges_from(pretend_directed_edges)
        for alpha in sub_directed_graph.nodes():
            for gamma in [gm for gm in set(sub_directed_graph.nodes()) - {alpha} if
                          get_curr_edge_type(alpha, gm, LEFT) == CIRC and
                          get_curr_edge_type(alpha, gm, RIGHT) == AROW]:
                for path in nx.all_simple_paths(sub_directed_graph, alpha, gamma):
                    if len(path) < 4: continue
                    cycle_is_uncovered = all(
                        (path[nid - 1], path[((nid + 1) % len(path))]) not in CURREDGES for nid in range(len(path)))
                    if cycle_is_uncovered:
                        changed_something = True
                        update_edge(alpha, gamma, DASH, None, reason='R9')
                        break
        return changed_something


    def _R10():
        # print('R10...')
        # Suppose α◦→γ, β →γ ←θ, p1 is an uncovered p.d. path from α to β, and p2 is an uncovered p.d. path from α to
        # θ.Let μ be the vertex adjacent to α on p1 (μ could be β), and ω be the vertex adjacent to α on p2 (ω could be θ).
        # If μ and ω are distinct, and are not adjacent, then orient α◦→γ as α→γ.
        changed_something = False
        circ_arrow_edges_exists = any(types == (CIRC, AROW) for types in CURREDGES.values())
        dash_arrow_edges_exists = any(types == (DASH, AROW) for types in CURREDGES.values())
        if not (circ_arrow_edges_exists and dash_arrow_edges_exists): return False
        current_semi_directed_edges_that_also_belongs_to_UTs = \
            {(x, y) for (x, y), types12 in CURREDGES.items() if
             types12 not in [(AROW, AROW), (DASH, DASH)] and x < y} & UNSHIELDED_TRIPLE_EDGES
        pretend_directed_edges = []
        for x, y in current_semi_directed_edges_that_also_belongs_to_UTs:
            type1, type2 = CURREDGES[(x, y)]
            if type1 == DASH or type2 == AROW:
                pretend_directed_edges.append((x, y))
            elif type1 == AROW or type2 == DASH:
                pretend_directed_edges.append((y, x))
            else:
                assert type1 == CIRC and type2 == CIRC
                pretend_directed_edges.extend([(x, y), (y, x)])
        sub_directed_graph = nx.DiGraph()
        sub_directed_graph.add_edges_from(pretend_directed_edges)
        for alpha in sub_directed_graph.nodes():
            for gamma in [gm for gm in set(sub_directed_graph.nodes()) - {alpha} if
                          get_curr_edge_type(alpha, gm, LEFT) == CIRC and
                          get_curr_edge_type(alpha, gm, RIGHT) == AROW]:
                gamma_parents = {p for p in sub_directed_graph.nodes() if
                                 get_curr_edge_type(p, gamma, LEFT) == DASH and
                                 get_curr_edge_type(p, gamma, RIGHT) == AROW}
                already_done_orientation = False
                for beta, theta in combinations(gamma_parents, 2):
                    if already_done_orientation: break
                    for path1 in nx.all_simple_paths(sub_directed_graph, alpha, beta):
                        if already_done_orientation: break
                        for path2 in nx.all_simple_paths(sub_directed_graph, alpha, theta):
                            mu, omega = path1[1], path2[1]
                            if mu != omega and (mu, omega) not in CURREDGES:
                                changed_something = True
                                already_done_orientation = True
                                update_edge(alpha, gamma, DASH, None, reason='R10')
                                break
        return changed_something

    def _R_no_latents():
        # print('R_no_latents...')
        # when we are sure that there are no latents, we confirm all ⚬-> as ->
        changed_something = False
        for (node1, node2), (type1, type2) in CURREDGES.items():
            if type1 == CIRC and type2 == AROW:
                update_edge(node1, node2, DASH, AROW, reason='no_latents')
                changed_something = True
        return changed_something

    def _R_no_selections():
        # print('R_no_selections...')
        # when we are sure that there are no selection, we confirm all ⚬-- as <-
        changed_something = False
        for (node1, node2), (type1, type2) in CURREDGES.items():
            if type1 == CIRC and type2 == DASH:
                update_edge(node1, node2, AROW, DASH, reason='no_selections')
                changed_something = True
        return changed_something

    # def _R_variables_affected():
    #     changed_something = False
    #     for (node1, node2), (type1, type2) in CURREDGES.items():
    #         if type1 == CIRC and type2 == AROW and node1 not in variables_affected:
    #             update_edge(node1, node2, DASH, AROW, reason='variables_affected')
    #             changed_something = True
    #     return changed_something

    # ============================= main part ======================================
    # first apply background knowledge (for now we dont do consistency check; just trust it)
    if background_knowledge_edges is not None:
        for (node1, node2), (type1, type2) in background_knowledge_edges.items():
            update_edge(node1, node2, type1, type2, reason='background')
            update_edge(node2, node1, type2, type1, reason='background')

    # then fix the unshielded triples using observed CIs
    _RO()

    # then iteratively apply the rules until no more changes
    rule_id_to_func = {1: _R1, 2: _R2, 3: _R3, 4: _R4, 5: _R5, 6: _R6, 7: _R7, 8: _R8, 9: _R9, 10: _R10}
    if rules_to_use is None: rules_to_use = list(range(1, 11))
    RULES = [rule_id_to_func[rule_id] for rule_id in rules_to_use]
    if sure_no_latents: RULES.append(_R_no_latents)
    if sure_no_selections: RULES.append(_R_no_selections)
    while True:
        changed_something = False
        for rule in RULES:
            changed_something |= rule()
        if not changed_something:
            break

    # finally, return the result
    # pag_edges = translate_edgemarks_to_readable(CURREDGES)
    return CURREDGES, UPDATEREASONS

def get_skeleton_and_sepsets(nodelist, CI_tester, sure_adjacencies=None, sure_dependencies=None):
    '''
    :param nodelist: a list of nodes
    :param CI_tester: a function that takes in (i, j, S) and returns True or False; i, j in nodelist; S subset of nodelist
    :param sure_adjacencies: list of tuples of nodes that are known to be adjacent; skip tests on them; always i < j
    :param sure_dependencies: list of (i, j, frozenset(S)) tuples that are known to be dependent, always i < j
    :return:
        skeleton: a list of tuples (i, j) representing edges in the skeleton; i < j always
        sepsets: a dictionary of the form {(i, j): S} where i indep j | S; i < j always
        dependencies: a set of (i, j, frozenset(S)) tuples of dependencies found; i < j always
    '''
    ALLNODES = set(nodelist)
    curr_skeleton = [(min(i, j), max(i, j)) for i, j in combinations(ALLNODES, 2)]
    curr_neighbors = {i: set(ALLNODES) - {i} for i in ALLNODES}
    sure_adjacencies = set(sure_dependencies) if sure_adjacencies is not None else set()
    sure_dependencies = set(sure_dependencies) if sure_dependencies is not None else set()
    Sepsets = {}
    Dependencies = set(sure_dependencies)
    l = -1
    while True:
        l += 1
        found_something = False
        visited_pairs = set()
        while True:
            this_i, this_j = None, None
            for i, j in curr_skeleton:
                if (i, j) in visited_pairs: continue
                if (i, j) in sure_adjacencies: continue
                assert j in curr_neighbors[i]
                if len(curr_neighbors[i]) - 1 >= l or len(curr_neighbors[j]) - 1 >= l:
                    this_i, this_j = i, j
                    found_something = True
                    break
            if this_i is None: break
            visited_pairs.add((this_i, this_j))
            choose_subset_from = set(map(frozenset, combinations(curr_neighbors[this_i] - {this_j}, l))) | \
                                 set(map(frozenset, combinations(curr_neighbors[this_j] - {this_i}, l)))
            for subset in choose_subset_from:
                if (this_i, this_j, frozenset(subset)) in sure_dependencies: continue
                if CI_tester(this_i, this_j, subset):
                    curr_skeleton.remove((this_i, this_j))
                    curr_neighbors[this_i].remove(this_j)
                    curr_neighbors[this_j].remove(this_i)
                    Sepsets[(this_i, this_j)] = set(subset)
                    break
                else:
                    Dependencies.add((this_i, this_j, frozenset(subset)))
        if not found_something: break

    ## so far it is done for PC (causal sufficiency case); however for FCI and MAG/PAG,
    ## we have to do skeleton refinement (some nonadjacencies may not have sepsets from just neighbors)
    while True:
        CURREDGES, _ = get_PAG_from_skeleton_and_sepsets(
            ALLNODES,
            curr_skeleton,
            Sepsets,
            background_knowledge_edges=None,
            sure_no_latents=False,
            sure_no_selections=False,
            rules_to_use=[],  # no R1 to R10, instead just R0 for vstrucs
        )
        Possible_Dsep = {i: {j for j in ALLNODES if (i, j) in CURREDGES.keys()} for i in ALLNODES}
        undiG = nx.Graph()
        undiG.add_nodes_from(ALLNODES)
        undiG.add_edges_from(curr_skeleton)
        for i in ALLNODES:
            all_simple_paths = list(chain.from_iterable([list(nx.all_simple_paths(undiG, i, j)) for j in ALLNODES - {i}]))
            longest_paths = []
            for path in all_simple_paths:
                if len(path) > 2 and not any(len(other_path) > len(path) and other_path[:len(path)] == path for other_path in all_simple_paths):
                    longest_paths.append(path)
            for path in longest_paths:
                for cid in range(1, len(path) - 1):
                    is_collider_or_unknown_in_triangle = ((CURREDGES[(path[cid - 1], path[cid])][1] == AROW and
                                              CURREDGES[(path[cid], path[cid + 1])][0] == AROW) or
                                              ((path[cid - 1], path[cid + 1]) in CURREDGES.keys()))
                    if is_collider_or_unknown_in_triangle: Possible_Dsep[i].add(path[cid + 1])
                    else: break  # this path cannot go on

        REMOVED_SOMETHING_NEW = False
        for this_i, this_j in curr_skeleton:
            choose_subset_from = set(map(frozenset, powerset(Possible_Dsep[this_i] - {this_j}))) | set(map(frozenset, powerset(Possible_Dsep[this_j] - {this_i})))
            choose_subset_from = {subset for subset in choose_subset_from if (not subset <= curr_neighbors[this_i]) and (not subset < curr_neighbors[this_j])}
            for subset in choose_subset_from:
                if (this_i, this_j, frozenset(subset)) in sure_dependencies: continue
                if CI_tester(this_i, this_j, subset):
                    curr_skeleton.remove((this_i, this_j))
                    curr_neighbors[this_i].remove(this_j)
                    curr_neighbors[this_j].remove(this_i)
                    Sepsets[(this_i, this_j)] = set(subset)
                    REMOVED_SOMETHING_NEW = True
                    break
                else:
                    Dependencies.add((this_i, this_j, frozenset(subset)))
        if not REMOVED_SOMETHING_NEW: break

    return set(curr_skeleton), Sepsets, Dependencies


def is_two_MAGs_equivalent(mag1, mag2, verbose=False):
    assert mag1.observed_nodes == mag2.observed_nodes
    for x, y in combinations(mag1.observed_nodes, 2):
        for Z in powerset(mag1.observed_nodes - {x, y}):
            mag_indep_1 = mag1.oracle_ci_with_selection(x, y, Z)
            mag_indep_2 = mag2.oracle_ci_with_selection(x, y, Z)
            if mag_indep_1 != mag_indep_2:
                if verbose: print(f'CI for {x} and {y} given {Z}: {mag_indep_1} vs {mag_indep_2}')
                return False
    return True


class MAG(object):

    def __init__(self, nxDiG, observed_nodes, latent_nodes, selected_nodes):
        assert set(observed_nodes) | set(latent_nodes) | set(selected_nodes) == set(nxDiG.nodes)
        assert set(observed_nodes) & set(latent_nodes) == set() and set(observed_nodes) & set(
            selected_nodes) == set() and set(latent_nodes) & set(selected_nodes) == set()
        self.observed_nodes = set(observed_nodes)
        self.latent_nodes = set(latent_nodes)
        self.selected_nodes = set(selected_nodes)

        self.dag = nxDiG
        assert nx.is_directed_acyclic_graph(self.dag)
        self.dag_parents = {i: set(self.dag.predecessors(i)) for i in self.dag.nodes}
        self.dag_children = {i: set(self.dag.successors(i)) for i in self.dag.nodes}
        self.dag_ancestors = {i: nx.ancestors(self.dag, i) | {i} for i in self.dag.nodes}  # including itself
        self.dag_descendants = {i: nx.descendants(self.dag, i) | {i} for i in self.dag.nodes}  # including itself

        self.init_MAG()

    def oracle_ci_with_selection(self, x, y, Z=None, allow_access_latents=False):
        if not hasattr(self, 'CI_cache'): self.CI_cache = {}
        Z = set() if Z is None else set(Z)
        assert x != y and {x, y} & Z == set()
        if not allow_access_latents: assert ({x, y} | Z) <= self.observed_nodes
        # allow_access_latents is a sanity check (usually only CIs among observd can be queried); but sometimes
        #    due to deterministic relations, condition on Xi also conditioned on Xi*, so we allow this backdoor call.
        x, y = min(x, y), max(x, y)
        cachekey = (x, y, frozenset(Z))
        if cachekey not in self.CI_cache:
            # BE SURE TO INCLUDE ALL THE SELECTED NODES HERE!
            self.CI_cache[cachekey] = nx.is_d_separator(self.dag, {x}, {y}, set(Z) | self.selected_nodes)
        return self.CI_cache[cachekey]

    def is_m_separated(self, x, y, Z=None):
        # confirmed: it's equivalent to oracle_ci_with_selection. But prevent from using it (too slow in finding paths)
        Z = set() if Z is None else set(Z)
        assert x != y and {x, y} & Z == set() and ({x, y} | Z) <= self.observed_nodes
        for path in nx.all_simple_paths(self.mag_undirected_graph, x, y):
            colliders_on_path = {path[i] for i in range(1, len(path) - 1) if
                                 {path[i - 1], path[i + 1]} <= self.mag_parents[path[i]] | self.mag_spouses[path[i]]}
            noncolliders_on_path = {path[i] for i in range(1, len(path) - 1) if path[i] not in colliders_on_path}
            if len(noncolliders_on_path & set(Z)) == 0 and all(
                    len(self.mag_descendants[c] & set(Z)) > 0 for c in colliders_on_path):
                return False
        return True


    def init_MAG(self):
        def _exists_inducing_path(x, y):
            # This is correct but we want to prevent using it: too slow!!
            for path in nx.all_simple_paths(self.dag.to_undirected(), x, y):
                observed_and_selected_nodes_on_path = {path[i] for i in range(1, len(path) - 1) if
                                                       path[i] in self.observed_nodes | self.selected_nodes}
                colliders_on_path = {path[i] for i in range(1, len(path) - 1) if
                                     {path[i - 1], path[i + 1]} <= self.dag_parents[path[i]]}
                ancestors_of_x_y_and_S = (self.dag_ancestors[x] | self.dag_ancestors[y]).union(
                    *[self.dag_ancestors[s] for s in self.selected_nodes])
                if observed_and_selected_nodes_on_path <= colliders_on_path and colliders_on_path <= ancestors_of_x_y_and_S:
                    return True
            return False

        # Since checking for inducing path (nx.all_simple_paths) is too slow, we run PC phase 1 to get adjacencies.
        # pdb.set_trace()
        self.skeleton_edges_in_mag, self.Sepset_cache, self.Dependencies_cache = \
            get_skeleton_and_sepsets(self.observed_nodes, self.oracle_ci_with_selection) ##TODO check complexity
        # adjacencies_found_by_inducing_path = {tuple(sorted([x, y]))
        #                 for x, y in combinations(self.observed_nodes, 2) if _exists_inducing_path(x, y)}
        # assert adjacencies_found_by_inducing_path == self.skeleton_edges_in_mag

        self.mag_edges = {'->': set(), '<->': set(), '--': set()}
        for x, y in self.skeleton_edges_in_mag:   # set of tuples (i, j) where i < j
            x_causes_y_or_selection = x in self.dag_ancestors[y].union(
                *[self.dag_ancestors[s] for s in self.selected_nodes])
            y_causes_x_or_selection = y in self.dag_ancestors[x].union(
                *[self.dag_ancestors[s] for s in self.selected_nodes])
            if x_causes_y_or_selection and not y_causes_x_or_selection:
                self.mag_edges['->'].add((x, y))
            elif not x_causes_y_or_selection and y_causes_x_or_selection:
                self.mag_edges['->'].add((y, x))
            elif not x_causes_y_or_selection and not y_causes_x_or_selection:
                self.mag_edges['<->'] |= {(x, y), (y, x)}
            else:
                self.mag_edges['--'] |= {(x, y), (y, x)}

        self.mag_undirected_graph = nx.Graph()
        self.mag_undirected_graph.add_nodes_from(self.observed_nodes)
        self.mag_undirected_graph.add_edges_from(self.skeleton_edges_in_mag)
        self.mag_only_directed_edges_graph = nx.DiGraph()
        self.mag_only_directed_edges_graph.add_nodes_from(self.observed_nodes)
        self.mag_only_directed_edges_graph.add_edges_from(self.mag_edges['->'])
        self.mag_parents = {i: set(self.mag_only_directed_edges_graph.predecessors(i)) for i in self.observed_nodes}
        self.mag_children = {i: set(self.mag_only_directed_edges_graph.successors(i)) for i in self.observed_nodes}
        self.mag_ancestors = {i: nx.ancestors(self.mag_only_directed_edges_graph, i) | {i} for i in self.observed_nodes}
        self.mag_descendants = {i: nx.descendants(self.mag_only_directed_edges_graph, i) | {i} for i in
                                self.observed_nodes}
        self.mag_spouses = {i: {j for j in self.observed_nodes if (i, j) in self.mag_edges['<->']} for i in
                            self.observed_nodes}
        self.mag_neighbors = {i: {j for j in self.observed_nodes if (i, j) in self.mag_edges['--']} for i in
                              self.observed_nodes}

    def get_MAG_orientations_based_on_given_skeleton(self, alternative_skeleton_edges):
        '''
        sometimes the skeleton found just by induced path (graphical criteria) may be too much..
        e.g., there are conditional independencies not captured by d/m-separations, in the twinG case.
        so we first use true CI to learn the skeleton, and then input here for orientation.
        '''
        mag_edges = {'->': set(), '<->': set(), '--': set()}
        for x, y in alternative_skeleton_edges:  # set of tuples (i, j) where i < j
            x_causes_y_or_selection = x in self.dag_ancestors[y].union(
                *[self.dag_ancestors[s] for s in self.selected_nodes])
            y_causes_x_or_selection = y in self.dag_ancestors[x].union(
                *[self.dag_ancestors[s] for s in self.selected_nodes])
            if x_causes_y_or_selection and not y_causes_x_or_selection:
                mag_edges['->'].add((x, y))
            elif not x_causes_y_or_selection and y_causes_x_or_selection:
                mag_edges['->'].add((y, x))
            elif not x_causes_y_or_selection and not y_causes_x_or_selection:
                mag_edges['<->'] |= {(x, y), (y, x)}
            else:
                mag_edges['--'] |= {(x, y), (y, x)}
        return mag_edges

    def get_PAG_edges(self):
        self.pag_edges = translate_edgemarks_to_readable(
            get_PAG_from_skeleton_and_sepsets(
                self.observed_nodes,
                self.skeleton_edges_in_mag,
                self.Sepset_cache
            )[0]
        )
        return self.pag_edges




if __name__ == '__main__':
    dag = nx.DiGraph()
    dag.add_nodes_from(['A', 'B', 'F', 'C', 'H', 'D', 'E', 'T1', 'T2'])
    dag.add_edges_from([('T1', 'A'), ('T1', 'B'), ('T2', 'D'), ('T2', 'E'), ('B', 'E'), ('F', 'B'), ('C', 'F'), ('C', 'H'), ('H', 'D'), ('D', 'A')])
    mag = MAG(dag, ['A', 'B', 'F', 'C', 'H', 'D', 'E'], ['T1', 'T2'], [])
    print(mag.mag_edges)

    pag_edges = get_PAG_from_skeleton_and_sepsets(
        mag.observed_nodes,
        mag.skeleton_edges_in_mag,
        mag.Sepset_cache,
        background_knowledge_edges=None)
    #

