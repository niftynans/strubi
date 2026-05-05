import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys   
from scipy.stats import spearmanr, pearsonr
import random
import itertools

















def apply_func(x, func_name):
    if func_name == 'linear': return x
    elif func_name == 'x^2': return x**2 
    elif func_name == 'x^3': return x**3
    elif func_name == 'tanh': return np.tanh(x)
    elif func_name == 'sinc': return np.sinc(x)
    else: return x

















def print_node_shifts(maps_obs, maps_conf, maps_coll):
    print("\n--- Individual Node Shift Assignments ---")
    
    all_maps = [
        ("Observed", maps_obs),
        ("Confounder", maps_conf),
        ("Collider", maps_coll)
    ]
    
    for label, node_map in all_maps:
        for node_id, contexts in node_map.items():
            # Find indices where the context is active (value == 1)
            shift_indices = np.where(contexts == 1)[0].tolist()
            
            if shift_indices:
                print(f"{label} Node {node_id: >3} shifts in contexts: {shift_indices}")
            else:
                print(f"{label} Node {node_id: >3} has NO shifts.")







def gen_exclusive_maps_old(n_contexts, obs_nodes, confounders, colliders, seed, obs_shift_ratio=0.25, dense=False):
    rng = np.random.RandomState(seed)
    available_contexts = list(range(1, n_contexts))
    rng.shuffle(available_contexts)
    
    num_obs = len(obs_nodes)
    num_conf = len(confounders)
    num_coll = len(colliders)
    
    # We want total_shifts = (num_obs * k) + (num_conf * 2k) + (num_coll * 2k)
    # total_shifts = k * (num_obs + 2*num_conf + 2*num_coll)
    weighted_total = num_obs + 2 * (num_conf + num_coll)
    
    global_max_shifts = int(n_contexts * 0.48)
    
    if weighted_total > 0:
        k = global_max_shifts // weighted_total
    else:
        k = 0
        
    k = max(1, k)
    
    if k * weighted_total > len(available_contexts):
        k = len(available_contexts) // weighted_total

    print(f"Entities: {num_obs + num_conf + num_coll} | Total Contexts: {n_contexts}")
    print(f"Result: Obs shifts: {0} | Confounders/Colliders shifts: {2*k}") # 0 for now, replace 0 by k or k / 2 later.

    maps_obs = {n: np.zeros(n_contexts, dtype=int) for n in obs_nodes}
    maps_conf = {n: np.zeros(n_contexts, dtype=int) for n in confounders}
    maps_coll = {n: np.zeros(n_contexts, dtype=int) for n in colliders}
    
    for id_ in confounders:
        for _ in range(2 * k):
            if not available_contexts: break
            ctx = available_contexts.pop()
            maps_conf[id_][ctx] = 1
            
    for id_ in colliders:
        for _ in range(2 * k):
            if not available_contexts: break
            ctx = available_contexts.pop()
            maps_coll[id_][ctx] = 1

    return maps_obs, maps_conf, maps_coll




def gen_exclusive_maps(n_contexts, obs_nodes, confounders, colliders, seed, obs_shift_ratio=0.25, dense=0.48):
    rng = np.random.RandomState(seed)
    available_contexts = list(range(1, n_contexts))
    rng.shuffle(available_contexts)
    
    num_obs = len(obs_nodes)
    num_lat = len(confounders) + len(colliders)
    
    sparsity_limit = dense if dense else 0.48
    global_max_shifts = int(n_contexts * sparsity_limit)
    
    denominator = num_lat + (obs_shift_ratio * num_obs)
    
    if denominator > 0:
        L = max(1, int(global_max_shifts // denominator))
    else:
        L = 0
        
    obs_k = int(obs_shift_ratio * L)

    print(f"Entities: {num_obs + num_lat} | Total Contexts: {n_contexts}")
    print(f"Shift Budget: Latent={L}, Observed={obs_k} (Ratio r={obs_shift_ratio})")

    maps_obs = {n: np.zeros(n_contexts, dtype=int) for n in obs_nodes}
    maps_conf = {n: np.zeros(n_contexts, dtype=int) for n in confounders}
    maps_coll = {n: np.zeros(n_contexts, dtype=int) for n in colliders}
    
    for id_ in (confounders + colliders):
        target_map = maps_conf if id_ in confounders else maps_coll
        for _ in range(L):
            if not available_contexts: break
            ctx = available_contexts.pop()
            target_map[id_][ctx] = 1
            
    for id_ in obs_nodes:
        for _ in range(obs_k):
            if not available_contexts: break
            ctx = available_contexts.pop()
            maps_obs[id_][ctx] = 1

    return maps_obs, maps_conf, maps_coll


class DAGConfoundedWithSelection:
    def __init__(self, seed, n_contexts, n_observed_nodes, n_confounders,
                n_colliders, func=None, signal_boost=1.0,  latent_type=None,  fixed_latent_size=None, obs_shift_ratio=0.25, dense_val=0.48):
        np.random.seed(seed)
        self.seed = seed
        self.rng_init = np.random.RandomState(seed)
        self.n_c = n_contexts
        self.n_nodes = n_observed_nodes
        self.n_confounders = n_confounders
        self.n_colliders = n_colliders
        self.signal_boost = signal_boost
        
        if func is not None:
            self.func = func    
        else:
            self.func = self.rng_init.choice(['linear'])
        
        self._gen_structure_and_partitions_no_backdoors_scaled(fixed_latent_size=fixed_latent_size)
        
        conf_ids = list(range(self.n_confounders))
        coll_ids = list(range(self.n_colliders))
        obs_ids = list(range(self.n_nodes))
        
        self.maps_nodes_star, self.maps_confounders, self.maps_colliders = gen_exclusive_maps(
            self.n_c, obs_ids, conf_ids, coll_ids, self.seed, obs_shift_ratio=obs_shift_ratio, dense=dense_val
        )

        self.mechanisms = {}
        all_nodes = list(self.G_true.nodes)
        self.collider_parents = set()
        
        start_colliders = self.n_nodes + self.n_confounders
        start_confounders = self.n_nodes
        collider_nodes = range(start_colliders, start_colliders + self.n_colliders)
        
        for s_node in collider_nodes:
            parents = list(self.G_true.predecessors(s_node))
            for p in parents:
                self.collider_parents.add(p)
        
        for node in all_nodes:
            start_colliders = self.n_nodes + self.n_confounders
            is_collider = node >= start_colliders
            is_confounder = self.n_nodes <= node < start_colliders
            parents = list(self.G_true.predecessors(node))
            
            self.mechanisms[node] = {
                0: self._roll_mechanism_params(node, parents, is_collider, is_confounder, start_confounders, start_colliders, state_id=0),
                1: self._roll_mechanism_params(node, parents, is_collider, is_confounder, start_confounders, start_colliders, state_id=1)
            }
        
        self.adj_observed = nx.to_numpy_array(self.G_observed, nodelist=range(self.n_nodes))
        all_nodes_list = list(range(self.n_nodes + self.n_confounders + self.n_colliders))
        self.adj_full = nx.to_numpy_array(self.G_true, nodelist=all_nodes_list)



    def _roll_mechanism_params(self, node, parents, is_collider, is_confounder, start_confounders, start_colliders, state_id):
        params = {}
        params['noise_scale'] = 0.05 
        
        def sample_bias(base_val):
            return base_val + np.random.uniform(-0.5, 0.5) if base_val > 0 else 0.0

        if is_collider: 
            params['active_selection'] = (state_id == 1)
            if self.func in ['x^3']:
                params['bias'] = sample_bias(10.0 if state_id == 1 else 5.0)
                weight_val = np.random.uniform(9.0, 11.0)
                params['parents'] = [{'parent': p, 'weight': weight_val, 'func': self.func} for p in parents]
            else:
                params['bias'] = sample_bias(5.0 if state_id == 1 else 0.0)
                weight_val = np.random.uniform(4.0, 6.0)
                params['parents'] = [{'parent': p, 'weight': weight_val, 'func': self.func} for p in parents]
            params['selection_mode'] = 'tail'

        elif is_confounder:
            params['bias'] = sample_bias(5.0 if state_id == 1 else 0.0)
            params['parents'] = [] 
            params['active_selection'] = False

        else:
            params['bias'] = sample_bias(5.0 if state_id == 1 else 0.0)
            params['active_selection'] = False
            
            parent_configs = []
            for p in parents:
                is_p_conf = (p >= start_confounders and p < start_colliders)
                if is_p_conf:
                    w = np.random.uniform(1.7, 2.3)
                else:
                    w = np.random.uniform(0.7, 1.3)
                parent_configs.append({'parent': p, 'weight': w, 'func': self.func})
            params['parents'] = parent_configs
            
        return params

    def _gen_structure_and_partitions_no_backdoors_scaled(self, fixed_latent_size=None):
        if self.n_nodes < 3:
            print("No can do")
            sys.exit()
        elif self.n_nodes < 4:
            print("here 2")
            self.G_true = nx.DiGraph()                          
            self.G_observed = nx.DiGraph()
            obs_nodes = list(range(self.n_nodes))
            self.G_true.add_nodes_from(obs_nodes)
            self.G_observed.add_nodes_from(obs_nodes)
            
            rng = np.random.RandomState(self.seed)
            all_nodes_ordered = list(rng.permutation(obs_nodes))
            
            root = all_nodes_ordered[0]
            target1 = all_nodes_ordered[1]
            target2 = all_nodes_ordered[2]
            target_nodes = [target1, target2]
            print(target_nodes)
            self.nodes_selection_parents = []
            self.nodes_confounded = []
            current_latent_idx = self.n_nodes
            
            self.G_true.add_edge(root, target1)
            self.G_observed.add_edge(root, target1)


            if self.n_colliders > 0:
                s_node = current_latent_idx
                self.G_true.add_node(s_node)
                self.nodes_selection_parents.append(target_nodes)
                for p in target_nodes:
                    self.G_true.add_edge(p, s_node)

            elif self.n_confounders > 0:
                z_node = current_latent_idx
                self.G_true.add_node(z_node)
                self.nodes_confounded.append(target_nodes)
                for c in target_nodes:
                    self.G_true.add_edge(z_node, c)

                
                
            
        else:
            self.G_true = nx.DiGraph()                          
            self.G_observed = nx.DiGraph()
            obs_nodes = list(range(self.n_nodes))
            self.G_true.add_nodes_from(obs_nodes)
            self.G_observed.add_nodes_from(obs_nodes)
            
            rng_struct = np.random.RandomState(self.seed)
            indices = rng_struct.permutation(obs_nodes)
            edge_prob = 0.5 
            
            for idx, i in enumerate(indices):
                for j in indices[idx+1:]: 
                    if rng_struct.rand() < edge_prob:
                        self.G_true.add_edge(i, j)
                        self.G_observed.add_edge(i, j)
            
            all_nodes_ordered = list(nx.topological_sort(self.G_observed))
            # k = self.n_nodes // 2
            if fixed_latent_size is not None:
                k = fixed_latent_size 
            else:
                k = max(2, self.n_nodes // 2)
            if self.n_nodes <= 4:
                pool = all_nodes_ordered[1:]
            else:
                pool = all_nodes_ordered[1:-1]

            self.nodes_selection_parents = []
            self.nodes_confounded = []
            current_latent_idx = self.n_nodes

            for _ in range(self.n_colliders):
                if len(pool) >= k:
                    s_node = current_latent_idx
                    current_latent_idx += 1
                    self.G_true.add_node(s_node)
                    
                    parents = list(rng_struct.choice(pool, k, replace=False))
                    self.nodes_selection_parents.append(parents)
                    
                    for p in parents:
                        self.G_true.add_edge(p, s_node)
                    
                    pool = [n for n in pool if n not in parents]

            for _ in range(self.n_confounders):
                if len(pool) >= k:
                    z_node = current_latent_idx
                    current_latent_idx += 1
                    self.G_true.add_node(z_node)
                    
                    children = list(rng_struct.choice(pool, k, replace=False))
                    
                    for u, v in itertools.combinations(children, 2):
                        if nx.has_path(self.G_observed, u, v) or nx.has_path(self.G_observed, v, u):
                            start, end = (u, v) if nx.has_path(self.G_observed, u, v) else (v, u)
                            if self.G_observed.has_edge(start, end):
                                self.G_observed.remove_edge(start, end)
                                self.G_true.remove_edge(start, end)

                    self.nodes_confounded.append(children)
                    for c in children:
                        self.G_true.add_edge(z_node, c)
                    
                    pool = [n for n in pool if n not in children]

        print(f"Selection Groups: {self.nodes_selection_parents}")
        print(f"Confounding Groups: {self.nodes_confounded}")








    def gen_data(self, seed, n_samp, slice_width=None, gap_params=None): 
        n_gen = n_samp * 50  
        cols_obs = [f"X{i}" for i in range(self.n_nodes)]
        cols_conf = [f"Z{i}" for i in range(self.n_confounders)]
        cols_coll = [f"S{i}" for i in range(self.n_colliders)]
        all_columns = cols_obs + cols_conf + cols_coll
        start_conf, start_coll = self.n_nodes, self.n_nodes + self.n_confounders
        topo = list(nx.topological_sort(self.G_true))
        list_dfs = []

        for c in range(self.n_c):
            ctx_rng = np.random.RandomState(seed + c * 999)
            data_c = np.zeros((n_gen, len(self.G_true.nodes)))
            rows = n_gen
            
            for node in topo:
                if node < start_conf: p_id = self.maps_nodes_star[node][c]
                elif node < start_coll: p_id = self.maps_confounders[node-start_conf][c]
                else: p_id = self.maps_colliders[node-start_coll][c]
                
                mech = self.mechanisms[node][p_id]
                val = np.full(rows, mech['bias'])
                
                for p_config in mech['parents']:
                    parent_data = data_c[:rows, p_config['parent']]
                    if self.func in ['x^2', 'x^3']:
                        parent_data_clipped = np.clip(parent_data, -3.5, 3.5)
                        transformed_signal = apply_func(parent_data_clipped, p_config['func'])    
                        sig_std = transformed_signal.std()
                        if sig_std > 1e-6:
                            normalized_signal = transformed_signal / sig_std
                        else:
                            normalized_signal = transformed_signal
                        val += p_config['weight'] * normalized_signal
                    else:
                        val += p_config['weight'] * apply_func(parent_data, p_config['func'])
                if ctx_rng.random() < 0.5:                                                              # NEW CODE (539 - 543)
                    noise = ctx_rng.normal(0, mech['noise_scale'], rows)
                else:
                    limit = np.sqrt(3) * mech['noise_scale']
                    noise = ctx_rng.uniform(-limit, limit, rows)
                # data_c[:rows, node] = val + ctx_rng.normal(0, mech['noise_scale'], rows)
                data_c[:rows, node] = val + noise

                if node >= start_coll:
                    s_vals = data_c[:rows, node]
                    if mech.get('active_selection', False) and mech.get('selection_mode') == 'tail':
                        if self.func in ['x^3']:
                            mask = s_vals >= np.quantile(s_vals, 0.95)                            
                        else:
                            mask = s_vals >= np.quantile(s_vals, 0.85)
                    else:
                        dist = np.abs(s_vals - np.median(s_vals))
                        mask = dist <= np.quantile(dist, 0.20)
                    data_c = data_c[mask]
                    rows = data_c.shape[0]

            df_c = pd.DataFrame(data_c[:min(rows, n_samp)], columns=all_columns)
            df_c['Context'] = c
            list_dfs.append(df_c)

        full_df = pd.concat(list_dfs, ignore_index=True)
        for col in all_columns:
            m, s = full_df[col].mean(), full_df[col].std()
            if s > 1e-6:
                full_df[col] = (full_df[col] - m) / s
            else:
                full_df[col] = full_df[col] - m
                
        return full_df, self.adj_observed, self.adj_full







    def gen_data_old_lintanhsinc(self, seed, n_samp, slice_width=None, gap_params=None): 
        n_gen = n_samp * 50  
        cols_obs = [f"X{i}" for i in range(self.n_nodes)]
        cols_conf = [f"Z{i}" for i in range(self.n_confounders)]
        cols_coll = [f"S{i}" for i in range(self.n_colliders)]
        all_columns = cols_obs + cols_conf + cols_coll
        start_conf, start_coll = self.n_nodes, self.n_nodes + self.n_confounders
        topo = list(nx.topological_sort(self.G_true))
        list_dfs = []
        for c in range(self.n_c):
            ctx_rng = np.random.RandomState(seed + c * 999)
            data_c = np.zeros((n_gen, len(self.G_true.nodes)))
            rows = n_gen
            for node in topo:
                if node < start_conf: p_id = self.maps_nodes_star[node][c]
                elif node < start_coll: p_id = self.maps_confounders[node-start_conf][c]
                else: p_id = self.maps_colliders[node-start_coll][c]
                mech = self.mechanisms[node][p_id]
                val = mech['bias']
                
                for p_config in mech['parents']:
                    val += p_config['weight'] * apply_func(data_c[:rows, p_config['parent']], p_config['func'])
                data_c[:rows, node] = val + ctx_rng.normal(0, mech['noise_scale'], rows)
                
                if node >= start_coll:
                    s_vals = data_c[:rows, node]
                    if mech.get('active_selection', False) and mech.get('selection_mode') == 'tail':
                        mask = s_vals >= np.quantile(s_vals, 0.85)
                    else:
                        dist = np.abs(s_vals - np.median(s_vals))
                        mask = dist <= np.quantile(dist, 0.20)
                    data_c = data_c[mask]
                    rows = data_c.shape[0]
            df_c = pd.DataFrame(data_c[:min(rows, n_samp)], columns=all_columns)
            print(f"\n[DEBUG] Context {c} | Function: {self.func}")
            for col in all_columns:
                raw_mean = df_c[col].mean()
                raw_std = df_c[col].std()
                raw_max = df_c[col].max()
                raw_min = df_c[col].min()
                snr = abs(raw_mean) / (raw_std + 1e-9)    
                print(f"  Node {col}: Mean={raw_mean:8.4f}, Std={raw_std:8.4f}, Range=[{raw_min:8.2f}, {raw_max:8.2f}], SNR={snr:4.2f}")
            for col in all_columns:
                col_data = df_c[col]
            if col_data.std() > 1e-6:
                df_c[col] = (col_data - col_data.mean()) / col_data.std()
            else:
                df_c[col] = col_data - col_data.mean()
            df_c['Context'] = c
            list_dfs.append(df_c)
        return pd.concat(list_dfs, ignore_index=True), self.adj_observed, self.adj_full














    def print_equations(self):
        def get_name(idx):
            if idx < self.n_nodes: return f"X{idx}"
            elif idx < self.n_nodes + self.n_confounders: return f"Z{idx - self.n_nodes}"
            else: return f"S{idx - (self.n_nodes + self.n_confounders)}"

        all_nodes = sorted(list(self.G_true.nodes))
        print(f"\n{'='*80}")
        print(f"{'NODE':<6} | {'STATE':<8} | EQUATION")
        print(f"{'='*80}")

        for node in all_nodes:
            target = get_name(node)
            for state_id, state_name in [(0, "Inherent Base"), (1, "Inherent Shift")]:
                mech = self.mechanisms[node][state_id]
                parts = []
                if abs(mech['bias']) > 0.01: parts.append(f"{mech['bias']:.2f}")
                if 'parents' in mech:
                    for p in mech['parents']:
                        p_name = get_name(p['parent'])
                        w = p['weight']
                        func = p['func']
                        if func == 'linear': term = f"{w:.2f}*{p_name}"
                        else: term = f"{w:.2f}*{func}({p_name})"
                        parts.append(term)
                parts.append(f"N(0, {mech['noise_scale']:.2f})")
                eq_str = " + ".join(parts)
                if mech.get('active_selection', False): eq_str += ""
                print(f"{target:<6} | {state_name:<8} | {target} = {eq_str}")
            print(f"{'-'*80}")
        print(f"{'='*80}\n")