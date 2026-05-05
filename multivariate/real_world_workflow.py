import numpy as np
import networkx as nx
import os
import pandas as pd
import sys
import glob
import matplotlib.pyplot as plt
# from causalchamber.datasets import Dataset
# from causalchamber import ground_truth
import seaborn as sns
from info import run_multivariate_kci
from synthetic_workflow import plot_context_data
import datetime
import math
import json





def draw_causal_graph(G, title="sachs"):
    plt.figure(figsize=(10, 8))
    pos = nx.shell_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='green', edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray', arrowsize=20, arrowstyle='-|>',connectionstyle='arc3,rad=0.1') 
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_weight='bold')
    plt.title(title, fontsize=15, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title}.png")






def plot_raw_distributions(df, mode='cytometry', highlight_nodes=None):
    if mode == 'cytometry':
        columns = ['Raf', 'Mek', 'Erk', 'PKA', 'PKC', 'Akt', 'P38', 'Jnk', 'Plcg', 'PIP2', 'PIP3']
        context_map = {
            1: "cd3cd28", 2: "cd3cd28.icam2", 3: "cd3cd28_aktinhib",
            4: "cd3cd28_g0076", 5: "cd3cd28_psitect", 6: "cd3cd28_u0126",
            7: "cd3cd28_ly", 8: "pma", 9: "b2camp", 10: "cd3cd28icam2_aktinhib"
        }
        fig_title = "Sachs Cytometry: Raw Protein Distributions"
    else:
        columns = [c for c in df.columns if c != 'Context']
        unique_ctx = sorted(df['Context'].unique())
        context_map = {c: f"Ref_Bin_{c}" if c <= 10 else f"Strong_Exp_{c}" for c in unique_ctx}
        fig_title = "Light Tunnel: Raw Sensor Distributions"
    highlight_nodes = highlight_nodes or []
    df_plot = df.copy()
    df_plot['Context_ID'] = pd.to_numeric(df_plot['Context'], errors='coerce')
    unique_contexts = sorted(df_plot['Context_ID'].unique())
    n_total = len(unique_contexts)
    n_cols = 5
    n_rows = math.ceil(n_total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=False, sharey=False)
    fig.suptitle(fig_title, fontsize=26, y=1.02)
    axes = axes.flatten()
    for i, ctx_id in enumerate(unique_contexts):
        ax = axes[i]
        context_data = df_plot[df_plot['Context_ID'] == ctx_id]
        label = context_map.get(ctx_id, f"Ctx {ctx_id}")
        if mode == 'light_tunnel' and ctx_id <= 10:
            ax.set_facecolor('#E3F2FD') 
        elif ctx_id > 0:
            ax.set_facecolor('#FDF2F2')
        if context_data.empty:
            ax.text(0.5, 0.5, "EMPTY", ha='center', va='center', transform=ax.transAxes)
            continue
        melted = context_data.melt(value_vars=columns, var_name='Node', value_name='Raw Value')
        palette = {n: "#E64A19" if n in highlight_nodes else "#0288D1" for n in columns}

        sns.violinplot(
            data=melted, x='Node', y='Raw Value', ax=ax, 
            inner="quartile", palette=palette, hue='Node', legend=False, linewidth=1
        )
        ax.set_title(f"{ctx_id}: {label}", fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_name = f"viz_raw_{mode}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_name}")
    plt.show()





class CausalTestbed:
    def __init__(self, cytometry_path="data/data_cytometry/dataset_*.csv"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cytometry_path = os.path.join(base_dir, cytometry_path)
        # self.light_tunnel_path = os.path.join(base_dir, "data/data_light_tunnel")
        # self.wind_tunnel_path = os.path.join(base_dir, "data/data_wind_tunnel")
        # os.makedirs(self.light_tunnel_path, exist_ok=True)
        # os.makedirs(self.wind_tunnel_path, exist_ok=True)
        
        self.cytometry_priority = ['Raf', 'Mek', 'Erk', 'PKA', 'PKC', 'Akt', 'P38', 'Jnk', 'Plcg', 'PIP2', 'PIP3']
        # self.light_tunnel_priority = ['timestamp', 'counter', 'flag', 'intervention', 'red',
        # 'green', 'blue', 'osr_c', 'v_c', 'current', 'pol_1', 'pol_2',
        # 'osr_angle_1', 'osr_angle_2', 'v_angle_1', 'v_angle_2', 'angle_1',
        # 'angle_2', 'ir_1', 'vis_1', 'ir_2', 'vis_2', 'ir_3', 'vis_3', 'l_11',
        # 'l_12', 'l_21', 'l_22', 'l_31', 'l_32', 'diode_ir_1', 'diode_vis_1',
        # 'diode_ir_2', 'diode_vis_2', 'diode_ir_3', 'diode_vis_3', 't_ir_1',
        # 't_vis_1', 't_ir_2', 't_vis_2', 't_ir_3', 't_vis_3', 'camera',
        # 'v_board', 'v_reg']

        # self.wind_tunnel_priority = ["hatch", "pot_1", "pot_2", "osr_1", "osr_2", "osr_mic", "osr_in", "osr_out", "osr_upwind", 
        # "osr_downwind", "osr_ambient", "osr_intake", "v_1", "v_2", "v_mic", "v_in", "v_out", #"load_in", 
        # "load_out", "current_in", "current_out", "res_in", "res_out", "rpm_in", "rpm_out", "pressure_upwind", "pressure_downwind", "pressure_ambient", 
        # "pressure_intake", "mic", "signal_1", "signal_2"]






    def load_cytometry(self, normalize=True):
        all_dfs = []
        files = glob.glob(self.cytometry_path)
        proteins = ['Raf', 'Mek', 'Erk', 'PKA', 'PKC', 'Akt', 'P38', 'Jnk', 'Plcg', 'PIP2', 'PIP3']
        if not files:
            raise FileNotFoundError(f"No cytometry files found at {self.cytometry_path}")
        for f in files:
            df = pd.read_csv(f, sep=None, engine='python')
            df.columns = [c.split('.')[-1].split('\t')[-1] for c in df.columns]
            dataset_name = os.path.basename(f).replace('dataset_', '').replace('.csv', '')
            df['Context'] = dataset_name
            if normalize:
                df[proteins] = np.arcsinh(df[proteins] / 5)
            all_dfs.append(df)        
        full_df = pd.concat(all_dfs, ignore_index=True)
        if normalize:
            baseline_mask = full_df['Context'].isin(['0', 'baseline'])
            baseline_df = full_df[baseline_mask]
            if not baseline_df.empty:
                print(f"Normalizing all data relative to baseline statistics...")
                mu = baseline_df[proteins].mean()
                sigma = baseline_df[proteins].std() + 1e-9
                full_df[proteins] = (full_df[proteins] - mu) / sigma
            else:
                print("Warning: Baseline context not found for scaling. Check filenames.")
                full_df[proteins] = (full_df[proteins] - full_df[proteins].mean()) / (full_df[proteins].std() + 1e-9)
        return full_df





    # def load_light_tunnel(self):
    #     ds = Dataset(name='lt_interventions_standard_v1', root=self.light_tunnel_path, download=True)
    #     all_experiments = ds.available_experiments()
    #     all_dfs = []
    #     ref_df = ds.get_experiment('uniform_reference').as_pandas_dataframe()
    #     bin_size = 1000
    #     for j in range(10):
    #         df_bin = ref_df.iloc[j*bin_size : (j+1)*bin_size].copy()
    #         df_bin['Context'] = j + 1
    #         all_dfs.append(df_bin)
    #     red_interventions = [exp for exp in all_experiments if 'red' in exp and 'strong' not in exp]
    #     blue_interventions = [exp for exp in all_experiments if 'blue' in exp and 'strong' not in exp]
    #     green_interventions = [exp for exp in all_experiments if 'green' in exp and 'strong' not in exp]
    #     i = 11
    #     for name in red_interventions + green_interventions + blue_interventions:
    #         df_exp = ds.get_experiment(name).as_pandas_dataframe()
    #         df_exp = df_exp.sample(n=1000, random_state=42)
    #         df_exp['Context'] = i
    #         all_dfs.append(df_exp)
    #         i += 1
    #     return pd.concat(all_dfs, ignore_index=True)




    # def load_wind_tunnel(self, samples_per_context=500):
    #     ds = Dataset(name='wt_pressure_control_v1', 
    #                 root=self.wind_tunnel_path, download=True)
        
    #     all_experiments = ds.available_experiments()
    #     all_dfs = []
        
    #     for ctx_id, name in enumerate(all_experiments, start=1):
    #         df_exp = ds.get_experiment(name).as_pandas_dataframe()
            
    #         n = min(samples_per_context, len(df_exp))
    #         df_exp = df_exp.sample(n=n, random_state=42).copy()
            
    #         df_exp['Context'] = ctx_id 
    #         df_exp['_experiment'] = name
    #         all_dfs.append(df_exp)
            
    #     full_df = pd.concat(all_dfs, ignore_index=True)
    #     print(f"Wind tunnel: Loaded {len(all_experiments)} contexts from wt_pressure_control_v1")
    #     return full_df





    def get_dataset(self, source='cytometry'):
        if source == 'cytometry':
            df = self.load_cytometry()
            priority = self.cytometry_priority
        # elif source == 'light_tunnel':
        #     df = self.load_light_tunnel()
        #     priority = self.light_tunnel_priority
        # else: # wind_tunnel
        #     df = self.load_wind_tunnel()
        #     priority = self.wind_tunnel_priority

        selected = [c for c in priority if c in df.columns]
        subset_df = df[selected + ['Context']].copy().dropna()
        G = self._generate_gt(source, selected)
        return subset_df, G





    
    def _generate_gt(self, source, cols):
        G = nx.DiGraph()
        G.add_nodes_from(cols)
        
        if source == 'cytometry':
            edges = [('Plcg', 'PIP2'), ('Plcg', 'PKA'), ('PIP2', 'PKC'), ('PKC', 'Raf'), ('PKC', 'PKA'), 
                    ('PKC', 'P38'), ('PKC', 'Jnk'), ('PKA', 'Erk'), ('Raf', 'Mek'), ('Mek', 'Erk'), 
                    ('PIP3', 'Akt'), ('PIP3', 'Mek'), ('PIP3', 'P38'), ('PIP3', 'PKA'), ('PIP3', 'Jnk'), 
                    ('Erk', 'P38'), ('Erk', 'Akt'), ('Erk', 'PIP2')]
        # elif source == 'light_tunnel':
        #     gt_graph = ground_truth.graph(chamber="lt", configuration="standard")
        #     edges = [(u, v) for u in gt_graph.index for v in gt_graph.columns if gt_graph.loc[u, v] > 0]
        # else: # wind_tunnel
        #     gt_graph = ground_truth.graph(chamber="wt", configuration="standard")
        #     edges = [(u, v) for u in gt_graph.index for v in gt_graph.columns if gt_graph.loc[u, v] > 0]

        G.add_edges_from([(u, v) for u, v in edges if u in cols and v in cols])
        return G





if __name__ == "__main__":
    if not os.path.exists("plots"): os.makedirs("plots")
    lab = CausalTestbed()
    results_list = []
    csv_master = "multivariate/results/real_world/sachs.csv" 
    
    nodes = ['PKC']  
    scenarios = []
    for node in nodes:
        scenarios.extend(
            [
            ('cytometry', 'collider',  node),
            ('cytometry', 'confounder', node),
            ('cytometry', '', node)
            ]
        )
    example = 0
    for source, lt, target in scenarios:
        print(f"\n--- Running Scenario: {source} | {lt} on {target} ---")
        
        X_raw, G_full = lab.get_dataset(source=source)
        
        # if source == 'light_tunnel':
        #     X_raw = X_raw.drop(columns=['flag', 'counter', 'intervention', 'timestamp', 'v_reg', 'v_board', 'camera'])
        #     G_full.remove_nodes_from(['flag', 'counter', 'intervention', 'timestamp', 'v_reg', 'v_board', 'camera'])
            
        #     cols_to_scale = [c for c in X_raw.columns if c != 'Context']
        #     X_raw[cols_to_scale] = (X_raw[cols_to_scale] - X_raw[cols_to_scale].mean()) / (X_raw[cols_to_scale].std() + 1e-9)
            
        #     sensor_target = 'vis_1'
        #     root_competitor = 'blue'
        #     X_raw['probe'] = (2.0 * X_raw[sensor_target]) - (0.5 * X_raw[root_competitor])
            
        #     unique_ctx = X_raw['Context'].unique()
        #     shift_ctx = np.random.choice(unique_ctx, size=2, replace=False)
            
        #     X_raw.loc[X_raw['Context'].isin(shift_ctx), 'probe'] += 15.0 
        #     X_raw['probe'] += np.random.normal(0, 0.05, size=len(X_raw))
            
        #     G_full.add_node('probe')
        #     G_full.add_edge(sensor_target, 'probe')
        #     G_full.add_edge(root_competitor, 'probe')

            
        #     for node in nx.ancestors(G_full, 'probe'):
        #         print(node)
        #         print(X_raw[node].mean(), X_raw[node].std())
        #     # sys.exit()
        X_raw.to_csv('sachs_processed.csv')
        G_dict = nx.to_dict_of_lists(G_full)
        print(G_dict)
        with open('G_obs.json', 'w') as f:
            json.dump(G_dict, f)

        print(X_raw.columns)
        print(X_raw['Context'].value_counts())
        print('\n\n')
        if target not in X_raw.columns:
            continue
        if lt == 'confounder':
            latent_label = 'Z' 
        elif lt == 'collider':
            latent_label = 'S'
        else:
            latent_label = str(target)
        G_true = nx.relabel_nodes(G_full, {target: latent_label}, copy=True)

        draw_causal_graph(G_true, title=f"Ground Truth Graph ({source} | {lt})")
        
        example += 1
        G_obs = G_full.copy()
        if lt == '':
            print("Unbiased, hence not dropping the node")
        else:
            G_obs.remove_node(target)
        if lt == 'confounder':
            X_biased = X_raw.drop(columns=[target])
        elif lt == 'collider':
            clean_target = X_raw[target].replace([np.inf, -np.inf], np.nan).dropna()
            cutoff_val = 0.80 
            thresh = clean_target.quantile(cutoff_val)
            X_biased = X_raw[X_raw[target] > thresh].copy()
        else:
            X_biased = X_raw.copy()

        unique_labels = sorted(X_biased['Context'].unique())
        X_kci = X_biased.copy()
        X_kci['Context'] = X_kci['Context'].astype(int) - 1
        print(X_kci['Context'].value_counts())
        m = {i: col for i, col in enumerate(X_kci.columns) if col != 'Context'}
        parts, Gr, k_labs, marg_labs, ab_l, M_graph = run_multivariate_kci(X_kci, G_obs, samples=None, real_world=True)

        print()
        print("MECHANISMS")
        current_mechanisms = {}
        for node, shifts in parts.items():
            mech_parts = shifts['mechanism_shifts']
            current_mechanisms[node] = [sorted(list(group)) for group in mech_parts]
            total = len(mech_parts)
            merged = sum(1 for p in mech_parts if len(p) > 1)
            print(f"\nNODE {node: <4} | {total} Groups ({merged} Merged)")
            print("-" * 30)
            for i, group in enumerate(mech_parts):
                marker = "[*]" if len(group) > 1 else "[ ]"
                print(f"  {marker} Group {i+1: >2}: {sorted(group)}")
        current_preds = [r['type'].lower() for r in ab_l]
        diag_clusters = [r['nodes'] for r in ab_l]
        synch_mechs = [r['sync_mech_ami'] for r in ab_l]
        synch_margs = [r['sync_marg_ami'] for r in ab_l]
        spectral_subsets = [r['spectral_subsets'] for r in ab_l]
        coverage = [r['coverage'] for r in ab_l]
        if lt == '':
            target = ''
        # probe_posterior_mean = X_kci['probe'].mean()
        # probe_posterior_std = X_kci['probe'].std()
        if ab_l:
            report_row = {
                'dataset':source,
                'gt_type': lt,
                'target':target,
                'predicted_gt_type': current_preds[0] if current_preds else "none",
                'subset_nodes': str(diag_clusters),
                'sync_mech_ami': synch_mechs,
                # 'sync_marg_ami': synch_margs,
                'mechanism_groups': str(current_mechanisms),
                'spectral_subsets': spectral_subsets,
                'coverage': coverage,
                # 'probe_prior' : f"{probe_prior_mean} +/- {probe_prior_std}",
                # 'probe_posterior' : f"{probe_prior_mean} +/- {probe_prior_std}"
            }
        else:
            report_row = {
                'dataset':source,
                'gt_type': lt,
                'predicted_gt_type': current_preds[0] if current_preds else "none",
                'target':target,
                'subset_nodes': str(diag_clusters),
                'sync_mech_ami': synch_mechs,
                # 'sync_marg_ami': "",
                'mechanism_groups': "",
                'spectral_subsets': "",
                'coverage':coverage,
                # 'probe_prior' : f"{probe_prior_mean} +/- {probe_prior_std}",
                # 'probe_posterior' : f"{probe_prior_mean} +/- {probe_prior_std}"
                }
        
        df_row = pd.DataFrame([report_row])
        print(df_row)
        df_row.to_csv(csv_master, mode='a', header=not os.path.exists(csv_master), index=False)

        print('\n\n')
    print(f"\nExperiment Complete. Results: {csv_master}")