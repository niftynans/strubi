import pandas as pd
import networkx as nx
import ast
import numpy as np

def calculate_sachs_union_coverage(csv_path='sachs.csv'):
    edges = [
        ('Plcg', 'PIP2'), ('Plcg', 'PKA'), ('PIP2', 'PKC'), ('PKC', 'Raf'), ('PKC', 'PKA'), 
        ('PKC', 'P38'), ('PKC', 'Jnk'), ('PKA', 'Erk'), ('Raf', 'Mek'), ('Mek', 'Erk'), 
        ('PIP3', 'Akt'), ('PIP3', 'Mek'), ('PIP3', 'P38'), ('PIP3', 'PKA'), ('PIP3', 'Jnk'), 
        ('Erk', 'P38'), ('Erk', 'Akt'), ('Erk', 'PIP2')
    ]
    G_known = nx.DiGraph(edges)

    def get_coverage_for_set(node_list, graph):
        valid_nodes = [n for n in node_list if n in graph.nodes()]
        if not valid_nodes:
            return 0.0

        max_ancestors = -1
        anchor_node = None
        
        for node in valid_nodes:
            ancestors = nx.ancestors(graph, node)
            overlap = len(set(ancestors).intersection(set(valid_nodes)))
            
            if overlap > max_ancestors:
                max_ancestors = overlap
                anchor_node = node
        
        if anchor_node is None:
            return 0.0
            
        true_ancestors = nx.ancestors(graph, anchor_node)
        if len(true_ancestors) > 0:
            return len(set(valid_nodes).intersection(true_ancestors)) / len(true_ancestors)
        return 0.0

    df = pd.read_csv(csv_path)
    final_results = []
    
    for idx, row in df.iterrows():
        try:
            subsets = ast.literal_eval(row['subset_nodes'])
        except (ValueError, SyntaxError):
            continue
            
        if subsets:
            
            union_nodes = list(set().union(*subsets))
            row_coverage = get_coverage_for_set(union_nodes, G_known)
            print(f"Row {row['target'], row['gt_type']} - Subsets: {subsets}, Union Nodes: {union_nodes}, Coverage: {row_coverage:.2f}")
        else:
            row_coverage = 0.0
        
        final_results.append({
            'target': row['target'],
            'type': row['gt_type'],
            'coverage': row_coverage
        })

    results_df = pd.DataFrame(final_results)
    
    # Summary by type
    print("\n--- Mean Coverage by Test Type (Union-Based) ---")
    print(results_df.groupby('type')['coverage'].mean())
    df['joint_coverage'] = results_df['coverage']
    df.loc[df['joint_coverage'] > 0.60, 'predicted_gt_type'] = 'collider'
    df.to_csv(csv_path)
    
    return results_df

# Execute
results = calculate_sachs_union_coverage('multivariate/results/real_world/sachs.csv')
print(results['coverage'])