import pandas as pd

df = pd.read_csv('multivariate/results/ablation_main.csv')
df_clean = df[df['runtime'] != 'runtime'].copy()
df_clean['runtime'] = pd.to_numeric(df_clean['runtime'])
df_clean['n_nodes'] = pd.to_numeric(df_clean['n_nodes'])

runtime_stats = df_clean.groupby(['oracle', 'n_nodes'])['runtime'].agg(['mean', 'std']).reset_index()
runtime_stats = runtime_stats.sort_values(by=['oracle', 'n_nodes'])

print("Runtime Statistics by Oracle and n_nodes:")
print(runtime_stats.to_string(index=False))

runtime_stats.to_csv('multivariate/results/runtime_summary.csv', index=False)