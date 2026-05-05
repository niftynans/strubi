# Reproduction Steps for StruBI

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Synthetic Data Ablations
```bash
# Run StruBI Synthetic Data Ablations
python multivariate/ablation.py
cp -r datasets competitors/coco/experiments
cp -r datasets competitors/latent-selection

# Run Synthetic Data for LS Variants
python competitors/latent-selection/synthetic_run.py 
mv baseline_latent_selection_context.csv multivariate/results/
mv baseline_latent_selection_pooled.csv multivariate/results/

# Run Synthetic Data for CoCo and FCI Variants
python competitors/coco/experiments/run_coco_fci_synthetic.py

# Process the ablations to see clear results
python multivariate/process_ablations_full.py
# All results are visible in the "csv_for_figs" folder.
```

### 3. Real-World Data (Sachs)
```bash
# Run StruBI on Sachs Data
python multivariate/real_world_workflow.py
python multivariate/joint_cov.py

# Run LS variants on Sachs Data
python competitors/latent-selection/sachs_run.py

# Run CoCo and FCI variants on Sachs Data
python competitors/coco/experiments/reproduce_sachs_results.py
```
