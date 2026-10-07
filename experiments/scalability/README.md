# MOSTA scalability benchmark

This benchmark trains on MOSTA 9.5 and 11.5 and evaluates interpolation at the
held-out 10.5 time point. It compares:

- `stfateflow`: all cells in each downsampled training set are available to the
  FUGWOT preprocessing and the flow-matching training pipeline.
- `stvcr-default`: stVCR's published fixed training sample size of 1,000 cells
  per time point.
- `stvcr-all-cells`: stVCR's `num_samples` is set to the complete number of
  cells in each downsampled time point.

The default fractions are `0.125,0.25,0.5,1.0`. Sampling is deterministic,
cell-type stratified, and nested: cells in a smaller subset are retained in all
larger subsets for the same seed. The complete 10.5 time point is always used
as ground truth.

## Run

Run one configuration per process so that CUDA peak-memory measurements are
independent:

```bash
/data/yuz/envs/stfm/bin/python experiments/scalability/run_benchmark.py run \
  --method stvcr-default --fraction 0.125 --device cuda:0

/data/yuz/envs/stfm/bin/python experiments/scalability/run_benchmark.py run \
  --method stvcr-all-cells --fraction 0.125 --device cuda:0

/data/yuz/envs/nicheflow/bin/python experiments/scalability/run_benchmark.py run \
  --method stfateflow --fraction 0.125 --device cuda:0
```

Generate and optionally execute the full command grid:

```bash
/data/yuz/envs/nicheflow/bin/python experiments/scalability/run_grid.py
/data/yuz/envs/nicheflow/bin/python experiments/scalability/run_grid.py --execute --device cuda:0
```

The full-cell stVCR run uses an exact OT loss over all sampled cells and may
run out of memory. The launcher records non-zero exit codes; an OOM is a valid
scalability outcome and must not be replaced by the default 1,000-cell run.

Aggregate predictions, resource profiles, and failures:

```bash
/data/yuz/envs/nicheflow/bin/python experiments/scalability/run_benchmark.py evaluate
/data/yuz/envs/nicheflow/bin/python experiments/scalability/plot_results.py
```

## Outputs

Results are written under `experiments/scalability/results/`:

- `profiles/*.json`: phase-level wall time, peak process GPU memory, peak
  PyTorch memory, and process RAM.
- `predictions/*.h5ad`: cell-level expression and spatial predictions.
- `models/`: checkpoints and per-run preprocessing artifacts.
- `benchmark_metrics.csv`: prediction metrics on the fixed full 10.5 target.
- `benchmark_resources.csv`: flattened resource measurements.
- `benchmark_summary.csv`: metrics and resources joined by run ID.
- `grid_status.csv`: subprocess exit status for grid runs.
- `figures/mosta_scalability_resources.{pdf,png}`: preprocessing/alignment,
  training, end-to-end time, and peak GPU-memory curves.
- `figures/mosta_scalability_accuracy.{pdf,png}`: RNA MMD/W1 and spatial
  Chamfer/HD95 curves.

The primary RNA metrics use a shared ground-truth scale for Wasserstein and
Sinkhorn and a ground-truth-derived RBF bandwidth for MMD. Spatial metrics use
the original MOSTA coordinate frame. stVCR's centered/scaled output is mapped
back using the preprocessing scale and source centroid before evaluation.
