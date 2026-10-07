# STFateFlow

STFateFlow is a flow-matching-based method for learning gene-expression and spatial dynamics from time-resolved spatial transcriptomics data.

## Project Layout

- `core/models/STFateFlow.py`: main flow-matching model
- `core/models/backbones/`: point-cloud transformer backbone
- `core/datasets/model_dataset.py`: timepoint datasets and FUGWOT sampling
- `core/utils/fgwot.py`: cached transport-plan sampler
- `notebooks/`: preprocessing, training, inference, and potential analysis
- `experiments/`: baselines, ablations, and evaluation scripts
- `docs/`: model and dataset documentation

## Requirements

The project is designed for Python 3.11 and a CUDA-enabled PyTorch environment. Its main dependencies include PyTorch, PyTorch Geometric, Scanpy, AnnData, moscot/OTT-JAX, NumPy, SciPy, pandas, and Matplotlib.

## Quick Start

Each dataset must be preprocessed and spatially aligned before STFateFlow training or inference. STFateFlow now includes an optional rigid-registration stage adapted from stVCR: FUGWOT replaces stVCR's balanced cross-slice OT, and the resulting coupling weights a generalized Procrustes rotation/translation. The preprocessing notebooks then compute full-resolution FUGWOT transport plans on the registered coordinates for cross-time flow-matching pairs.

Run the preprocessing notebook for the selected dataset first, then run its corresponding inference notebook:

| Dataset | Preprocessing and alignment | Training and inference |
|---|---|---|
| ARTISTA axolotl regeneration | `notebooks/preprocess_axolotl.ipynb` | `notebooks/axolotl_infer.ipynb` |
| MOSTA mouse embryogenesis | `notebooks/preprocess_mosta.ipynb` | `notebooks/mosta_infer.ipynb` |
| PRISTA4D regeneration | `notebooks/preprocess_flysta4d.ipynb` | `notebooks/prista4d_infer.ipynb` |

Update the input-data and output paths in both notebooks when necessary. For example, the ARTISTA workflow is:

```bash
cd /data/yuz/spatialtranscriptomics/spatiotemporal/spatiotemporal_codes/STFateFlow
jupyter lab notebooks/preprocess_axolotl.ipynb
# Run all preprocessing and FUGWOT alignment cells before continuing.
jupyter lab notebooks/axolotl_infer.ipynb
```

The generated FUGWOT cache must match the dataset, selected genes, spatial coordinates, and time points used by the inference notebook.

### Optional FUGWOT-weighted rigid registration

Use the command below when input slices have not already been placed in a
common global coordinate frame:

```bash
python scripts/rigid_fugwot_preprocess.py input.h5ad aligned.h5ad \
  --time-key time \
  --spatial-key spatial \
  --output-spatial-key X_spatial_input \
  --downsample-total 5000 \
  --alpha 0.5 --rank 200 --tau-a 0.97 --tau-b 0.93 --device gpu
```

For features already stored in `adata.obsm`, add for example
`--joint-attr-key X_gene_input`. By default the registration coupling uses
`adata.X`. The first time point is centered; each subsequent time point is
rigidly aligned to the preceding aligned slice. Reflections are disabled by
default and can be enabled explicitly with `--allow-reflection`.

The command writes registered coordinates, transform metadata, and the
downsampled couplings used to estimate the rigid transforms. These couplings
are audit artifacts only. Run the dataset preprocessing notebook afterward to
compute full-resolution FUGWOT plans on `X_spatial_input`; do not use the
downsampled registration plans for flow training.

The main Python interfaces are:

```python
from core.datasets.model_dataset import STFateFlowDataset, stfateflow_collate
from core.models.STFateFlow import STFateFlow_Module
from core.models.checkpointing import load_stfateflow_checkpoint
from core.preprocessing import FUGWOTConfig, rigid_register_time_series
from core.training import EarlyStoppingConfig, train_stfateflow
```

Training defaults to a maximum of 1,000 optimizer steps. The reusable trainer
supports leakage-free early stopping based on the rolling stochastic
flow-matching objective:

```python
result = train_stfateflow(
    stfateflow,
    train_loader,
    optimizer,
    device=device,
    max_steps=1000,
    early_stopping=EarlyStoppingConfig(),
)
print(result.summary())
```

The default policy starts checking after 500 steps, checks every 50 steps,
smooths over the latest 100 losses, and stops after four checks without at
least 0.5% relative improvement. It restores the best checked model. The held-out
interpolation time point is never used by this criterion. Set
`early_stopping=None` to disable it and run exactly 1,000 steps.

See `docs/fugwot_parameters_and_dataset_summary.md` for transport parameters and dataset statistics, and `docs/stfateflow_flow_matching_model.md` for the model formulation.

## Experimental Results

Precomputed experimental results and evaluation data are available from the following Google Drive folder:

[Download STFateFlow experimental results from Google Drive](https://drive.google.com/drive/folders/16EHm-zLZLWMB2Wc-8qOqZfrNCQSR-Q1X?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)

After downloading, place the result files under `experiments/results/` or update the paths in the evaluation notebooks accordingly. Large result files, datasets, and model checkpoints are intentionally excluded from the Git repository.

## Included Datasets

The current workflows cover ARTISTA axolotl regeneration, MOSTA mouse embryogenesis, and PRISTA4D regeneration data.
