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

Each dataset must be preprocessed and spatially aligned before STFateFlow training or inference. The preprocessing notebooks prepare the expression and spatial inputs and compute the FUGWOT transport plans used for cross-time cell pairing.

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

The main Python interfaces are:

```python
from core.datasets.model_dataset import STFateFlowDataset, stfateflow_collate
from core.models.STFateFlow import STFateFlow_Module
from core.models.checkpointing import load_stfateflow_checkpoint
```

See `docs/fugwot_parameters_and_dataset_summary.md` for transport parameters and dataset statistics, and `docs/stfateflow_flow_matching_model.md` for the model formulation.

## Experimental Results

Precomputed experimental results and evaluation data are available from the following Google Drive folder:

[Download STFateFlow experimental results from Google Drive](https://drive.google.com/drive/folders/16EHm-zLZLWMB2Wc-8qOqZfrNCQSR-Q1X?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)

After downloading, place the result files under `experiments/results/` or update the paths in the evaluation notebooks accordingly. Large result files, datasets, and model checkpoints are intentionally excluded from the Git repository.

## Included Datasets

The current workflows cover ARTISTA axolotl regeneration, MOSTA mouse embryogenesis, and PRISTA4D regeneration data.
