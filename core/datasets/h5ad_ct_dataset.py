from typing import TypedDict

import numpy as np
import torch
from scanpy import AnnData
from scipy import sparse
from torch.utils.data import Dataset


class CellTypeBatch(TypedDict):
    X: torch.Tensor
    y: torch.Tensor


class H5ADCTDataset(Dataset):
    def __init__(self, adata: AnnData, cell_type_column: str) -> None:
        x_features = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        self.X = torch.Tensor(x_features.astype(np.float32, copy=False))

        # The cell types
        ct_obs = adata.obs[cell_type_column]
        if hasattr(ct_obs, "cat"):
            categories = sorted(ct_obs.cat.categories)
        else:
            categories = sorted(np.asarray(ct_obs.dropna().unique()).tolist())
        ct_to_int = {ct: i for i, ct in enumerate(categories)}
        ct_to_int_vec = np.vectorize(ct_to_int.get)
        self.ct = torch.Tensor(ct_to_int_vec(np.array(ct_obs))).to(torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, index: int) -> CellTypeBatch:
        if index > len(self) or index < 0:
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")

        return {"X": self.X[index], "y": self.ct[index]}
