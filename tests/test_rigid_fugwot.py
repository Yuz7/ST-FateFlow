from __future__ import annotations

import anndata as ad
import numpy as np

from core.preprocessing.rigid_fugwot import rigid_register_time_series, weighted_procrustes


def test_weighted_procrustes_recovers_2d_rigid_transform() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    theta = 0.7
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    target = (source - np.array([2.0, -1.0])) @ rotation.T
    plan = np.eye(len(source)) / len(source)
    aligned, fitted_rotation, _ = weighted_procrustes(source, target, plan)
    np.testing.assert_allclose(aligned, source, atol=1e-7)
    assert np.linalg.det(fitted_rotation) > 0


def test_time_series_registration_uses_plan_and_preserves_input() -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    theta = 0.4
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    moved = (base - np.array([3.0, 2.0])) @ rotation.T
    data = ad.AnnData(X=np.ones((8, 3), dtype=np.float32))
    data.obs["time"] = [0.0] * 4 + [1.0] * 4
    data.obsm["spatial"] = np.concatenate([base, moved]).astype(np.float32)

    def identity_plan(source, target, spatial_key, joint_attr_key, config):
        del spatial_key, joint_attr_key, config
        assert source.n_obs == target.n_obs == 4
        return np.eye(4) / 4

    result = rigid_register_time_series(
        data,
        normalize_coordinates=False,
        downsample_total=None,
        plan_solver=identity_plan,
    )
    aligned = result.adata.obsm["X_spatial_input"]
    np.testing.assert_allclose(aligned[:4], aligned[4:], atol=1e-6)
    np.testing.assert_allclose(data.obsm["spatial"][:4], base, atol=1e-7)
    assert (0.0, 1.0) in result.plans
