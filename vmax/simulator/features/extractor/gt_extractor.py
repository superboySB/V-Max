# Copyright 2025 Valeo.


"""Feature extractor for ground truth (GT) path target features."""

import jax
import jax.numpy as jnp
import matplotlib as mpl
from waymax import datatypes
from waymax.utils import geometry

from vmax.simulator import features, operations
from vmax.simulator.features import extractor


class GTFeaturesExtractor(extractor.BaseFeaturesExtractor):
    """Feature extractor for ground truth (GT) path target features."""

    def __init__(self, path_target_config: dict | None = None) -> None:
        """Initialize the GT features extractor.

        Args:
            path_target_config: Configuration for path target features.

        """
        super().__init__(path_target_config=path_target_config)

    def extract_features(self, state: datatypes.SimulatorState) -> jax.Array:
        """Extract target features from the simulator state.

        Args:
            state: The simulator state.

        Returns:
            The extracted path target features.

        """
        sdc_observation = self._get_sdc_observation(state)

        path_target_features = self._build_target_features(state, sdc_observation)

        return path_target_features.data

    def unflatten_features(self, vectorized_obs: jax.Array) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Unflatten a vectorized observation into path target features.

        Args:
            vectorized_obs: The vectorized observation.

        Returns:
            A tuple containing the path target features and None.

        """
        batch_dims = vectorized_obs.shape[-3:-1]
        flatten_size = vectorized_obs.shape[-1]
        unflatten_size = 0

        path_target_feature_size = self._get_features_size(self._path_target_features_key)

        path_target_size = self._num_target_path_points * path_target_feature_size
        path_target_features = vectorized_obs[..., unflatten_size : unflatten_size + path_target_size]
        path_target_features = path_target_features.reshape(
            *batch_dims,
            self._num_target_path_points,
            path_target_feature_size,
        )
        unflatten_size += path_target_size

        assert flatten_size == unflatten_size, f"Unflatten size {unflatten_size} does not match {flatten_size}"

        return path_target_features, None

    def plot_features(self, state: datatypes.SimulatorState, ax: mpl.axes.Axes) -> None:
        """Plot the path target features on a given axis.

        Args:
            state: The simulator state.
            ax: The matplotlib axes object.

        """
        sdc_observation = self._get_sdc_observation(state)

        path_target_features = self._build_target_features(state, sdc_observation)

        # 4. Plot path target
        path_target_features.plot(ax)

    def _build_target_features(self, simulator_state, sdc_obs):
        """Build the target features for the SDC based on simulation state and observation.

        Args:
            simulator_state: The current simulator state.
            sdc_obs: The SDC observation.

        Returns:
            An instance of PathTargetFeatures.

        """
        batch_dims = simulator_state.batch_dims
        len_dims = len(batch_dims)

        sdc_idx = operations.get_index(sdc_obs.is_ego)
        sdc_log_xy = jnp.take_along_axis(simulator_state.log_trajectory.xy, sdc_idx[None, None, None], axis=len_dims)
        sdc_pose2d = sdc_obs.pose2d.matrix

        sdc_log_xy = geometry.transform_points(pts=sdc_log_xy, pose_matrix=sdc_pose2d)

        target = jnp.full_like(sdc_log_xy, sdc_log_xy[:, -1])
        target = jnp.take(target, jnp.arange(self._num_target_path_points), axis=-2)

        sdc_log_xy = jax.lax.dynamic_slice_in_dim(
            sdc_log_xy,
            simulator_state.timestep,
            self._num_target_path_points,
            axis=-2,
        )
        target = jax.lax.dynamic_update_slice_in_dim(target, sdc_log_xy, 0, axis=-2).squeeze(0)
        target = extractor.normalize_path(target, self._max_meters)

        return features.PathTargetFeatures(xy=target)
