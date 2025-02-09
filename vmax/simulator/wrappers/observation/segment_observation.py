# Copyright 2025 Valeo.

"""Observation wrappers for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import env as waymax_env

from vmax.simulator.features import extractor
from vmax.simulator.wrappers import observation


class ObsSegmentWrapper(observation.AbstractObservationWrapper):
    """Observation wrapper that returns a segment global vector observation."""

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        obs_past_num_steps: int = 1,
        objects_config: dict | None = None,
        roadgraphs_config: dict | None = None,
        traffic_lights_config: dict | None = None,
        path_target_config: dict | None = None,
    ) -> None:
        """Initialize the observation wrapper.

        Args:
            env: Environment to wrap.
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for objects.
            roadgraphs_config: Configuration for roadgraphs.
            traffic_lights_config: Configuration for traffic lights.
            path_target_config: Configuration for the path target.

        """
        super().__init__(
            env,
            extractor.SegmentFeaturesExtractor(
                obs_past_num_steps=obs_past_num_steps,
                objects_config=objects_config,
                roadgraphs_config=roadgraphs_config,
                traffic_lights_config=traffic_lights_config,
                path_target_config=path_target_config,
            ),
        )

    def observe(self, simulator_state: datatypes.SimulatorState) -> jax.Array:
        """Return the observation for the agent.

        Args:
            simulator_state: Current simulator state.

        Returns:
            The concatenated observation array.

        """
        sdc_traj_matrix, other_traj_matrix, rg_matrix, tl_matrix, gps_path_matrix = (
            self._features_extractor.extract_features(simulator_state)
        )

        # Features reshape by batch dimension - and flatten
        sdc_traj_matrix = jnp.reshape(sdc_traj_matrix, (*simulator_state.batch_dims, -1))
        other_traj_matrix = jnp.reshape(other_traj_matrix, (*simulator_state.batch_dims, -1))
        rg_matrix = jnp.reshape(rg_matrix, (*simulator_state.batch_dims, -1))
        tl_matrix = jnp.reshape(tl_matrix, (*simulator_state.batch_dims, -1))
        gps_path_matrix = jnp.reshape(gps_path_matrix, (*simulator_state.batch_dims, -1))

        # Flatten all the matrices - Vec Obs
        return jnp.concatenate(
            [sdc_traj_matrix, other_traj_matrix, rg_matrix, tl_matrix, gps_path_matrix],
            axis=len(simulator_state.batch_dims),
        )
