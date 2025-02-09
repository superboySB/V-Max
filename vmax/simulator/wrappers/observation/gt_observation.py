# Copyright 2025 Valeo.


"""Observation wrappers for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import env as waymax_env

from vmax.simulator.features import extractor
from vmax.simulator.wrappers import observation


class ObsGTWrapper(observation.AbstractObservationWrapper):
    """Observation wrapper that returns the ground truth trajectory."""

    def __init__(self, env: waymax_env.PlanningAgentEnvironment, path_target_config: dict | None = None) -> None:
        """Initialize the GT observation wrapper.

        Args:
            env: Environment to wrap.
            path_target_config: Configuration for the path target.

        """
        super().__init__(env, extractor.GTFeaturesExtractor(path_target_config=path_target_config))

    def observe(self, simulator_state: datatypes.SimulatorState) -> jax.Array:
        """Return the ground truth observation for the agent.

        Args:
            simulator_state: Current simulator state.

        Returns:
            The reshaped ground truth observation.

        """
        gt_matrix = self._features_extractor.extract_features(simulator_state)

        batch_dims = simulator_state.batch_dims

        return jnp.reshape(gt_matrix, (*batch_dims, -1))
