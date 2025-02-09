# Copyright 2025 Valeo.

"""Abstract observation wrapper for the simulator."""

from abc import ABC, abstractmethod

import jax
from waymax import datatypes
from waymax import env as waymax_env

from vmax.simulator.features import extractor
from vmax.simulator.wrappers import environment


class AbstractObservationWrapper(environment.Wrapper, ABC):
    """Abstract observation wrapper for the simulator."""

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        features_extractor: extractor.BaseFeaturesExtractor,
    ) -> None:
        """Initialize the abstract observation wrapper for the simulator.

        Args:
            env: The environment object.
            features_extractor: The features extractor.

        """
        super().__init__(env)
        self._features_extractor = features_extractor

    @property
    def features_extractor(self):
        """Return the features extractor."""
        return self._features_extractor

    def observation_spec(self, scenario: datatypes.SimulatorState) -> int:
        """Return the observation space size.

        Args:
            scenario: The current simulator state.

        Returns:
            The observation space size.

        """
        if scenario.batch_dims != ():
            scenario = jax.tree_util.tree_map(lambda x: x[scenario.batch_dims], scenario)

        obs = self.observe(scenario)

        return obs.shape[-1]

    @abstractmethod
    def observe(self, simulator_state: datatypes.SimulatorState) -> jax.Array:
        pass
