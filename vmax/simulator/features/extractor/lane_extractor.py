# Copyright 2025 Valeo.


"""Feature extractor for lane-specific features."""

import jax
from waymax import datatypes

from vmax.simulator.features import extractor


class LaneFeaturesExtractor(extractor.BaseFeaturesExtractor):
    """Feature extractor for lane-specific features."""

    def __init__(
        self,
        obs_past_num_steps: int = 1,
        objects_config: dict | None = None,
        roadgraphs_config: dict | None = None,
        traffic_lights_config: dict | None = None,
        path_target_config: dict | None = None,
    ) -> None:
        """Initialize the lane features extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features

        """
        super().__init__(
            obs_past_num_steps,
            objects_config,
            roadgraphs_config,
            traffic_lights_config,
            path_target_config,
        )
        self._dict_mapping["types"] = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _filter(self, roadgraph: datatypes.RoadgraphPoints) -> jax.Array:
        """Filter to retain only lane surface street points.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A boolean array where True indicates lane surface street points.

        """
        return roadgraph.types == datatypes.MapElementIds.LANE_SURFACE_STREET
