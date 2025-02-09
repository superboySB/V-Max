# Copyright 2025 Valeo.


"""Feature extractor for road-based features."""

import jax
from waymax import datatypes

from vmax.simulator.features import extractor


class RoadFeaturesExtractor(extractor.BaseFeaturesExtractor):
    """Feature extractor for road-based features."""

    def __init__(
        self,
        obs_past_num_steps: int = 1,
        objects_config: dict | None = None,
        roadgraphs_config: dict | None = None,
        traffic_lights_config: dict | None = None,
        path_target_config: dict | None = None,
    ) -> None:
        """Initialize the road features extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features.

        """
        super().__init__(
            obs_past_num_steps,
            objects_config,
            roadgraphs_config,
            traffic_lights_config,
            path_target_config,
        )

        self._dict_mapping["types"] = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 0)

    def _filter(self, roadgraph: datatypes.RoadgraphPoints) -> jax.Array:
        """Determine lane center points based on road element types.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A boolean array marking the indices of the lane center points.

        """
        return (
            (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
            | (roadgraph.types == datatypes.MapElementIds.ROAD_EDGE_MEDIAN)
        )
