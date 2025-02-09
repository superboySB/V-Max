# Copyright 2025 Valeo.

"""Module for visualization of the simulator."""

import dataclasses
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax.visualization import utils as waymax_utils
from waymax.visualization import viz as waymax_viz

from vmax.simulator import waymax_overrides


def plot_input_agent(
    state: datatypes.SimulatorState,
    env: Any,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = True,
) -> np.ndarray:
    """Plot np array image for SimulatorState.

    Args:
        state: A SimulatorState instance.
        env: The environment instance.
        viz_config: dict for optional config.
        batch_idx: optional batch index.
        highlight_obj: Represents the type of objects that will be highlighted with
            `color.COLOR_DICT['controlled']` color.
        plot_sdc_paths: Flag to plot SDC paths.

    Returns:
        np image.

    """
    if batch_idx > -1:
        if len(state.shape) != 1:
            raise ValueError(f"Expecting one batch dimension, got {len(state.shape)}")
        state = waymax_viz._index_pytree(state, batch_idx)
    if state.shape:
        raise ValueError(f"Expecting 0 batch dimension, got {len(state.shape)}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)

    dict_viz_config = dataclasses.asdict(viz_config)

    features_extractor = env.get_wrapper_attr("features_extractor")
    sdc_observation = features_extractor._get_sdc_observation(state)

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 8))
    fig.set_facecolor("white")

    # First ax: SimulatorState
    waymax_overrides.plot_simulator_state(
        state=state,
        use_log_traj=False,
        viz_config=dict_viz_config,
        batch_idx=batch_idx,
        highlight_obj=highlight_obj,
        plot_sdc_paths=plot_sdc_paths,
        ax=axs[0],
    )

    axs[0].set_aspect("equal", "box")

    # Second ax: Observation
    waymax_overrides.plot_observation(
        obs=sdc_observation,
        viz_config=dict_viz_config,
        batch_idx=batch_idx,
        highlight_obj=highlight_obj,
        plot_sdc_paths=plot_sdc_paths,
        ax=axs[1],
    )
    axs[1].set_aspect("equal", "box")

    # Third ax: Features
    features_extractor.plot_features(state, axs[2])
    axs[2].axis((-1, +1, -1, +1))
    axs[2].set_aspect("equal", "box")

    return waymax_utils.img_from_fig(fig)
