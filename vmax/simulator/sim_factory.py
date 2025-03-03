# Copyright 2025 Valeo.


"""Factory functions for creating environments and data generators."""

from collections.abc import Iterator

from waymax import config as _config
from waymax import dataloader, datatypes, dynamics, env
from waymax import metrics as waymax_metrics

from vmax.simulator import metrics
from vmax.simulator.wrappers import environment, observation


def make_data_generator(
    path: str,
    max_num_objects: int = 128,
    include_sdc_paths: bool = True,
    seed: int | None = None,
    batch_dims: tuple = (),
    distributed: bool = False,
    repeat: int | None = None,
) -> Iterator[datatypes.SimulatorState]:
    """Create a data generator for simulator states.

    Args:
        path: Path to the dataset.
        max_num_objects: Maximum number of objects.
        include_sdc_paths: Whether to include SDC paths.
        seed: Seed for shuffling.
        batch_dims: Batch dimensions.
        distributed: Whether to distribute the dataset.
        repeat: Number of repetitions (None for infinite).

    Returns:
        A generator yielding simulator states.

    """
    return dataloader.simulator_state_generator(
        _config.DatasetConfig(
            path=path,
            max_num_objects=max_num_objects,
            batch_dims=batch_dims,
            distributed=distributed,
            shuffle_seed=seed,
            repeat=repeat,
            include_sdc_paths=include_sdc_paths,
            num_paths=10,
            num_points_per_path=300,
        ),
    )


def make_env(
    max_num_objects: int = 128,
    dynamics_model: dynamics.DynamicsModel | None = None,
    observation_type: str = "gt",
    observation_config: dict | None = None,
    reward_type: str = "linear",
    reward_config: dict | list = ("offroad", "overlap"),
    noisy_init: bool = False,
) -> env.PlanningAgentEnvironment:
    """Create a planning agent environment.

    Args:
        max_num_objects: Maximum number of objects.
        dynamics_model: Dynamics model to use.
        observation_type: Type of observation.
        observation_config: Configuration for observations.
        reward_type: Type of reward.
        reward_config: Configuration for rewards.
        noisy_init: Whether to add initialization noise.

    Returns:
        A PlanningAgentEnvironment instance.

    """
    # Standard Waymax metrics
    metrics_to_run = ["log_divergence", "overlap", "offroad"]
    # Add SDC Waymax metrics
    metrics_to_run += ["sdc_wrongway", "sdc_progression", "sdc_off_route"]
    # Add SDC V-Max metrics
    metrics_to_run += metrics._VMAX_METRICS_REGISTRY.keys()

    init_steps = 1 if noisy_init else 11

    # Register metrics
    for key in metrics._VMAX_METRICS_REGISTRY:
        waymax_metrics.register_metric(key, metrics.get_metrics(key))

    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        metrics=_config.MetricsConfig(metrics_to_run=metrics_to_run),
        init_steps=init_steps,
    )
    waymax_env = env.PlanningAgentEnvironment(dynamics_model, env_config)
    waymax_env = _add_observation_wrapper(waymax_env, observation_type, observation_config)
    waymax_env = _add_reward_wrapper(waymax_env, reward_type, reward_config)

    return waymax_env


def make_env_for_training(
    max_num_objects: int = 128,
    dynamics_model: dynamics.DynamicsModel | None = None,
    sdc_paths_from_data: bool = True,
    observation_type: str = "gt",
    observation_config: dict | None = None,
    reward_type: str = "linear",
    reward_config: dict | list = ("offroad", "overlap"),
    termination_keys: list[str] = ("offroad", "overlap"),
) -> env.PlanningAgentEnvironment:
    """Create an environment configured for training.

    Args:
        max_num_objects: Maximum number of objects.
        dynamics_model: Dynamics model to use.
        sdc_paths_from_data: Whether to include SDC paths.
        observation_type: Type of observation.
        observation_config: Configuration for observations.
        reward_type: Type of reward.
        reward_config: Configuration for rewards.
        termination_keys: Termination keys.

    Returns:
        A PlanningAgentEnvironment instance configured for training.

    """
    waymax_env = make_env(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics_model,
        observation_type=observation_type,
        observation_config=observation_config,
        reward_type=reward_type,
        reward_config=reward_config,
    )
    waymax_env = environment.BraxWrapper(waymax_env, termination_keys)
    if not sdc_paths_from_data:
        waymax_env = environment.SDCPathWrapper(waymax_env)
    waymax_env = environment.VmapWrapper(waymax_env)
    waymax_env = environment.AutoResetWrapper(waymax_env)

    return waymax_env


def make_env_for_evaluation(
    max_num_objects: int = 128,
    dynamics_model: dynamics.DynamicsModel | None = None,
    sdc_paths_from_data: bool = True,
    observation_type: str = "gt",
    observation_config: dict | None = None,
    reward_type: str = "",
    reward_config: dict | None = None,
    termination_keys: list[str] = ("offroad", "overlap"),
    noisy_init: bool = False,
) -> env.PlanningAgentEnvironment:
    """Create an environment configured for evaluation.

    Args:
        max_num_objects: Maximum number of objects.
        dynamics_model: Dynamics model to use.
        sdc_paths_from_data: Whether to include SDC paths.
        observation_type: Type of observation.
        observation_config: Configuration for observations.
        reward_type: Type of reward.
        reward_config: Configuration for rewards.
        termination_keys: Termination keys.
        noisy_init: Whether to add initialization noise.

    Returns:
        A PlanningAgentEnvironment instance configured for evaluation.

    """
    waymax_env = make_env(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics_model,
        observation_type=observation_type,
        observation_config=observation_config,
        reward_type=reward_type,
        reward_config=reward_config,
        noisy_init=noisy_init,
    )
    waymax_env = environment.BraxWrapper(waymax_env, termination_keys)
    if not sdc_paths_from_data:
        waymax_env = environment.SDCPathWrapper(waymax_env)
    if noisy_init:
        waymax_env = environment.NoisyInitWrapper(waymax_env)
    waymax_env = environment.VmapWrapper(waymax_env)

    return waymax_env


def _add_observation_wrapper(
    env: env.PlanningAgentEnvironment,
    type: str,
    config: dict,
) -> env.PlanningAgentEnvironment:
    """Wrap the environment with the specified observation wrapper.

    Args:
        env: Environment to wrap.
        type: Observation type.
        config: Observation configuration.

    Returns:
        The wrapped environment.

    """
    if type == "base":
        env = observation.ObsBaseWrapper(
            env,
            obs_past_num_steps=config["obs_past_num_steps"],
            objects_config=config["objects"],
            roadgraphs_config=config["roadgraphs"],
            traffic_lights_config=config["traffic_lights"],
            path_target_config=config["path_target"],
        )
    elif type == "segment":
        env = observation.ObsSegmentWrapper(
            env,
            obs_past_num_steps=config["obs_past_num_steps"],
            objects_config=config["objects"],
            roadgraphs_config=config["roadgraphs"],
            traffic_lights_config=config["traffic_lights"],
            path_target_config=config["path_target"],
        )
    elif type == "lane":
        env = observation.ObsLaneWrapper(
            env,
            obs_past_num_steps=config["obs_past_num_steps"],
            objects_config=config["objects"],
            roadgraphs_config=config["roadgraphs"],
            traffic_lights_config=config["traffic_lights"],
            path_target_config=config["path_target"],
        )
    elif type == "road":
        env = observation.ObsRoadWrapper(
            env,
            obs_past_num_steps=config["obs_past_num_steps"],
            objects_config=config["objects"],
            roadgraphs_config=config["roadgraphs"],
            traffic_lights_config=config["traffic_lights"],
            path_target_config=config["path_target"],
        )
    elif type == "gt":
        env = observation.ObsGTWrapper(env, path_target_config=config["path_target"])
    else:
        raise ValueError(f"Unknown observation type: {type}")

    return env


def _add_reward_wrapper(
    env: env.PlanningAgentEnvironment,
    type: str,
    reward_config: dict,
) -> env.PlanningAgentEnvironment:
    """Wrap the environment with the specified reward wrapper.

    Args:
        env: Environment to wrap.
        type: Reward type.
        reward_config: Reward configuration.

    Returns:
        The wrapped environment.

    """
    if type == "linear":
        env = environment.RewardLinearWrapper(env, reward_config)
    elif type == "custom":
        env = environment.RewardCustomWrapper(env)

    return env
