# Copyright 2025 Valeo.


"""Utility functions for evaluation script."""

import os
import pickle
import re
from functools import partial
from pathlib import Path
from typing import Any

import jax
import mediapy
import numpy as np
import pandas as pd
import yaml
from etils import epath
from waymax import dynamics

from vmax.simulator import make_env_for_evaluation, vizualization, waymax_overrides
from vmax.simulator.metrics.aggregators import nuplan_aggregate_score, vmax_aggregate_score
from vmax.simulator.metrics.collector import _metrics_operands
from vmax.simulator.wrappers import action


def setup_evaluation(dummy_scenario, eval_args, sdc_paths_from_data=True):
    """Create the evaluation environment and return required components.

    Args:
        dummy_scenario: A dummy scenario for environment initialization.
        eval_args: Parsed evaluation arguments.
        sdc_paths_from_data: Flag indicating whether to include SDC paths from dataset.

    Returns:
        A tuple containing the environment, step function, evaluation path, and termination keys.

    """

    if eval_args.sdc_actor == "ai":
        run_path = f"runs/{eval_args.path_model}/"
        model_path, model_name = get_model_path(run_path + "model/")

        eval_path = f"{eval_args.eval_name}/ai/{eval_args.path_dataset}/{run_path.replace('runs/', '')}{model_name.replace('.pkl', '')}/"  # noqa: E501

        # Training config
        with open(run_path + ".hydra/config.yaml", "r") as f:  # noqa: UP015
            eval_config = yaml.safe_load(f)

        eval_config["observation_config"]["max_num_objects"] = eval_args.max_num_objects
        eval_config["encoder"] = eval_config["network"]["encoder"]
        eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
        eval_config["value"] = eval_config["algorithm"]["network"]["value"]
        eval_config["unflatten_config"] = eval_config["observation_config"]
        eval_config["action_distribution"] = eval_config["algorithm"]["network"]["action_distribution"]

        termination_keys = eval_config["termination_keys"]

        env = make_env_for_evaluation(
            max_num_objects=eval_args.max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            observation_type=eval_config["observation_type"],
            observation_config=eval_config["observation_config"],
            termination_keys=termination_keys,
            noisy_init=eval_args.noisy_init,
        )
        policy = load_model(
            env,
            dummy_scenario,
            eval_args.algorithm,
            eval_config,
            model_path,
        )
        step_fn = make_step_fn(env, eval_args.sdc_actor, policy)
    else:
        eval_path = f"{eval_args.eval_name}/rule_based/{eval_args.sdc_actor}/{eval_args.path_dataset}/"
        eval_config = {"max_num_objects": eval_args.max_num_objects}
        termination_keys = ["overlap", "offroad", "run_red_light"]

        env = make_env_for_evaluation(
            max_num_objects=eval_args.max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            termination_keys=termination_keys,
            noisy_init=eval_args.noisy_init,
        )
        step_fn = make_step_fn(env, eval_args.sdc_actor)

    if eval_args.noisy_init:
        eval_path += "noisy_init/"
    os.makedirs(eval_path, exist_ok=True)

    return env, step_fn, eval_path, termination_keys


def load_model(env, dummy_scenario, algorithm, config, model_path):
    """Load and return the model for the specified learning algorithm.

    Args:
        env: The simulation environment.
        dummy_scenario: Scenario used for environment initialization.
        algorithm: Learning algorithm identifier.
        config: Configuration for model and training.
        model_path: Path to the model file.

    Returns:
        The inference policy created with the loaded parameters.

    """
    obs_size = env.observation_spec(dummy_scenario)
    action_size = 2  # env.action_spec().data.shape[0]

    if algorithm == "sac":
        from vmax.learning.algorithms.rl.sac.sac_factory import make_inference_fn
        from vmax.learning.algorithms.rl.sac.sac_factory import make_networks as build_network
    elif algorithm == "bc":
        from vmax.learning.algorithms.imitation.bc.bc_factory import make_inference_fn
        from vmax.learning.algorithms.imitation.bc.bc_factory import make_networks as build_network
    elif algorithm == "ppo":
        from vmax.learning.algorithms.rl.ppo.ppo_factory import make_inference_fn
        from vmax.learning.algorithms.rl.ppo.ppo_factory import make_networks as build_network
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    if algorithm in ["sac", "bc", "ppo"]:
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
            learning_rate=config["algorithm"]["learning_rate"],
            network_config=config,
        )
    make_policy = make_inference_fn(network)
    params = load_params(model_path)

    if algorithm == "bc":
        policy_params = params.policy
        policy = make_policy(policy_params, deterministic=True)
        return policy
    else:
        policy_params = params.policy
        policy = make_policy(policy_params, deterministic=True)

        return policy


def make_step_fn(env, type_step: str, policy_fn=None) -> callable:
    """Return a step function based on the provided action type.

    Args:
        env: The simulation environment.
        type_step: The type of step function to use.
        policy_fn: Optional policy function for action selection.

    Returns:
        A partially applied step function.

    """
    if type_step == "ai":
        return partial(action.policy_step, env=env, policy_fn=policy_fn)
    elif type_step == "expert":
        return partial(action.expert_step, env=env)
    elif type_step == "random":
        return partial(action.random_step, env=env)
    elif type_step == "constant":
        return partial(action.constant_step, env=env)
    else:
        raise ValueError(f"Invalid type step: {type_step}")


# Evaluate Utils
def load_params(path: str) -> Any:
    """Load serialized parameters from the specified file.

    Args:
        path: File path to the serialized model parameters.

    Returns:
        Loaded model parameters.

    """
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def get_model_path(model_path: str) -> tuple[str, str] | None:
    """Identify and return the latest model file path and its name.

    Args:
        model_path: Directory path containing model files.

    Returns:
        A tuple of the model file path and name if found; otherwise, None.

    """
    # Filter to get only files with .pkl extension
    pkl_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]
    if "model_final.pkl" in pkl_files:
        model_name = "model_final.pkl"
    else:
        pkl_files = [f for f in sorted(pkl_files, key=lambda f: int(re.findall(r"\d+", f)[0]))]
        model_name = pkl_files[-1] if pkl_files else None

    if model_name:
        print("Model name: ", model_name)
        return model_path + model_name, model_name
    else:
        print("No .pkl files found in the directory")
        return None


def list_folders_in_directory(directory: str, algorithm_name: str) -> list:
    """List folders matching the naming pattern for the given algorithm.

    Args:
        directory: Path of the directory to search.
        algorithm_name: The algorithm name filter.

    Returns:
        List of matching folder names.

    """
    regex = re.compile(rf"{algorithm_name}_.*")

    folders = [item for item in os.listdir(directory) if Path(directory, item).is_dir() and regex.match(item)]
    folders_with_ctime = [(folder, os.path.getctime(Path(directory, folder))) for folder in folders]
    sorted_folders = sorted(folders_with_ctime, key=lambda x: x[1])

    return [folder[0] for folder in sorted_folders]


def choose_folder(folders):
    """Prompt the user to select a folder from a list.

    Args:
        folders: List of folder names.

    Returns:
        The chosen folder name or the last folder if input is invalid.

    """
    for i, folder in enumerate(folders, start=1):
        print(f"[{i}] {folder}")

    try:
        choice = input(
            f"-> Enter the number of the folder you choose (default is {len(folders)}): ",
        )
        choice = int(choice) if choice else len(folders)
        return folders[choice - 1]
    except (ValueError, IndexError):
        print("Using the last folder as default.")
        return folders[-1]


def write_video(run_path, episode_images, idx, is_noisy=False):
    """Write and save a video for a given evaluation episode.

    Args:
        run_path: Base directory for saving the video.
        episode_images: List of images for the episode.
        idx: Episode index.
        is_noisy: Flag for noisy videos.

    """
    video_path = run_path + "/mp4/" + ("noisy/" if is_noisy else "")

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    mediapy.write_video(video_path + "eval_" + str(idx) + ".mp4", episode_images, fps=10)


def write_generator_result(save_path, eval_results):
    """Save evaluation results and metrics to CSV and text files.

    Args:
        save_path: Directory path to save results.
        eval_results: A tuple containing the number of scenarios and a metrics dictionary.

    """
    num_scenarios, metrics = eval_results
    df = pd.DataFrame(metrics)
    df.index.name = "scenario_index"
    df.to_csv(save_path + "/evaluation_episodes.csv", index=True)

    metrics = {
        key: np.mean([val.item() if isinstance(val, np.ndarray) else val for val in value])
        for key, value in metrics.items()
    }
    metrics.update(
        {"accuracy_only_at_fault": metrics["accuracy"] + (metrics["overlap"] - metrics["at_fault_collision"])},
    )

    # Determine the width for formatting
    key_width = max(len(key) for key in metrics)
    value_width = 10

    with open(save_path + "/evaluation_results.txt", "w") as txtfile:
        txtfile.write("=" * 50 + "\n")
        txtfile.write(f"Evaluation - {num_scenarios} scenarios :\n")
        txtfile.write("=" * 50 + "\n")
        txtfile.write(f"{'Metric':<{key_width}} {'Value':>{value_width}}\n")
        txtfile.write("-" * (key_width + value_width + 1) + "\n")
        for key, value in metrics.items():
            txtfile.write(f"{key:<{key_width}} {value:>{value_width}.5f}\n")


def plot_scene(
    env,
    env_transition,
    sdc_pov,
):
    """Generate and return a plot image of the simulation state.

    Args:
        env: The simulation environment.
        env_transition: The current environment transition state.
        sdc_pov: Boolean flag for SDC point-of-view.

    Returns:
        Plot image of the scene.

    """
    if sdc_pov:
        img = vizualization.plot_input_agent(jax.tree_util.tree_map(lambda x: x[0], env_transition.state), env)
    else:
        img = waymax_overrides.plot_simulator_state(
            env_transition.state,
            use_log_traj=False,
            batch_idx=0,
        )
    return img


def append_episode_metrics(env_transition, eval_metrics, episode_metrics, term_keys):
    """Aggregate and append episode metrics to overall evaluation results.

    Args:
        env_transition: The final state of the episode.
        eval_metrics: Dictionary of cumulative evaluation metrics.
        episode_metrics: Dictionary of metrics for the current episode.
        term_keys: List of termination keys.

    Returns:
        Updated evaluation metrics dictionary.

    """
    for key, value in episode_metrics.items():
        operand = _metrics_operands[key]
        if not isinstance(operand, dict):
            new_value = operand(value)
            if key not in eval_metrics:
                eval_metrics[key] = []
            eval_metrics[key].append(new_value)
        else:
            for sub_key, sub_operand in operand.items():
                new_value = sub_operand(value)
                if sub_key not in eval_metrics:
                    eval_metrics[sub_key] = []
                eval_metrics[sub_key].append(new_value)

    eval_metrics["episode_length"].append(env_transition.info["steps"])

    # Termination condition config
    is_episode_not_finished = 0
    for key in term_keys:
        is_episode_not_finished += np.sum(episode_metrics[key])

    eval_metrics["accuracy"].append(1 - (is_episode_not_finished > 0))

    eval_metrics_last = {k: v[-1] for k, v in eval_metrics.items() if k != "nuplan_aggregate_score"}
    if "nuplan_aggregate_score" not in eval_metrics:
        eval_metrics["nuplan_aggregate_score"] = []
    eval_metrics["nuplan_aggregate_score"].append(nuplan_aggregate_score(eval_metrics_last))

    if "vmax_aggregate_score" not in eval_metrics:
        eval_metrics["vmax_aggregate_score"] = []
    eval_metrics["vmax_aggregate_score"].append(vmax_aggregate_score(eval_metrics_last))

    return eval_metrics
