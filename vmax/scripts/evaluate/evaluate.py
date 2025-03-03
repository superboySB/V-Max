import argparse
import os
import time

import jax

from vmax.scripts.evaluate import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_eval_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluation script arguments")
    parser.add_argument(
        "--sdc_actor",
        "-sdc",
        type=str,
        default="expert",
        help="Actor type: 'ai' for learned policy, otherwise rule-based (default: expert)",
    )
    parser.add_argument(
        "--max_num_objects",
        "-o",
        type=int,
        default=64,
        help="Maximum number of objects in the scene (default: 64)",
    )
    parser.add_argument(
        "--scenario_indexes",
        "-si",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of scenario indexes to evaluate",
    )
    parser.add_argument("--render", "-r", type=str2bool, default=False, help="Render the evaluation (default: False)")
    parser.add_argument(
        "--sdc_pov",
        "-pov",
        type=str2bool,
        default=False,
        help="Render from self-driving car's point of view (default: False)",
    )
    parser.add_argument(
        "--path_dataset",
        "-pd",
        type=str,
        default="local_womd_valid",
        help="Path to the dataset (default: local_womd_valid)",
    )
    parser.add_argument(
        "--path_model",
        "-pm",
        type=str,
        default="SAC_MTR_CI",
        help="Identifier of the model to evaluate",
    )
    parser.add_argument(
        "--eval_name",
        "-en",
        type=str,
        default="benchmark",
        help="Base evaluation directory name or identifier",
    )
    parser.add_argument(
        "--noisy_init",
        "-ni",
        type=str2bool,
        default=False,
        help="Flag to enable noisy initialization (default: False)",
    )
    parser.add_argument(
        "--src_dir",
        "-sd",
        type=str,
        default="runs",
        help="Source directory for evaluation results (default: runs)",
    )
    return parser


def run_evaluation(
    env,
    data_generator,
    step_fn,
    run_path: str = "",
    scenario_indexes: list | None = None,
    termination_keys: list | None = None,
    render: bool = False,
    render_pov: bool = False,
):
    """Evaluate the model over multiple episodes and store the results.

    Args:
        env: The simulation environment.
        data_generator: Generator providing evaluation scenarios.
        step_fn: Step function for simulation steps.
        run_path: Directory for saving evaluation outputs.
        eval_args: Evaluation arguments.
        termination_keys: List of keys that determine episode termination.

    """
    # JIT compile the step function and reset for speed.
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    rng_key = jax.random.PRNGKey(0)
    eval_metrics = {"episode_length": [], "accuracy": []}
    start_time_total = time.time()

    for idx, scenario in enumerate(data_generator):
        # Skip scenarios if scenario_indexes is provided.
        if scenario_indexes is not None and idx not in scenario_indexes:
            continue

        # Reset environment for the new scenario.
        env_transition = jitted_reset(scenario)
        list_images = []
        done = False
        episode_metrics = {}

        if render:
            list_images.append(utils.plot_scene(env, env_transition, render_pov))

        # Step until the episode ends.
        while not done:
            rng_key, step_key = jax.random.split(rng_key)
            env_transition, transition = jitted_step_fn(env_transition, key=step_key)

            done = transition.done

            # Collect episode metrics.
            for key, value in env_transition.metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)

            if render:
                list_images.append(utils.plot_scene(env, env_transition, render_pov))

        # Aggregate metrics for the episode.
        eval_metrics = utils.append_episode_metrics(env_transition, eval_metrics, episode_metrics, termination_keys)

        if render:
            utils.write_video(run_path, list_images, idx)

    # Write aggregated evaluation results.
    utils.write_generator_result(run_path, idx + 1, eval_metrics)
    print(f"Duration eval {idx + 1} episodes is ", time.time() - start_time_total)


def main():
    parser = parse_eval_args()
    eval_args = parser.parse_args()

    data_generator = make_data_generator(
        path=datasets.get_dataset(eval_args.path_dataset),
        max_num_objects=eval_args.max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=0,
        repeat=1,
    )
    dummy_scenario = next(data_generator)

    env, step_fn, eval_path, termination_keys = utils.setup_evaluation(
        dummy_scenario,
        eval_args.sdc_actor,
        eval_args.path_model,
        eval_args.src_dir,
        eval_args.path_dataset,
        eval_args.eval_name,
        eval_args.max_num_objects,
        eval_args.noisy_init,
    )
    run_evaluation(
        env,
        data_generator,
        step_fn,
        eval_path,
        eval_args.scenario_indexes,
        termination_keys,
        eval_args.render,
        eval_args.sdc_pov,
    )


if __name__ == "__main__":
    main()
