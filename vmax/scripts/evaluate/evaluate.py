import argparse
import os
import time

import jax

from vmax.scripts.evaluate import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_eval_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", "-a", type=str, default="none")  # sac, bc, ppo
    parser.add_argument(
        "--sdc_actor",
        "-sdc",
        type=str,
        default="expert",
    )  # ai, expert, random, constant, constant_speed
    parser.add_argument("--max_num_objects", "-o", type=int, default=64)
    parser.add_argument(
        "--scenario_indexes",
        "-si",
        nargs="*",
        type=int,
        default=None,
    )

    parser.add_argument("--render", "-r", type=str2bool, default=False)
    parser.add_argument("--sdc_pov", "-pov", type=str2bool, default=False)
    parser.add_argument("--path_dataset", "-pd", type=str, default="local_womd_valid")
    parser.add_argument("--path_model", "-pm", type=str, default="SAC_MTR_CI")
    parser.add_argument("--eval_name", "-en", type=str, default="benchmark")
    parser.add_argument("--noisy_init", "-ni", type=str2bool, default=False)

    return parser


def run_evaluation(
    env,
    data_generator,
    step_fn,
    run_path: str = "",
    eval_args: dict | None = None,
    termination_keys: list | None = None,
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
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)
    rng_key = jax.random.PRNGKey(0)
    eval_metrics = {"episode_length": [], "accuracy": []}
    start_time_total = time.time()

    for ind_scenario, scenario in enumerate(data_generator):
        if eval_args.scenario_indexes is not None and ind_scenario not in eval_args.scenario_indexes:
            continue

        env_transition = jitted_reset(scenario)
        list_images = []
        done = False
        episode_metrics = {}

        if eval_args.render:
            img = utils.plot_scene(
                env,
                env_transition,
                eval_args.sdc_pov,
            )
            list_images.append(img)

        while not done:
            rng_key, step_key = jax.random.split(rng_key)
            env_transition, rl_transition = jitted_step_fn(env_transition, key=step_key)

            done = rl_transition.done

            for key, value in env_transition.metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)

            if eval_args.render:
                img = utils.plot_scene(
                    env,
                    env_transition,
                    eval_args.sdc_pov,
                )
                list_images.append(img)

        eval_metrics = utils.append_episode_metrics(
            env_transition,
            eval_metrics,
            episode_metrics,
            termination_keys,
        )
        if eval_args.render:
            utils.write_video(run_path, list_images, ind_scenario)

    # Average metrics
    eval_results = ind_scenario + 1, eval_metrics
    utils.write_generator_result(run_path, eval_results)

    print(f"Duration eval {ind_scenario + 1} episodes is ", time.time() - start_time_total)


if __name__ == "__main__":
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
        eval_args,
        sdc_paths_from_data=True,
    )

    run_evaluation(
        env,
        data_generator,
        step_fn,
        eval_path,
        eval_args,
        termination_keys,
    )
