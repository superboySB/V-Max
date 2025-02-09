# Copyright 2025 Valeo.


"""Algorithms module."""


def get_train_fn(algorithm_type: str) -> callable:
    """Retrieve the training function for the specified algorithm.

    Args:
        algorithm_type: The name of the algorithm (e.g. 'SAC', 'PPO').

    Returns:
        A callable training function corresponding to the specified algorithm.

    Raises:
        ValueError: If the algorithm type is unknown.

    """
    if algorithm_type == "PPO":
        from .rl.ppo import train
    elif algorithm_type == "SAC":
        from .rl.sac import train
    elif algorithm_type == "BC":
        from .imitation.bc import train
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    return train
