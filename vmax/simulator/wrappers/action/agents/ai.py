# Copyright 2025 Valeo.

"""Module for AI-based actors."""

from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core


def create_ai_actor(
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    env: Any,
    policy: Callable,
) -> actor_core.WaymaxActorCore:
    """Create an actor following the AI policy.

    Args:
        is_controlled_func: Defines which objects are controlled by this actor.
        env: The environment object.
        policy: The policy function.

    Returns:
        An stateless actor that drives the controlled objects with constant speed.

    """

    def select_action(
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state=None,
        rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        del params, actor_state, rng  # unused.

        state = jax.lax.stop_gradient(state)  # for ReGentS
        observation = env.observe(state)
        actions, _ = policy(jnp.expand_dims(observation, axis=0), None)
        actions = datatypes.Action(data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_))
        actions.validate()

        is_controlled = is_controlled_func(state)
        return actor_core.WaymaxActorOutput(actor_state=None, action=actions, is_controlled=is_controlled)

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name="ai",
    )
