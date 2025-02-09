# Copyright 2025 Valeo.

from collections.abc import Callable

import jax
from jax import numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core


def create_random_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
) -> actor_core.WaymaxActorCore:
    """Create an actor that produces random actions within model-specified bounds.

    Args:
        dynamics_model: The dynamics model.
        is_controlled_func: A function that determines which objects are controlled by this actor.

    Returns:
        A random actor.

    """

    def select_action(
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state=None,
        rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        """Generate a random action using the dynamics model's action specification.

        Args:
            params: The actor parameters.
            state: The simulator state.
            actor_state: The actor state.
            rng: The random number generator.

        """
        del params, actor_state  # unused.
        # Generating actions within the specified bounds
        dynamics_model_spec = dynamics_model.action_spec()
        actions = jax.random.uniform(
            key=rng,
            shape=dynamics_model_spec.shape,
            minval=dynamics_model_spec.minimum,
            maxval=dynamics_model_spec.maximum,
        )
        # actions = jnp.expand_dims(actions, axis=0) # TO ADD in EVAL
        actions = datatypes.Action(data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_))
        actions.validate()

        is_controlled = is_controlled_func(state)
        return actor_core.WaymaxActorOutput(actor_state=None, action=actions, is_controlled=is_controlled)

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name="random",
    )
