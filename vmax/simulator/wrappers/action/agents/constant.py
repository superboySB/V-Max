# Copyright 2025 Valeo.

from collections.abc import Callable

import jax
from jax import numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core


def create_constant_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
) -> actor_core.WaymaxActorCore:
    """Create an actor that always returns constant zero actions.

    Args:
        dynamics_model: The dynamics model.
        is_controlled_func: A function that determines which objects are controlled by this actor.

    Returns:
        A constant actor.

    """

    def select_action(
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state=None,
        rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        """Return a constant (zero) action independent of the state.

        Args:
            params: The actor parameters.
            state: The simulator state.
            actor_state: The actor state.
            rng: The random number generator.

        Returns:
            The constant action.

        """
        del params, actor_state, rng  # unused.
        dynamics_model_spec = dynamics_model.action_spec()
        actions = jnp.zeros((state.object_metadata.num_objects, dynamics_model_spec.shape[0]))
        actions = datatypes.Action(data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_))
        actions.validate()

        is_controlled = is_controlled_func(state)
        return actor_core.WaymaxActorOutput(actor_state=None, action=actions, is_controlled=is_controlled)

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name="constant",
    )
