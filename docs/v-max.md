# V-Max Simulator Documentation

This document provides an extensive overview of how V-Max works, its core components, internal flow, and how it integrates with the training and evaluation pipelines.

---

## Overview

V-Max is a plug-and-play extension of the [Waymax](https://github.com/waymo-research/waymax) simulator. It is designed to support research in autonomous driving by providing a unified framework for:

- **Simulation of driving scenarios** using recorded or generated data.
- **Flexible observation and action wrappers** for different operating modes.
- **Integrated metrics and rewards** for training reinforcement or imitation learning agents.
- **Seamless integration with training pipelines** via Hydra-managed configuration files.

---

## Core Components

### Simulator State and Data Structures

- **SimulatorState Dataclass**
  The simulation state is held in a `SimulatorState`  that encapsulates all relevant data:
  - **Roadgraph**: Static elements including lane center points, road boundaries.
  - **Trajectories**: Both logged and simulated trajectories for all objects.
  - **Object Metadata**: Identification of the self-driving car (SDC) and other objects.
  - **Timestep Information**: Current simulation step used for slicing trajectory and computing metrics.

### Environment Wrappers

The simulator provides several wrappers to standardize the interface and manage simulation behavior.

- **Brax-like Training Wrapper**
  Located in [`vmax/simulator/wrappers/environment/training.py`](vmax/simulator/wrappers/environment/training.py), the BraxWrapper standardizes the interaction with the environment in a way that is compatible with high-performance JAX pipelines. It handles:
  - **State Transitions**
  - **Metrics Computation**
  - **Episode Resets and Steps**

### Observation and Action Wrappers

- **Observation Wrappers**
  Modules such as [`base_observation.py`](vmax/simulator/wrappers/observation/base_observation.py) export observation functions that extract various data (e.g., GPS points, roadgraph features) from the simulator state to be consumed by learning algorithms.

- **Action Wrappers**
  Action wrappers in the simulator are used to transform agent outputs into a form that can be executed by the underlying simulation dynamics. They are integrated with both RL and imitation learning pipelines.

### Metrics

The simulator provides a variety of metrics that can be used both for reward calculations during training and for evaluations:

- **Comfort Metric**
  Implemented in [`vmax/simulator/metrics/comfort.py`](vmax/simulator/metrics/comfort.py), it calculates measures such as lateral acceleration, longitudinal acceleration, jerk, and yaw changes. Examples include:
  - `_compute_lateral_acceleration`
  - `_compute_longitudinal_acceleration`

- **Speed Limit**
  Functions like [`infer_speed_limit_from_simulator_state`](vmax/simulator/metrics/speed_limit.py) compute the applicable speed limit by comparing roadgraph data and expert trajectories.

- **Other Metrics**
  Additional metrics include time-to-collision, running red light detection, and driving direction compliance, all of which are aggregated and collected during a simulation session.

---

## Simulator Workflow

### 1. Initialization

- **Loading a Scenario**
  A new scenario is typically loaded from a data generator that emits instances of simulator states (see use in training notebooks such as [`notebooks/dev_new_obs.ipynb`](notebooks/dev_new_obs.ipynb)).

- **Environment Reset**
  The environment is reset using a method like:
  ```python
  simulator_state = env.reset(scenario)

This initializes the internal state, roadgraph, and trajectories for the upcoming simulation.

### 2. Interaction

- **Step Function**
    After initialization, the environment’s step function is called with an action from the agent. This step updates the simulation state via:
    - Physics and dynamic models defined in the simulator.
    - Wrappers that adjust the state and compute new metrics.
    - The new state is then used as input for the next agent action.

- **Reward & Metrics**
At every step the simulator computes various metrics (comfort, speed limit violations, etc.) using its metric modules. These metrics are later aggregated for training and logging, as seen in training trainer files like sac_trainer.py.

### 3. Training and Evaluation Integration
- **Training Loop**
Trainer scripts (for example, vmax/learning/algorithms/rl/sac/sac_trainer.py) integrate the simulator by:
    - Collecting batches of scenarios.
    - Running the simulation steps.
    - Updating replay buffers and running the learning updates.
    - Periodically saving model checkpoints based on simulation progress.
- Evaluation
- Separate evaluation scripts (located under vmax/scripts/evaluate/) use the simulator to run episodes with evaluation metrics computed after defined intervals.

---

## Configuration
V-Max leverages configuration files managed via Hydra. Key configuration files include:

- Base Config: base_config.yaml
    Contains universal settings such as environment parameters, metric thresholds, and buffer sizes.
- Algorithm-specific Configs: Stored under vmax/config/algorithms/
- Encoder-specific Configs: Stored under vmax/config/encoders/

These files allow users to customize:

- Observation function selection and parameters
- Reward and penalty metrics settings
- Learning parameters such as batch size, learning rates, imitation frequency, etc.


---

## Summary
The V-Max simulator is built as a modular extension of Waymax, with well-defined components for simulating real-world driving scenarios. Its design incorporates:

- Flexible wrappers for environment interactions.
- Detailed metric computations for realistic reward shaping.
- Configurable pipelines that integrate into robust training and evaluation routines.

For further details, see:

- Simulator Wrappers
- Metrics Modules
- Training Integrations
- Configuration Files
