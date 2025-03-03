# :video_game: **<span style="color:pink">Trainings & Configurations</span>**

Welcome! This document details the training process along with configuration options for environment, reward functions, observations, and network settings.

We provide more details about how to use in trainings:

- Configuration files (environment parameters, algorithm, network selections)
- Observation function and features selections
- Reward function and metrics selections

## V-Max Configuration Files

Below are the key configuration files used for training and evaluation.

### Base Training Configuration

File: `vmax/config/base_config.yaml`

This file sets parameters for training, environment, rewards, and observations.

| Parameter                  | Description                                     | Possible Values                      | Default        |
|----------------------------|-------------------------------------------------|--------------------------------------|----------------|
| **total_timesteps**        | Number of environment steps                     | *integer*                            | `50,000,000`   |
| **num_envs**               | Number of vectorized environments               | *integer*                            | `16`           |
| **num_episode_per_epoch**  | Scenarios per training iteration                | *integer*                            | `4`            |
| **num_scenario_per_eval**  | Scenarios for policy evaluation                 | *integer*                            | `256`          |
| **scenario_length**        | Timesteps per scenario                          | *integer*                            | `80`           |
| **log_freq**               | Frequency for training logs                     | *integer*                            | `2`            |
| **save_freq**              | Frequency of model saving                       | *integer*                            | `50`           |
| **eval_freq**              | Frequency of model evaluation                   | *integer*                            | `0`            |
| **seed**                 | Random seed for reproducibility                 | *integer*                            | `0`            |
| **name_run**              | Identifier for the training run                 | *string*                             | `null`         |
| **name_exp**              | Experiment name                                 | *string*                             | `null`         |

### Environment Configuration

Defines dataset paths and environment constraints:

| Parameter               | Description                                     | Possible Values   | Default  |
|-------------------------|-------------------------------------------------|-------------------|----------|
| **path_dataset**        | Dataset path for training                       | *string*          | `???`    |
| **path_dataset_eval**   | Dataset path for evaluation                     | *string*          | `null`   |
| **max_num_objects**     | Maximum objects in the environment              | *integer*         | `64`     |
| **termination_keys**    | Conditions that terminate a scenario            | List of conditions| `[offroad, overlap, run_red_light]`     |

### Running Trainings

Execute the training with:

```
python vmax/scripts/train.py total_timesteps=$num_env_steps path_dataset="" path_dataset_eval="" algorithm=$alg_type network/encoder=$encoder_type
```

Override default parameters directly via command line:

```
python vmax/scripts/train.py parameter1_name=$value1 parameter2_name=$value2
```

### Reward Function Configuration

Configure rewards with:

| Parameter            | Description                             | Possible Values         | Default  |
|----------------------|-----------------------------------------|-------------------------|----------|
| **reward_type**      | Reward function type                    | `linear`, `custom`      | `linear` |

###### Reward Components

| Component               | Description                                      | Parameters                           |
|-------------------------|--------------------------------------------------|--------------------------------------|
| **log_div_clip**        | Proximity to a log position threshold            | `threshold`: `0.3`, `bonus`: `0.0`, `penalty`: `0.0`, `weight`: `1.0` |
| **log_div**             | Reward based on log divergence                   | `weight`: `1.0`                                                        |
| **overlap**             | Penalty when overlap occurs                      | `bonus`: `0.0`, `penalty`: `-1.0`, `weight`: `1.0`                     |
| **offroad**             | Penalty when offroad occurs                      | `bonus`: `0.0`, `penalty`: `-1.0`, `weight`: `1.0`                     |
| **ttc**                 | Time-to-collision threshold                      | `threshold`: `1.5`, `bonus`: `0.0`, `penalty`: `-1.0`, `weight`: `1.0` |
| **red_light**           | Penalty for crossing a red light                 | `penalty`: `-1.0`, `weight`: `1.0`                                      |
| **comfort**             | Reward related to comfort                       | `weight`: `1.0`                                                        |
| **speed**               | Reward based on speed-limit adherence            | `penalty`: `-1.0`, `bonus`: `0.0`, `weight`: `1.0`                      |
| **driving_direction**   | Reward for driving direction compliance          | `penalty`: `-1.0`, `bonus`: `0.0`, `weight`: `1.0`                      |
| **lane_deviation**      | Reward penalizing lane deviations                | `penalty`: `-1.0`, `bonus`: `0.0`, `weight`: `1.0`                      |
| **progression**         | Reward promoting progress                        | `penalty`: `0.0`, `bonus`: `1.0`, `weight`: `1.0`                      |

---

### Observation Function Configuration

| Component               | Description                                      | Parameters                           | Default         |
|-------------------------|--------------------------------------------------|--------------------------------------|-----------------|
| **observation_type**    | Type of observation function                     | `base`, `road`, `lane`, `gt`, `segment`| `base`   |

###### Observation Components

| Component                      | Description                                      | Default         | Used in         |
|--------------------------------|--------------------------------------------------|-----------------|-----------------|
| **obs_past_num_steps**         | Number of past steps in observation              | `5`             | base, road, lane, gt, segment |
| **objects**.num_closest_objects   | Number of closest objects to consider             | `8`             | base, road, lane, segment |
| **roadgraphs**.meters_box   | Rectangle to filter the BEV                    | {front: 50, back: 10, left: 20, right: 20}            | base, road, lane |
| **roadgraphs**.roadgraph_top_k   | Maximum number of points to keep after filtering of the BEV                   | 500        | base, road, lane |
| **roadgraphs**.max_meters  | Max meters for feature normalisation                        | `50`            | base, road, lane, segment |
| **roadgraphs**.interval   | Take 1 point data every X meters                       | `50`            | segment |
| **roadgraphs**.max_num_points_per_lane   | Number of points per segment road                | `10`            | segment |
| **traffic_lights**.num_closest_traffic_lights     | Maximum number of traffic lights                 | `16`            | base, road, lane |
| **path_target**.num_points   | Number of points in the target path               | `10`             | base, road, lane, gt, segment |
| **path_target**.points_gap   | Gap between points in the target path               | `5`             | base, road, lane, gt, segment |

###### Features for Observations

| Feature                       | Description                                        | Possible Values                                          |
|-------------------------------|----------------------------------------------------|---------------------------------------------------------|
| **objects_features**     | List of object-related features                | `waypoints`, `velocity`, `yaw`, `size`, `valid` |
| **roadgraph_features**        | List of road graph-related features                | `waypoints`, `direction`, `types`, `valid`           |
| **traffic_lights_features**   | List of traffic light-related features             | `waypoints`, `state`, `valid`                             |
| **path_target_features**         | List of target path-related features                  | `waypoints`                                               |

This configuration file provides detailed control over the training, environment, reward, and observation functions, enabling flexibility and customization for autonomous driving research.

### Additional Configuration Files

- **Algorithm Configuration:** Located at `vmax/config/algorithm/$alg_name.yaml`; sets learning parameters and network specifics.
- **Encoder Configuration:** Located at `vmax/config/network/encoder/$enc_name.yaml`; defines network embedding layers for feature extraction.

#### `vmax/config/algorithm/$alg_name.yaml`

###### **Example with SAC Algorithm Configuration**

| Parameter               | Description                                      | Possible Values                    | Default         |
|-------------------------|--------------------------------------------------|------------------------------------|-----------------|
| **name**                | Name of the algorithm                            | *string*                           | `SAC`           |
| **unroll_length**       | Number of steps to unroll in training            | *integer*                          | `1`             |
| **learning_rate**       | Learning rate for training                       | *float*                            | `0.0001`          |
| **discount**            | Discount factor for future rewards               | *float*                            | `0.99`          |
| **tau**                 | Soft update coefficient for target network       | *float*                            | `0.005`         |
| **alpha**               | Entropy regularization coefficient               | *float*                            | `0.2`           |
| **batch_size**          | Number of samples per batch                      | *integer*                          | `64`            |
| **grad_updates_per_step** | Gradient updates per environment step         | *integer*                          | `1`             |
| **buffer_size**         | Size of the replay buffer                | *integer*                          | `1000000`       |
| **learning_start**      | Random steps before training begins                     | *integer*                          | `50000`        |

---

###### **Policy Network**

| Parameter               | Description                                      | Possible Values                    | Default         |
|-------------------------|--------------------------------------------------|------------------------------------|-----------------|
| **layer_sizes**         | Number and size of hidden layers                 | *list of integers*                 | `[256, 256]`    |
| **activation**          | Activation function for hidden layers            | `relu`, `gelu`, `selu`             | `relu`          |
| **final_activation**    | Activation function for the output layer         | `null`                             | `null`          |

###### **Value Network**

| Parameter               | Description                                      | Possible Values                    | Default         |
|-------------------------|--------------------------------------------------|------------------------------------|-----------------|
| **layer_sizes**         | Number and size of hidden layers                 | *list of integers*                 | `[256, 256]`    |
| **activation**          | Activation function for hidden layers            | `relu`, `gelu`, `selu`             | `relu`          |
| **final_activation**    | Activation function for the output layer         | `relu`, `tanh`                     | `relu`          |
| **num_networks**        | Number of value networks                         | *integer*                          | `2`             |
| **shared_encoder**      | Whether policy and value networks share encoder  | `true`, `false`                    | `false`         |

---

#### `vmax/config/network/encoder/$enc_name.yaml`

###### **Example with MGAIL Encoder Configuration**

| Parameter                   | Description                                      | Possible Values                    | Default         |
|-----------------------------|--------------------------------------------------|------------------------------------|-----------------|
| **type**                    | Type of the encoder                             | *string*                           | `mgail`         |
| **embedding_layer_sizes**    | Sizes of the layers in the embedding network    | *list of integers*                 | `[256, 256]`    |
| **embedding_activation**     | Activation function for embedding layers         | `relu`, `gelu`, `selu`             | `relu`          |
| **dk**                      | Dimensionality of the key and value embeddings   | *integer*                          | `64`            |
| **num_latents**            | Number of latent variables                        | *integer*                          | `16`            |
| **latent_num_heads**       | Number of attention heads in the latent space    | *integer*                          | `2`             |
| **latent_head_features**    | Features per attention head                      | *integer*                          | `16`            |
| **ff_mult**                 | Multiplier for feedforward network size         | *integer*                          | `2`             |
| **attn_dropout**            | Dropout rate for attention layers                | *float*                            | `0.0`           |
| **ff_dropout**              | Dropout rate for feedforward layers              | *float*                            | `0.0`           |
| **use_self_attention**      | Whether to use self-attention                    | `true`, `false`                    | `false`         |
