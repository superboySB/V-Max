# Metrics Documentation

This document details the metrics implemented in the [Waymax](https://github.com/waymo-research/waymax) simulator along with additional metrics introduced in the V-max framework. These metrics can be used during training and evaluation to assess model performance, and some also serve as reward components in training.

---

## Waymax Metrics

### Log Divergence
- **Type:** Float
- **Description:**
  Returns the L2 distance between the controlled object's current XY position and its corresponding position in the logged (ground truth) history at the same timestep.

### Offroad
- **Type:** Boolean
- **Description:**
  Returns `1.0` if the object is offroad; otherwise, returns `0`.

### Overlap
- **Type:** Boolean
- **Description:**
  Determines if the object's bounding box overlaps with that of another object. Returns `1.0` for an overlap, and `0` otherwise.

### SDC Off Route
- **Type:** Boolean
- **Description:**
  Evaluates whether the Self-Driving Car (SDC) is off-route based on one of the following conditions:
  - The SDC is farther than `MAX_DISTANCE_TO_ROUTE_PATH` from the closest on-route path.
  - The SDC is farther from the closest on-route path than the closest off-route path by a margin defined as `MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH`.
  If the SDC is on-route, the trajectory is either completely invalid or there are no valid on-route paths, in which case the metric returns zero. If off-route, it returns the distance to the nearest valid on-route path.

### SDC Wrongway
- **Type:** Float
- **Description:**
  Checks if the SDC is driving in the wrong direction.
  - It computes the distance from the SDC's starting position to the closest roadgraph point along all valid paths.
  - If this distance exceeds the threshold (`WRONG_WAY_THRES`), the metric returns the distance (representing a wrongway deviation); otherwise, it returns `0.0`.

### SDC Progression
- **Type:** Float
- **Description:**
  Measures the progress of the SDC along the intended route. It calculates the arc length between:
  - The SDC’s initial logged position.
  - Its current position along the closest on-route path, determined using Euclidean distance.
  If the SDC’s trajectory is invalid or no valid on-route paths exist, the metric returns zero.

---

## V-max Metrics

### Time-To-Collision (TTC)
- **Type:** Float
- **Description:**
  Returns the minimum time-to-collision between the SDC and any object in the scenario, computed over a specified time horizon.

### Run Red Light
- **Type:** Float
- **Description:**
  Returns `1.0` if the ego vehicle has run a red light; otherwise, returns `0`.

### Speed Limit Violation
- **Type:** Float
- **Description:**
  Measures by how much the ego vehicle's speed exceeds the inferred speed limit (in m/s). Higher values indicate greater violations.

### Progress Ratio
- **Type:** Float
- **Description:**
  Computes a normalized progress metric toward the expert (logged) trajectory. Values are scaled relative to the expert’s total route length.

### On Multiple Lanes
- **Type:** Float
- **Description:**
  Quantifies lane deviation by combining data on speed and variance in lane-crossing. Higher metric values indicate more frequent or severe deviations.

### Driving Direction Compliance
- **Type:** Float
- **Description:**
  Evaluates whether the ego vehicle is driving into oncoming traffic:
  - This metric measures the distance covered in the wrong driving direction.
  - A higher value indicates a greater deviation from proper driving direction.

### Comfort
- **Type:** Boolean
- **Description:**
  Assesses if the vehicle’s driving dynamics meet comfort criteria under nuPlan standards (including measures such as lateral acceleration, longitudinal acceleration, jerk, and yaw rate/acceleration).
  - Returns `1.0` if criteria are met.
  - Returns `0` otherwise.

### At Fault Collision
- **Type:** Float
- **Description:**
  Computes the number of collisions attributable to the ego vehicle’s actions. A value of `0` indicates no at fault collisions.
