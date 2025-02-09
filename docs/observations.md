# Observation Wrappers Documentation

This document explains the observation wrappers used in V-Max and delineates the role of their corresponding feature extractors.

## Observation Wrappers

We use wrappers to apply an observation function on top of the environment. The goal is to override the `observe` function to apply a custom transformation.

We designed several configurable observations:
- `base_observation.py`
- `lane_observation.py`
- `road_observation.py`
- `segment_observation.py`
- `gt_observation.py`

Each observation works with a feature extractor that is instantiated within its wrapper. The feature extractor transforms a `datatypes.Observation` into a structured set of features for downstream processing.

We define four feature groups in `features_datatypes.py`, along with their associated feature types:

- **Objects** (vehicles, pedestrians, bicycles):
    - waypoints (xy)
    - velocity (vel_xy)
    - speed (speed)
    - yaw (yaw)
    - size (length, width)
    - object_types (object_types)
    - valid (valid)
- **Roadgraph points**:
    - waypoints (xy)
    - direction (dir_xy)
    - types (types)
    - valid (valid)
- **Traffic Lights**:
    - waypoints (xy)
    - state (state)
    - valid (valid)
- **Path Target**:
    - waypoints (xy)

For each feature group, a common extraction process is defined:
- **Objects**: Dynamic objects (up to N objects, with N = max_num_objects - 1) are sorted by distance to the SDC.
- **Traffic Lights**: A similar strategy is employed, considering only traffic lights relevant to the SDC.
- **Path Target**: Selects the longest SDC path containing the SDC's log trajectory and subsamples it into N points spaced every X meters.
- **Roadgraph points**: Involves a multi-step process:
    - First, select points from the SDC observation using a top_k strategy to keep a fixed number of the closest points. Alternatively, a rectangular selection can be applied to filter points within a specified range in front, back, left, and right of the SDC.
    - Points are then downsampled by a specified interval (default is one, meaning every point is kept; using 2 means one point every two points).
    - Finally, a filter is applied to retain only points of certain types (e.g., lane center points, road lines, or road edges).

### Base Observation & Extractor

![alt text](https://github.com/valeoai/V-Max/tree/main/docs/assets/png/base_obs.png)


The Base Observation wrapper provides a standard interface that:
- Initializes the observation with default configurations.
- Instantiates a Base Features Extractor that sets up necessary normalization routines.
- Handles the aggregation of various feature groups (objects, roadgraph points, traffic lights, and path targets) into a single concatenated observation array.

The associated Base Features Extractor is designed for extensibility:
- It establishes default parameters for each feature group while allowing overrides.
- It supports dynamic extraction and normalization of features.
- Its modular structure simplifies the creation of specialized extractors for different observation requirements.

### Lane Extractor

![alt text](https://github.com/valeoai/V-Max/tree/main/docs/assets/png/lane_obs.png)

Specializes in extracting lane-specific features from roadgraph points. It applies a lane-specific filtering to retain only lane surface street points, ensuring that only relevant roadway lane data is processed. This targeted design aids in constructing a more precise representation of the roadway lanes.

### Road Extractor

![alt text](https://github.com/valeoai/V-Max/tree/main/docs/assets/png/road_obs.png)

Focuses on extracting features related to road lines and edge boundaries. This extractor is essential for identifying marked road elements and contributes to safe navigation by highlighting lane demarcations.

### Segment Extractor

![alt text](https://github.com/valeoai/V-Max/tree/main/docs/assets/png/segment_obs.png)

Implements a sampling strategy to efficiently extract features from high-density observations. It subsamples roadgraph points and other features to reduce computational overhead while still preserving essential information.

### GT Extractor

![alt text](https://github.com/valeoai/V-Max/tree/main/docs/assets/png/gt_obs.png)

Designed to compute ground truth path target features. It analyzes the SDC’s path history to generate accurate target waypoints, often used in supervised learning scenarios for trajectory prediction.
