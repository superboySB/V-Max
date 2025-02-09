# Copyright 2025 Valeo.


"""Datasets paths."""

# WAYMO DATASETS
# fmt: off
WOD_TRAINING = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
WOD_VALID = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
WOD_TEST = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/testing/testing_tfexample.tfrecord@150"
# fmt: on

LOCAL_WOMD = "mini_datasets/mini_womd_dataset.tfrecord"
LOCAL_NUPLAN = "mini_datasets/mini_nuplan_dataset.tfrecord"


def get_dataset(dataset: str) -> str | None:
    """Retrieve the dataset path from the dataset name.

    Args:
        dataset: Name of the dataset.

    Returns:
        The dataset path if recognized, else the provided string.

    """
    dataset_dict = {
        "womd_waymo_training": WOD_TRAINING,
        "womd_waymo_validation": WOD_VALID,
        "womd_waymo_testing": WOD_TEST,
        "local_womd": LOCAL_WOMD,
        "local_nuplan": LOCAL_NUPLAN,
    }

    if dataset in dataset_dict:
        return dataset_dict[dataset]
    else:
        return dataset
