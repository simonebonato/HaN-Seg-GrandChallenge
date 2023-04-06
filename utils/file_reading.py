import os
import json
import re
import numpy as np
import nrrd

from typing import Tuple


def read_config(path: str = "config.yaml") -> dict:
    """
    Reads a config file and returns a dictionary of the contents

    Parameters
    :param path: The path to the config file

    Returns
    :return: A dictionary of the contents of the config file
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)


def get_number_from_name(name: str) -> str:
    """
    Gets the number from a name

    Parameters
    :param name: The name to get the number from

    Returns
    :return: The number from the name
    """
    return re.findall(r"\d+", name)[0].lstrip("0")


def merge_segmentations(
    case_path: str, segmentation_files: str
) -> Tuple[np.ndarray, dict]:
    """
    Merges the segmentations of a case into one segmentation.
    Repeats it for all the segmentation files.

    Parameters
    :param case_path: The path to the case
    :param segmentation_files: The segmentation files of the case

    Returns
    :return: The merged segmentation and a dictionary of the names of the organs
    """

    names_dict = {0: "background"}
    merged_segmentation = 0

    for idx, segm_file in enumerate(segmentation_files):
        oar_name = re.search(r"_OAR_(.+)\.seg\.nrrd", segm_file).group(1)
        names_dict[idx + 1] = oar_name

        data, _ = nrrd.read(os.path.join(case_path, segm_file))
        merged_segmentation += data * (idx + 1)

    return merged_segmentation.astype(np.uint8), names_dict
