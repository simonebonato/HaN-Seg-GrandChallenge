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


def load_sample(
    case_number: int,
    imagesTr_folder: str = "data/imagesTr/",
    labelsTr_folder: str = "data/labelsTr/",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CT, MRI and segmentation for a sample given the case number.

    Parameters
    :param case_number: The case number of the sample
    :param imagesTr_folder: The path to the imagesTr folder
    :param labelsTr_folder: The path to the labelsTr folder

    Returns
    :return: The CT, MRI and segmentation of the sample
    """

    CT_path, MR_path, label_path = get_sample_paths(
        case_number, imagesTr_folder, labelsTr_folder
    )

    assert os.path.exists(CT_path), f"CT image not found at {CT_path}"
    assert os.path.exists(MR_path), f"MR image not found at {MR_path}"
    assert os.path.exists(label_path), f"Label not found at {label_path}"

    CT, _ = nrrd.read(CT_path)
    MR, _ = nrrd.read(MR_path)
    label, _ = nrrd.read(label_path)

    return CT, MR, label


def get_sample_paths(
    case_number: int,
    imagesTr_folder: str = "data/imagesTr/",
    labelsTr_folder: str = "data/labelsTr/",
) -> Tuple[str, str, str]:
    """
    Gets the file paths of the CT, MR and segmentation for a sample given the case number.

    Parameters
    :param case_number: The case number of the sample
    :param imagesTr_folder: The path to the imagesTr folder

    Returns
    :return: The CT, MR and segmentation file paths of the sample
    """
    case_number = str(case_number)

    CT_path = os.path.join(imagesTr_folder, f"case_{case_number.zfill(2)}_IMG_CT.nrrd")
    MR_path = os.path.join(
        imagesTr_folder, f"case_{case_number.zfill(2)}_IMG_MR_T1.nrrd"
    )
    label_path = os.path.join(
        labelsTr_folder, f"case_{case_number.zfill(2)}_segmentation.nrrd"
    )

    return CT_path, MR_path, label_path
