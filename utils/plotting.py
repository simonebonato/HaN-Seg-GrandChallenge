import matplotlib.pyplot as plt
import numpy as np
from utils.file_reading import load_sample


def plot_sample_from_dir(
    plane: str,
    case_number: int,
    slice_number: int,
    imagesTr_folder: str = "data/imagesTr/",
    labelsTr_folder: str = "data/labelsTr/",
) -> None:
    """
    Plots the CT, MRI and segmentation of a sample given the case number and the slice number.
    If the slice number is too high for the MR image, it will only plot the CT and segmentation.

    Parameters
    :param case_number: The case number of the sample
    :param slice_number: The slice number of the sample
    :param imagesTr_folder: The path to the imagesTr folder
    :param labelsTr_folder: The path to the labelsTr folder
    """
    assert plane in [
        "a",
        "c",
        "s",
    ], "The plane must be one of the following: a (axial), c (coronal) or s (sagittal)"

    CT, MR, segmentation = load_sample(case_number, imagesTr_folder, labelsTr_folder)

    if plane == "a":
        CT = np.transpose(CT, (2, 1, 0))
        MR = np.transpose(MR, (2, 1, 0))
        segmentation = np.transpose(segmentation, (2, 1, 0))
    elif plane == "s":
        CT = np.transpose(CT, (0, 2, 1))
        MR = np.transpose(MR, (0, 2, 1))
        segmentation = np.transpose(segmentation, (0, 2, 1))
    elif plane == "c":
        CT = np.transpose(CT, (1, 2, 0))
        MR = np.transpose(MR, (1, 2, 0))
        segmentation = np.transpose(segmentation, (1, 2, 0))

    if plane in ["s", "c"]:
        CT = np.flip(CT, axis=1)
        MR = np.flip(MR, axis=1)
        segmentation = np.flip(segmentation, axis=1)

    assert (
        slice_number < CT.shape[0]
    ), f"The slice number is too high, select a number lower than {CT.shape[0]}"

    CT_slice = CT[slice_number]
    segmentation_slice = segmentation[slice_number]

    plot_MR = False
    MR_slice = np.array([])
    if slice_number <= MR.shape[0]:
        MR_slice = MR[slice_number]
        plot_MR = True

    plt.figure(figsize=(40, 10))
    if plot_MR:
        plot_all(CT_slice, MR_slice, segmentation_slice)
    else:
        print("The slice number is too high for the MR image.")
        print(
            f"If you want to plot the MR image, please choose a slice number lower than {MR.shape[0]}"
        )
        plot_CT_label(CT_slice, segmentation_slice)


def plot_all(
    CT_slice: np.ndarray, MR_slice: np.ndarray, segmentation_slice: np.ndarray
) -> None:
    """
    Plots the CT, MRI and segmentation of a sample given the case number and the slice number.

    Parameters
    :param CT_slice: The CT slice
    :param MR_slice: The MR slice
    :param segmentation_slice: The segmentation slice
    """
    plt.subplot(2, 2, 1)
    plt.imshow(CT_slice, cmap="gray")
    plt.title("CT")

    plt.subplot(2, 2, 2)
    plt.imshow(segmentation_slice, cmap="jet")
    plt.title("Segmentation")

    plt.subplot(2, 2, 3)
    plt.imshow(CT_slice, cmap="gray")
    plt.imshow(segmentation_slice, alpha=0.5, cmap="jet")
    plt.title("CT + Segmentation")

    plt.subplot(2, 2, 4)
    plt.imshow(MR_slice, cmap="gray")
    plt.title("MR")

    plt.tight_layout()
    plt.show()


def plot_CT_label(CT_slice: np.ndarray, segmentation_slice: np.ndarray) -> None:
    """
    Plots the CT and segmentation of a sample given the case number and the slice number.

    Parameters
    :param CT_slice: The CT slice
    :param segmentation_slice: The segmentation slice
    """

    plt.subplot(1, 3, 1)
    plt.imshow(CT_slice, cmap="gray")
    plt.title("CT")

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_slice, cmap="jet")
    plt.title("Segmentation")

    plt.subplot(1, 3, 3)
    plt.imshow(CT_slice, cmap="gray")
    plt.imshow(segmentation_slice, alpha=0.5, cmap="jet")
    plt.title("CT + Segmentation")

    plt.tight_layout()
    plt.show()
