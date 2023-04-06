import os
import json
import re


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
