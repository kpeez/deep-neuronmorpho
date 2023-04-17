"""Perturbations of neuron structures for contrastive learning."""
import pandas as pd


def perturb_points(neuron_swc: pd.DataFrame, prop: float = 0.5) -> pd.DataFrame:
    """Perturb a neuron by shifting a prop of points by a random distance (0-10 µm).

    Args:
        neuron_swc (pd.DataFrame): swc data.
        prop (float, optional): Proportion of points to shift. Defaults to 0.5.

    Returns:
        pd.DataFrame: Neuron swc data with perturbed points.
    """
    # TODO:
    # 1. select prop of points to perturb
    # 2. set random distance between 0 and 10 µm
    # 3. shift all points by distance

    raise NotImplementedError


def drop_branches(neuron_swc: pd.DataFrame) -> pd.DataFrame:
    """Drop a random branch from a neuron."""
    # TODO:
    # 1. implement probability of branch selection from ZhaoEtAl2022
    # 2. select 2% of branches to drop based on probability from 1.
    # 3. drop branches from neuron

    raise NotImplementedError


def rotate_neuron(neuron_swc: pd.DataFrame) -> pd.DataFrame:
    """Rotate a neuron about an axis by a random angle (0-360 degrees)."""
    # TODO:
    # 1. select random axis to rotate about
    # 2. select random angle between 0 and 360 degrees
    # 3. rotate neuron about axis by angle

    raise NotImplementedError
