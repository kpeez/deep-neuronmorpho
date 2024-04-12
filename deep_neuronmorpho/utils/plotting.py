from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from umap import UMAP

from deep_neuronmorpho.data import SWCData
from deep_neuronmorpho.engine.evaluation import get_similar_neurons


def check_labels(df: pd.DataFrame, neuron_labels: Mapping[str, str]) -> pd.DataFrame:
    if "label" not in df.columns:
        df["label"] = df["neuron_name"].map(neuron_labels)
    return df


def get_umap_embeddings(
    df: pd.DataFrame,
    labels: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
        **kwargs,
    )
    drop_cols = ["neuron_name", "label"] if labels else ["neuron_name"]
    umap_embeds = umap.fit_transform(df.drop(columns=drop_cols))
    umap_embeds = pd.DataFrame(umap_embeds, columns=["UMAP 1", "UMAP 2"])
    umap_embeds.insert(0, "neuron_name", df["neuron_name"])
    if labels:
        umap_embeds.insert(1, "label", df["label"])

    return umap_embeds


def plot_embeddings(
    df: pd.DataFrame,
    figsize: Sequence[int] = (6, 6),
    by_class: bool = False,
    ax: Any | None = None,
    **kwargs: Any,
) -> None:
    """Plot UMAP embeddings.

    Args:
        df (pd.DataFrame): The input dataframe.
        ax (Any | None): The matplotlib axis to plot on.
        figsize (tuple[int, int], optional): The figure size. Defaults to (6, 6).
        **kwargs (Any): Additional keyword arguments.

    Returns:
        None
    """

    def _plot_embeds(
        df: pd.DataFrame,
        ax: Any | None,
        figsize: tuple[int, int] = (6, 6),
        **kwargs: Any,
    ) -> None:
        if not ax:
            _, ax = plt.subplots(figsize=figsize)

        umap_embs = get_umap_embeddings(df, **kwargs)
        sns.scatterplot(
            x="UMAP 1",
            y="UMAP 2",
            data=umap_embs,
            alpha=0.6,
            s=72,
            hue="label",
            palette="tab10",
            ax=ax,
            edgecolor=None,
        )
        ax.legend(title=None)

    def _plot_embeds_by_class(
        df: pd.DataFrame,
        axs: Any,
        figsize: Sequence[int] = figsize,
        **kwargs: Any,
    ) -> None:
        if "label" not in df.columns:
            raise ValueError("Dataframe must contain a 'label' column")
        unique_labels = sorted(df["label"].unique())
        colors = sns.color_palette("tab10", n_colors=len(unique_labels))
        num_subplots = len(unique_labels)
        nrows = 1
        ncols = num_subplots
        while ncols > 2:
            nrows += 1
            ncols = (num_subplots + nrows - 1) // nrows

        if not axs:
            _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axs = axs.flatten()
        else:
            axs = [axs] if not isinstance(axs, (list, np.ndarray)) else axs

        for lab, ax, color in zip(unique_labels, axs, colors, strict=False):
            umap_embs = get_umap_embeddings(df.loc[df["label"] == lab, :], **kwargs)
            ax.plot(
                umap_embs["UMAP 1"], umap_embs["UMAP 2"], "o", color=color, alpha=0.6, label=lab
            )
            ax.title.set_text(lab)

        for ax in axs[len(unique_labels) :]:
            ax.axis("off")

    if by_class:
        _plot_embeds_by_class(df, axs=ax, figsize=figsize, **kwargs)
    else:
        _plot_embeds(df, ax=ax, **kwargs)

    sns.despine()
    plt.tight_layout()


def plot_neighbor_swc(
    df: pd.DataFrame,
    neuron_name: str,
    swc_path: Path | str,
    num_neighbors: int,
    closest: bool = True,
    within_class: bool = False,
) -> None:
    # Find neighbors
    neighbors = get_similar_neurons(
        df=df,
        target_sample=neuron_name,
        k=num_neighbors,
        closest=closest,
        within_class=within_class,
    )
    swc_neuron = SWCData(f"{swc_path}/{neuron_name}.swc", standardize=False, align=False)
    swc_neighbors = {
        neuron: SWCData(f"{swc_path}/{neuron}.swc", standardize=False, align=False)
        for neuron in neighbors
    }

    # # Determine grid dimensions
    cols_for_neighbors = min(3, num_neighbors)  # 3 cols max for neighbors subplot
    # Calculate number of rows needed for neighbors
    rows_for_neighbors = (num_neighbors + cols_for_neighbors - 1) // cols_for_neighbors

    fig = plt.figure(figsize=(14, max(6, 2 * rows_for_neighbors)))
    gs = gridspec.GridSpec(rows_for_neighbors, 4, figure=fig)
    ax_target = fig.add_subplot(gs[:, 0])
    swc_neuron.view(ax=ax_target)
    ax_target.set_axis_off()
    ax_target.set_title(f"Target Neuron: \n{neuron_name}")

    axs_neighbors = [fig.add_subplot(gs[i // 3, i % 3 + 1]) for i in range(num_neighbors)]
    for ax, (_neuron, data) in zip(axs_neighbors, swc_neighbors.items(), strict=False):
        data.view(ax=ax)
        ax.set_axis_off()
        # ax.set_title(_neuron)

    plt.tight_layout()
    plt.show()
