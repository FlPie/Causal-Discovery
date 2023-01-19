import io

import pandas as pd
import numpy as np
import PIL

from tqdm import tqdm

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.cm as cm

import torch
import torch_geometric as pyg
import pytorch_lightning as pl


def get_graph_figure(G1: nx.Graph, G2: nx.Graph, **kwargs) -> figure:
    """Returns a figure of NetworkX graph with intersectioned edges & reversed edges drawn.

    Args:
        G1 (nx.Graph): Graph to draw.
        G2 (nx.Graph): Graph to compare with. (Ground truth)

    Returns:
        figure: Matplotlib figure.Figure with graph drawn.
    """
    options = {
        "prog": "circo",
        "graph": {
            "font_size": 15,
            "node_size": 2000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
            "with_labels": True,
        },
        "intersectioned": {
            "edge_color": "green",
            "width": 8,
            "alpha": 0.3,
        },
        "reversed": {
            "edge_color": "orange",
            "width": 8,
            "alpha": 0.3,
        }
    }
    options.update(kwargs)

    ax = plt.subplot()

    if G2 is not None:
        I = nx.intersection(G1, G2)
        R = nx.intersection(G1, nx.reverse(G2))

        pos = nx.nx_agraph.graphviz_layout(G2, prog=options.get("prog"))

        nx.draw_networkx_edges(I, pos, ax=ax, **options.get("intersectioned"))
        nx.draw_networkx_edges(R, pos, ax=ax, **options.get("reversed"))
    else:
        pos = nx.nx_agraph.graphviz_layout(G1, prog=options.get("prog"))

    nx.draw(G1, pos, ax=ax, **options.get("graph"))

    fig = ax.get_figure()
    plt.close()

    return fig


def get_adj_figure(adj_A, **kwargs) -> figure:
    """Returns a figure of adjacency matrix.

    Args:
        adj_A (pd.DataFrame or np.ndarray): Adjacency matrix to plot.

    Returns:
        figure: Plot of adjacency matrix.
    """
    options = {
        "vmin": -1,
        "vmax": 1,
        "cmap": cm.RdBu_r,
    }
    options.update(kwargs)
    
    ax = plt.subplot()

    plt.imshow(adj_A, **options)
    plt.colorbar()

    fig = ax.get_figure()
    plt.close()

    return fig


def fig2img(fig: figure) -> PIL.Image:
    """Converts matplotlib figure to PIL image.

    Args:
        fig (figure): Matplotlib figure.Figure.

    Returns:
        PIL.Image: PIL image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def figure2image(fig: figure) -> PIL.Image:
    """Converts matplotlib figure to PIL image.

    Args:
        fig (figure): Matplotlib figure.Figure.

    Returns:
        PIL.Image: PIL image.
    """
    return PIL.Image.frombytes("RGB",
                               fig.canvas.get_width_height(),
                               fig.canvas.tostring_rgb())


def get_plot_imgs(graph, G: nx.Graph, ground_truth_G: nx.Graph):
    graph_fig = get_graph_figure(G, ground_truth_G)
    graph_img = fig2img(graph_fig)

    adj_fig = get_adj_figure(graph)
    adj_img = fig2img(adj_fig)

    return graph_img, adj_img


def count_accuracy(
    G_true: nx.DiGraph,
    G: nx.DiGraph,
    G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.
    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr':fdr, 'tpr':tpr, 'fpr':fpr, 'shd':shd, 'pred_size':pred_size}