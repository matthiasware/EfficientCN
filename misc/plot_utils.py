import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_mat(mat, row_names=None, col_names=None, scale_factor=2, title=None, xlabel=None, ylabel=None, p_file=None, vmin=None, vmax=None):
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(n_cols * scale_factor,
                                    n_rows * scale_factor))
    im = ax.imshow(mat, cmap="copper", vmin=vmin, vmax=vmax)
    #
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    #

    if row_names is not None:
        assert len(row_names) == n_rows
        ax.set_yticklabels(row_names)
    if col_names is not None:
        assert len(col_names) == n_cols
        ax.set_xticklabels(col_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            text = ax.text(col_idx, row_idx, "{:.2f}".format(mat[row_idx, col_idx]),
                           ha="center", va="center", color="w")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    if p_file is not None:
        plt.savefig(p_file)
    plt.show()

def plot_mat2(mat, row_names=None, col_names=None, scale_factor=2, title=None, xlabel=None, ylabel=None, p_file=None, vmin=None, vmax=None, ax=None):
    n_rows, n_cols = mat.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(n_cols * scale_factor,
                                        n_rows * scale_factor))
    im = ax.imshow(mat, cmap="copper", vmin=vmin, vmax=vmax)
    #
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    #

    if row_names is not None:
        assert len(row_names) == n_rows
        ax.set_yticklabels(row_names)
    if col_names is not None:
        assert len(col_names) == n_cols
        ax.set_xticklabels(col_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            text = ax.text(col_idx, row_idx, "{:.2f}".format(mat[row_idx, col_idx]),
                           ha="center", va="center", color="w")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if ax is None:
        fig.tight_layout()
        if p_file is not None:
            plt.savefig(p_file)
        plt.show()


def imshow(img, cmap="gray"):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="cmap")
    plt.show()


def plot_couplings(couplings: List[np.ndarray], scale_factor=2, title: str=None, ax=None, show=True):
    """
        couplings     List of coupling coefficients of shapes (hi,wi) for i=1 to n
                      List of 2D arrays
        scale_factor  figure scale factor
        title         title of plot
    """
    if ax is None:
        figsize=(len(couplings) * scale_factor, len(couplings)*scale_factor)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for idx in range(1, len(couplings) + 1):
        C = couplings[idx - 1]
        assert np.all(C < 1 + 1e-4)
        C = np.minimum(C, 1)
        nl, nh = C.shape
        ax.scatter(range(nl), np.ones(nl) * idx, c=[plt.cm.prism(i) for i in range(nl)])
        ax.scatter(range(nh), np.ones(nh) * idx + 1, c=[plt.cm.prism(i) for i in range(nh)])
        for l in range(nl):
            for h in range(nh):
                c = C[l][h]
                ax.plot((l,h), (idx, idx+1), c="blue", alpha=c)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()

def plot_capsules(capsules: List[np.ndarray], title=None, scale_factor=2, ax=None, show=True):
    """128
        capsules:  list of capsules activations -> 1D arrays
    """
    if ax is None:
        figsize=(len(capsules) * scale_factor, len(capsules)*scale_factor)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for idx in range(len(capsules)):
        u = capsules[idx]
        ax.scatter(range(len(u)), np.ones(len(u)) * idx)
        for ui_idx, ui in enumerate(u):
            ax.plot((ui_idx, ui_idx), (idx, idx + ui), c=plt.cm.prism(ui_idx))
    
    #ax.axis("off")
    ax.set_xticks(list(range(max(len(c) for c in capsules))))
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()