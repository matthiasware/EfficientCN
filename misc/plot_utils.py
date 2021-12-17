import matplotlib.pyplot as plt
import numpy as np


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


def imshow(img, cmap="gray"):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="cmap")
    plt.show()