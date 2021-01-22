from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

from svm.utils import (
    generate_linear_separable_dataset,
    generate_nonlinear_separable_dataset,
)

button_dict = {1: "left_click", 3: "right_click"}
key_dict = {
    "f1": "debug",
    "r": "reset",
    "R": "redraw",
    "2": "create_linear_separable",
    "3": "create_nonlinear_separable",
    " ": "infer",
    "f2": "toggle_kernels",
    "/": "toggle_help",
}
kernel_list = ["linear", "poly", "rbf", "sigmoid"]


class Param:
    """Param class
    This class contains all the parameters involved in the plotting process.
    In order to minimize the number of params pass to the callback.
    """

    def __init__(self):
        self.kernel_idx = 0
        self.allow_press = True
        self.allow_infer = True
        self.show_tooltip = False


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the contour of the meshgrid after infer.

    Args:
        ax (Axes): the ax of the figure.
        clf (SVM): SVM model
        xx (array): The coordinates of the x-axis.
        yy (array): The coordinates of the y-axis.

    Returns:
        [type]: [description]
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)


def make_meshgrid(xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, h=0.02):
    """Make meshgrid for SVM color plot given a region.

    Args:
        x_min (float, optional): x_limit. Defaults to 0..
        x_max (float, optional): x_limit. Defaults to 10..
        y_min (float, optional): y_limit. Defaults to 0..
        y_max (float, optional): y_limit. Defaults to 10..
        h (float, optional): step size for each grid. Defaults to 0.02.

    Returns:
        xx, yy (np.array, np.array): meshgrid
    """
    x_min, x_max = xmin - 1, xmax + 1
    y_min, y_max = ymin - 1, ymax + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    return xx, yy


def init_ax(fig, ax, refresh=True):
    """Clear the axes and redraw

    Args:
        refresh: True to refresh the plot immediately.
    """
    ax.clear()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    if refresh:
        fig.canvas.draw()


def reset(fig, ax, X, Y):
    """Remove all points and clear the plot"""
    init_ax(fig, ax)
    X.clear()
    Y.clear()


def redraw(fig, ax, X, Y, refresh=True):
    """Redraw the plot with the given data points."""
    init_ax(fig, ax)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        marker="o",
        c=Y,
        s=25,
        edgecolor="k",
    )
    if refresh:
        fig.canvas.draw()


def plot_svc_decision_function(model, ax, plot_support=True):
    """Plot the SVM decision function and bold the support vectors

    Args:
        model (SVM): The model to classify
        ax (Axes): ax of the figure
        plot_support (bool, optional): True to bold the support vectors. Defaults to True.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(
        X,
        Y,
        P,
        # colors="k",
        levels=[-1, 0, 1],
        alpha=1,
        linestyles=["--", "-", "--"],
    )

    # plot support vectors
    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=300,
            linewidth=1,
            facecolors="none",
            edgecolor="black",
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def infer(fig, ax, X, Y, plot_param):
    model = SVC(kernel=kernel_list[plot_param.kernel_idx], C=1e10)
    print("Fitting....")
    model.fit(X, Y)
    print("Plotting....")
    xx, yy = make_meshgrid()

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plot_svc_decision_function(model, ax)
    fig.canvas.draw()


def onclick(event, fig, ax, X, Y, plot_param):
    if not plot_param.allow_press:
        print("You can't add more data points. Press R to restart.")
        return
    if event.xdata is None and event.ydata is None:
        return

    point_opt = "bo"
    y = 1
    if button_dict.get(event.button) == "right_click":
        point_opt = "rx"
        y = -1
    X.append([event.xdata, event.ydata])
    Y.append(y)
    plt.plot(event.xdata, event.ydata, point_opt)
    fig.canvas.draw()


def onpress(event, fig, ax, X, Y, plot_param):
    key = event.key
    pressed_key = key_dict.get(key)
    if pressed_key == "debug":
        print(X, Y)
    elif pressed_key == "reset":
        reset(fig, ax, X, Y)
        plot_param.allow_press = True
        plot_param.allow_infer = True
        plot_param.show_tooltip = False
    elif pressed_key == "redraw":
        if not X or not Y:
            reset(fig, ax, X, Y)
        else:
            redraw(fig, ax, np.array(X), Y)
        plot_param.allow_press = True
        plot_param.allow_infer = True
        plot_param.show_tooltip = False
    elif pressed_key == "create_linear_separable":
        reset(fig, ax, X, Y)
        plot_param.allow_press = False
        plot_param.allow_infer = True
        plot_param.show_tooltip = False

        a, b = generate_linear_separable_dataset()
        X.extend(a)
        Y.extend(b)
        ax.scatter(
            a[:, 0],
            a[:, 1],
            marker="o",
            c=b,
            s=25,
            edgecolor="k",
        )
        fig.canvas.draw()
    elif pressed_key == "create_nonlinear_separable":
        reset(fig, ax, X, Y)
        plot_param.allow_press = False
        plot_param.allow_infer = True
        plot_param.show_tooltip = False

        a, b = generate_nonlinear_separable_dataset(mean=[5.0, 5.0])
        X.extend(a)
        Y.extend(b)
        ax.scatter(
            a[:, 0],
            a[:, 1],
            marker="o",
            c=b,
            s=25,
            edgecolor="k",
        )
        fig.canvas.draw()
    elif pressed_key == "infer":
        if not X or not Y:
            print("No data to infer.")
            return
        elif not plot_param.allow_infer:
            print("You've already infered.")
            return
        plot_param.allow_press = False
        plot_param.allow_infer = False
        plot_param.show_tooltip = False

        infer(fig, ax, X, Y, plot_param)
    elif pressed_key == "toggle_kernels":
        plot_param.kernel_idx = (plot_param.kernel_idx + 1) % len(kernel_list)
        kernel = kernel_list[plot_param.kernel_idx]
        print("Current kernel:", kernel)
    elif pressed_key == "toggle_help":
        if plot_param.show_tooltip:
            return
        s = ""
        for i, (k, v) in enumerate(key_dict.items()):
            k = k if k != " " else "<space>"
            s += k + ": " + v
            if (i + 1) % 3 == 0:
                s += "\n"
            else:
                s += "; "

        plt.text(
            x=0.0,
            y=0.03,
            s=s,
            fontsize=10,
            ha="left",
            transform=fig.transFigure,
        )
        plot_param.show_tooltip = True
        fig.canvas.draw()


def create_data_then_infer():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    plot_param = Param()

    X, Y = [], []
    fig.canvas.mpl_connect(
        "button_press_event",
        partial(onclick, fig=fig, ax=ax, X=X, Y=Y, plot_param=plot_param),
    )
    fig.canvas.mpl_connect(
        "key_press_event",
        partial(onpress, fig=fig, ax=ax, X=X, Y=Y, plot_param=plot_param),
    )
    plt.show()


if __name__ == "__main__":
    create_data_then_infer()
