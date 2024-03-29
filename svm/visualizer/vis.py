import logging
import pprint
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from svm.utils import (
    generate_linear_separable_dataset,
    generate_linear_separable_dataset_overlap,
    generate_nonlinear_separable_dataset,
    generate_nonlinear_separable_dataset_2,
)
from svm.visualizer.helper import (
    Param,
    change_parameters,
    generate_data_then_plot,
    infer,
    plot_contours,
    plot_scatter,
    plot_svc_decision_function,
    redraw,
    reset,
)

logging.getLogger("matplotlib.font_manager").disabled = True

button_dict = {1: "left_click", 3: "right_click"}
key_dict = {
    "f1": "debug",
    "r": "reset",
    "R": "redraw",
    "p": "save",
    "2": "create_linear_separable",
    "3": "create_nonlinear_separable_1",
    "4": "create_nonlinear_separable_2",
    "5": "create_linear_separable_overlap",
    " ": "infer",
    "f2": "toggle_kernels",
    "f3": "toggle_models",
    "c": "change_parameters",
    "/": "show_help",
}


def onclick(event, fig, ax, X, Y, plot_param: Param):
    if not plot_param.allow_press:
        print("You can't add more data points. Press R to restart.")
        return
    if event.xdata is None and event.ydata is None:
        return

    point_opt = "bo"
    y = -1
    if button_dict.get(event.button) == "right_click":
        point_opt = "rx"
        y = 1
    X.append([event.xdata, event.ydata])
    Y.append(y)
    plt.plot(event.xdata, event.ydata, point_opt)
    fig.canvas.draw()


def onpress(event, fig, ax, X, Y, plot_param: Param):
    key = event.key
    pressed_key = key_dict.get(key)
    if pressed_key == "debug":
        print(np.array(X).shape)
        print(np.array(Y).shape)
        print("Model:", plot_param.model_list[plot_param.model_idx])
        print("Kernel:", plot_param.kernel_list[plot_param.kernel_idx])
        print("Model parameters:")
        pprint.pprint(plot_param.get_model_params())
    elif pressed_key == "reset":
        reset(fig, ax, X, Y)
        plot_param.allow_press = True
        plot_param.allow_infer = True
        plot_param.showed_tooltip = False
    elif pressed_key == "redraw":
        if not X or not Y:
            reset(fig, ax, X, Y)
        else:
            redraw(fig, ax, np.array(X), Y)
        plot_param.allow_press = True
        plot_param.allow_infer = True
        plot_param.showed_tooltip = False
    elif pressed_key == "save":
        print("Saving...")
        Xs = np.array(X)
        Ys = np.array(Y)
        np.save("X_non_linear_separable", Xs)
        np.save("Y_non_linear_separable", Ys)
        plt.savefig("fig.png")
        print("File saved at the relative folder.")
    elif pressed_key == "create_linear_separable":
        generate_data_then_plot(
            fig, ax, X, Y, plot_param, generate_linear_separable_dataset
        )
    elif pressed_key == "create_nonlinear_separable_1":
        generate_data_then_plot(
            fig,
            ax,
            X,
            Y,
            plot_param,
            partial(generate_nonlinear_separable_dataset, mean=[5.0, 5.0]),
        )
    elif pressed_key == "create_nonlinear_separable_2":
        generate_data_then_plot(
            fig,
            ax,
            X,
            Y,
            plot_param,
            generate_nonlinear_separable_dataset_2,
        )
    elif pressed_key == "create_linear_separable_overlap":
        generate_data_then_plot(
            fig,
            ax,
            X,
            Y,
            plot_param,
            generate_linear_separable_dataset_overlap,
        )
    elif pressed_key == "infer":
        if not X or not Y:
            print("No data to infer.")
            return
        elif not plot_param.allow_infer:
            print("You've already infered.")
            return
        plot_param.allow_press = False
        plot_param.allow_infer = False
        plot_param.showed_tooltip = False

        # model = SVC(kernel=plot_param.get_kernel(), C=1e10)
        model = plot_param.get_model()(**plot_param.get_model_params())
        infer(fig, ax, model, X, Y, plot_param)
    elif pressed_key == "toggle_kernels":
        plot_param.kernel_idx = (plot_param.kernel_idx + 1) % len(
            plot_param.kernel_list
        )
        kernel = plot_param.get_kernel()
        print("Current kernel:", kernel)
    elif pressed_key == "toggle_models":
        plot_param.model_idx = (plot_param.model_idx + 1) % len(
            plot_param.model_list
        )
        model = plot_param.get_model()
        print("Current model:", model)
    elif pressed_key == "show_help":
        if plot_param.showed_tooltip:
            return
        s = ""
        for i, (k, v) in enumerate(key_dict.items()):
            k = k if k != " " else "<space>"
            s += k + ": " + v
            if (i + 1) % 4 == 0:
                s += "\n"
            else:
                s += "; "

        plt.text(
            x=0.0,
            y=0.02,
            s=s,
            fontsize=10,
            ha="left",
            transform=fig.transFigure,
        )
        plot_param.showed_tooltip = True
        fig.canvas.draw()
    elif pressed_key == "change_parameters":
        change_parameters(plot_param)


def create_data_then_infer():
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_param = Param()

    ax.set_xlim(plot_param.ax_x_lim)
    ax.set_ylim(plot_param.ax_y_lim)

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
