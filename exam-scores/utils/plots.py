import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os


def compute_colours():
    # colours
    alphas = [1., 0.3]
    colours = []
    for alpha in alphas:
        colours.append([f'rgba(255, 190, 11, {alpha})',
                        f'rgba(251, 86, 7, {alpha})',
                        f'rgba(202, 0, 87, {alpha})',
                        f'rgba(131, 56, 236, {alpha})',
                        f'rgba(58, 134, 255, {alpha})',
                        f'rgba(70, 190, 190, {alpha})',
                        f'rgba(119, 191, 25, {alpha})'])
    return colours


def plot_1D_latent(x, title, result_path):
    fig = go.Figure()
    colours = compute_colours()[0]

    idx = 0
    for x_group in x[:32]:
        idcs = [idx for _ in x_group]
        fig.add_trace(go.Box(
            x=x_group[:, 0], name=f"Group {idx + 1}",
            legendgroup=f"{idx}", showlegend=True,
            marker=dict(color=colours[idx % len(colours)])))
        fig.add_trace(go.Scatter(
            x=x_group[:, 0], y=idcs,
            legendgroup=f"{idx}", showlegend=False, mode="markers",
            marker_line_color=colours[idx % len(colours)],
            marker_symbol="line-ns", marker_line_width=2))
        idx += 1

    fig.update_xaxes(title_text="V Values")
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=title,
        legend_title="Groups",
        width=1200,
        height=1200,
        barmode='stack'
    )
    fig.write_image(result_path + "_latent.svg")


def plot_1D_trans(x, y, trans, title, result_path):
    num_rows = 3
    num_cols = 4
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        vertical_spacing=0.01, horizontal_spacing=0.01,
        shared_xaxes=True, shared_yaxes=True)
    colours = compute_colours()[0]

    ok1 = True
    ok2 = True
    ok3 = True
    for row in range(num_rows):
        for col in range(num_cols):
            group_idx = num_cols * row + col

            fig.add_trace(go.Box(
                x=x[group_idx][:, 0], name="Group A",
                legendgroup="0", showlegend=ok1,
                marker=dict(color=colours[0])), row=row+1, col=col+1)
            ok1 = False
            for (a, b) in zip(x[group_idx], trans[group_idx]):
                fig.add_trace(go.Scatter(
                    x=[a[0], b[0]], y=[0, 1],
                    legendgroup="2", showlegend=False, mode="lines+markers",
                    marker_line_color="gray", line_color="gray"),
                    row=row+1, col=col+1)
                fig.add_trace(go.Box(
                    x=trans[group_idx][:, 0], name="Translation A->B",
                    legendgroup="2", showlegend=ok2,
                    marker=dict(color=colours[2])),
                    row=row+1, col=col+1)
                ok2 = False
                fig.add_trace(go.Box(
                    x=y[group_idx][:, 0], name="Group B",
                    legendgroup="1", showlegend=ok3,
                    marker=dict(color=colours[1])),
                    row=row+1, col=col+1)
                ok3 = False
                fig.add_trace(go.Scatter(
                    x=y[group_idx][:, 0], y=[2 for _ in trans[group_idx]],
                    legendgroup="1", showlegend=False, mode="markers",
                    marker_line_color=colours[1],
                    marker_symbol="line-ns", marker_line_width=2),
                    row=row+1, col=col+1)

    # fig.update_xaxes(title_text="X Values")
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=title,
        width=1800,
        height=1200,
        legend_title="Groups",
        barmode='stack'
    )
    fig.write_image(result_path + "_trans.svg")


def plot_1D_rec(x, y, title):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        subplot_titles=["Ground Truth", "Reconstruction"],
        vertical_spacing=0.05)
    colours = compute_colours()[0]

    idx = 0
    for x_group, y_group in zip(x, y):
        idcs = [idx for _ in x_group]
        fig.add_trace(go.Box(
            x=x_group[:, 0], name=f"Group {idx+1}",
            legendgroup=f"{idx}", showlegend=True,
            marker=dict(color=colours[idx % len(colours)])
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_group[:, 0], y=idcs,
            legendgroup=f"{idx}", showlegend=False, mode="markers",
            marker_line_color=colours[idx % len(colours)],
            marker_symbol="line-ns", marker_line_width=2
        ), row=1, col=1)
        fig.add_trace(go.Box(
            x=y_group[:, 0],
            legendgroup=f"{idx}", showlegend=False,
            marker=dict(color=colours[idx % len(colours)])
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=y_group[:, 0], y=idcs,
            legendgroup=f"{idx}", showlegend=False, mode="markers",
            marker_line_color=colours[idx % len(colours)],
            marker_symbol="line-ns", marker_line_width=2
        ), row=2, col=1)
        idx += 1

    fig.update_xaxes(title_text="X Values", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(
        title=title,
        legend_title="Groups",
        width=1200,
        height=1200,
        barmode='stack'
    )
    fig.show(renderer="firefox")


def moving_avg(a, n):
    s = np.cumsum(a)
    s[n:] = s[n:] - s[:-n]
    s = s[n-1:] / n
    return s


def plot_results(test_dict, result_dir):
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        subplot_titles=["Reconstruction Error", "Translation Error",
                        "Error of Mean of Group of Instance Variables",
                        "Error of Variance of Group of Instance Variables"],
        vertical_spacing=0.05)
    colours = compute_colours()

    model_names = list(test_dict.keys())
    for idx in range(len(model_names)):
        model_name = model_names[idx]
        plot_names = list(test_dict[model_name].keys())
        rows = [1, 1, 2, 2]
        cols = [1, 2, 1, 2]

        is_showlegend = True
        for plot_name, row, col in zip(plot_names, rows, cols):
            epochs = list(test_dict[model_name][plot_name].keys())

            error_med = []
            error_lo = []
            error_hi = []
            for epoch in epochs:
                runs = test_dict[model_name][plot_name][epoch]
                error_med.append(np.mean(runs))
                error_lo.append(np.mean(np.sort(
                    [np.mean(run) for run in runs])[:2]))
                error_hi.append(np.mean(np.sort(
                    [np.mean(run) for run in runs])[1:]))
            n = 5
            error_med = moving_avg(error_med, n)
            error_lo = moving_avg(error_lo, n)
            error_hi = moving_avg(error_hi, n)

            fig.add_trace(go.Scatter(
                x=epochs,
                y=error_med,
                legendgroup=model_name, showlegend=is_showlegend,
                mode="lines+markers",
                marker=dict(color=colours[0][idx % len(colours[0])]),
                name=model_name
            ), row=row, col=col)
            is_showlegend = False
            fig.add_trace(go.Scatter(
                x=epochs,
                y=error_lo,
                legendgroup=model_name, showlegend=is_showlegend,
                mode="lines", line=dict(width=0)
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=epochs,
                y=error_hi,
                legendgroup=model_name, showlegend=is_showlegend,
                mode="lines", line=dict(width=0),
                fillcolor=colours[1][idx % len(colours[0])],
                fill='tonexty'
            ), row=row, col=col)

    fig.update_yaxes(type='log')
    fig.update_xaxes(title_text="Epochs", row=2)
    fig.update_layout(
        legend_title="Conditions",
        width=1800,
        height=1200
    )
    fig.write_image(os.path.join(result_dir, "results.svg"))
