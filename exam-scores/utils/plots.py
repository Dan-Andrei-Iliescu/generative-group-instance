import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np


def plot_2D_groups(x, title, xaxis, yaxis, fname):
    fig = go.Figure()
    idx = 0
    for x_group in x:
        fig.add_trace(go.Scatter(
            x=x_group[:, 0], y=x_group[:, 1],
            name=f"Group {idx + 1}", mode='markers'))
        idx += 1

    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        legend_title="Group",
        width=1000,
        height=1000
    )
    fig.show()


def plot_1D_latent(x, title, fname):
    fig = go.Figure()
    colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    idx = 0
    for x_group in x:
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
        legend_title="Groups",
        width=1000,
        height=1000,
        barmode='stack'
    )
    fig.show()


def plot_1D_trans(x, y, trans, title, fname):
    fig = go.Figure()
    colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    fig.add_trace(go.Box(
        x=x[:, 0], name="Group A",
        legendgroup="0", showlegend=True,
        marker=dict(color=colours[0])))
    for (a, b) in zip(x, trans):
        fig.add_trace(go.Scatter(
            x=[a[0], b[0]], y=[0, 1],
            legendgroup="2", showlegend=False, mode="lines+markers",
            marker_line_color="gray", line_color="gray"))
    fig.add_trace(go.Box(
        x=trans[:, 0], name="Translation A->B",
        legendgroup="2", showlegend=True,
        marker=dict(color=colours[2])))
    fig.add_trace(go.Box(
        x=y[:, 0], name="Group B",
        legendgroup="1", showlegend=True,
        marker=dict(color=colours[1])))
    fig.add_trace(go.Scatter(
        x=y[:, 0], y=[2 for _ in trans],
        legendgroup="1", showlegend=False, mode="markers",
        marker_line_color=colours[1],
        marker_symbol="line-ns", marker_line_width=2))

    fig.update_xaxes(title_text="X Values")
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        legend_title="Groups",
        width=1000,
        height=1000,
        barmode='stack'
    )
    fig.show()


def plot_1D_rec(x, y, title, fname):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        subplot_titles=["Ground Truth", "Reconstruction"],
        vertical_spacing=0.05)
    colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

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
        legend_title="Groups",
        width=1000,
        height=1000,
        barmode='stack'
    )
    fig.show()
