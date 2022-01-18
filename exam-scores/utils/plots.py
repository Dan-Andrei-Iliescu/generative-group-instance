import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
import time
import pandas as pd

NUM_GROUPS_PER_PLOT = 16
FIG_SIZE = 800
BIG_FIG_SIZE = 1000


def compute_colours(palette=None):
    # colours
    alphas = [1., 0.3]
    colours = []
    for alpha in alphas:
        if palette == "results":
            colours.append([f'rgba(200, 85, 61, {alpha})',
                            f'rgba(45, 48, 71, {alpha})',
                            f'rgba(45, 48, 71, {alpha})',
                            f'rgba(45, 48, 71, {alpha})'])
        elif palette == "ablation":
            colours.append([f'rgba(147, 183, 190, {alpha})',
                            f'rgba(45, 48, 71, {alpha})',
                            f'rgba(45, 48, 71, {alpha})',
                            f'rgba(200, 85, 61, {alpha})',
                            f'rgba(88, 139, 139, {alpha})',
                            f'rgba(88, 139, 139, {alpha})'])
        else:
            colours.append([f'rgba(255, 190, 11, {alpha})',
                            f'rgba(251, 86, 7, {alpha})',
                            f'rgba(202, 0, 87, {alpha})',
                            f'rgba(131, 56, 236, {alpha})',
                            f'rgba(58, 134, 255, {alpha})',
                            f'rgba(70, 190, 190, {alpha})',
                            f'rgba(119, 191, 25, {alpha})'])
    return colours


def colour_assign(model_name):
    colour_dict = {
        'Change regularization': 'rgb(45, 48, 71)',
        'Change instance conditioning': 'rgb(200, 85, 61)',
        'Change group encoder': 'rgb(88, 139, 139)',
        'CxVAE': 'rgb(200, 85, 61)',
        'Other SOTA': 'rgb(45, 48, 71)'
    }
    if model_name in colour_dict.keys():
        return colour_dict[model_name]
    return colour_dict['def']


def group_assign(model_name):
    name_dict = {
        'True_ours_None': 'CxVAE',
        'True_nemeth_None': 'Change regularization',
        'True_None_None': 'Change regularization',
        'False_ours_None': 'Change instance conditioning',
        'True_ours_mul': 'Change group encoder',
        'True_ours_med': 'Change group encoder',
        'False_None_mul': 'Other SOTA',
        'False_None_med': 'Other SOTA',
        'False_nemeth_med': 'Other SOTA'
    }
    if model_name in name_dict.keys():
        return name_dict[model_name]
    return model_name


def name_assign(model_name):
    name_dict = {
        'True_ours_None': 'CxVAE',
        'True_nemeth_None': 'A',
        'True_None_None': 'B',
        'False_ours_None': 'C',
        'True_ours_mul': 'D',
        'True_ours_med': 'E',
        'False_None_mul': 'ML-VAE',
        'False_None_med': 'GVAE',
        'False_nemeth_med': 'GVAE+Reg'
    }
    if model_name in name_dict.keys():
        return name_dict[model_name]
    return model_name


def plot_1D_data(x, title, result_path):
    fig = go.Figure()
    colours = compute_colours()[0]

    idx = 0
    for x_group in x[:NUM_GROUPS_PER_PLOT]:
        idcs = [idx for _ in x_group]
        fig.add_trace(go.Box(
            x=x_group[:, 0], name=f"School {idx + 1}",
            legendgroup=f"{idx}", showlegend=True,
            marker=dict(color=colours[idx % len(colours)])))
        fig.add_trace(go.Scatter(
            x=x_group[:, 0], y=idcs,
            legendgroup=f"{idx}", showlegend=False, mode="markers",
            marker_line_color=colours[idx % len(colours)],
            marker_symbol="line-ns", marker_line_width=2))
        idx += 1

    fig.update_xaxes(title_text="Exam Scores")
    fig.update_yaxes(showticklabels=False, title_text="Schools")
    fig.update_layout(
        title=title,
        legend_title="Groups",
        width=FIG_SIZE,
        height=FIG_SIZE,
        barmode='stack'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="right",
        x=1
    ))
    fig.write_image(result_path + ".svg")


def plot_1D_latent(x, title, result_path):
    fig = go.Figure()
    colours = compute_colours()[0]

    idx = 0
    for x_group in x[:NUM_GROUPS_PER_PLOT]:
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
        width=FIG_SIZE,
        height=FIG_SIZE,
        barmode='stack'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="right",
        x=1
    ))
    fig.write_image(result_path + "_latent.svg")


def plot_1D_trans(x, y, trans, title, result_path):
    num_rows = 2
    num_cols = 2
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
        width=FIG_SIZE,
        height=FIG_SIZE,
        legend_title="Groups",
        barmode='stack'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="right",
        x=1
    ))
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
    a[n-1:] = s[n-1:] / n
    return a


def plot_results(df, result_dir, palette):
    titles = ["a) Reconstruction Error", "b) Translation Error",
              "c) U Prediction Error",
              "d) V Prediction Error"]
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=titles,
        vertical_spacing=0.05, horizontal_spacing=0.05)

    df = df.loc[df['epoch'] >= 25]

    model_names = pd.unique(df['model_name'])
    print(model_names)

    rows = [1, 1, 1, 1]
    cols = [1, 2, 3, 4]
    test_names = ['rec_error', 'trans_error', 'u_error', 'v_error']
    seen = []
    for plt_idx in range(4):
        test_df = df.loc[df['test_name'] == test_names[plt_idx]]
        group_df = test_df.groupby('model_name')['value'].mean()\
            .reset_index().sort_values(by=['value'])
        model_names = group_df['model_name']
        for model_name in model_names:
            model_df = test_df.loc[test_df['model_name'] == model_name]
            fig.add_trace(go.Box(
                x=model_df['model_name'].apply(name_assign),
                y=model_df['value'],
                marker_color=colour_assign(group_assign(model_name)),
                notched=True,
                name=group_assign(model_name),
                legendgroup=group_assign(model_name),
                showlegend=(group_assign(model_name) not in seen)
            ), row=rows[plt_idx], col=cols[plt_idx])
            seen.append(group_assign(model_name))
        fig.add_trace(go.Scatter(
            x=group_df['model_name'].apply(name_assign),
            y=group_df['value'],
            mode='lines+markers',
            marker_color='green',
            name='Mean error',
            showlegend=("green" not in seen)
        ), row=rows[plt_idx], col=cols[plt_idx])
        seen.append("green")

    fig.update_yaxes(title_text='Error (log MSE)', col=1)
    fig.update_xaxes(row=1)
    fig.update_layout(
        legend_title="Legend",
        width=BIG_FIG_SIZE,
        height=FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0
    ))
    fig['layout'].update(margin=dict(l=0, r=0, b=0))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)

    fig.write_image(os.path.join(result_dir, "results.pdf"))
    time.sleep(2)
    fig.write_image(os.path.join(result_dir, "results.pdf"))


def violin_plot(test_dict, result_dir, palette):
    titles = ["a) Reconstruction Error", "b) Translation Error",
              "c) U Prediction Error",
              "d) V Prediction Error"]
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=titles,
        vertical_spacing=0.05, horizontal_spacing=0.05)
    colours = compute_colours(palette)

    rows = [1, 1, 1, 1]
    cols = [1, 2, 3, 4]
    cond_names = list(test_dict.keys())
    totals = np.zeros([len(cond_names), ])
    for idx in range(len(cond_names)):
        cond_name = cond_names[idx]
        test_names = list(test_dict[cond_name].keys())

        total = 0
        for test_name, row, col in zip(test_names, rows, cols):
            epochs = list(test_dict[cond_name][test_name].keys())
            errors = []
            n = 20
            for epoch in epochs[-n:]:
                runs = test_dict[cond_name][test_name][epoch]
                for run in runs:
                    errors.append(np.mean(run))
            total += np.mean(np.array(errors))
        totals[idx] = total

    p = np.argsort(totals)
    s = np.empty(p.size, dtype=np.int32)
    for i in np.arange(p.size):
        s[p[i]] = i

    cond_names = list(test_dict.keys())
    for idx in range(len(cond_names)):
        cond_name = cond_names[idx]
        test_names = list(test_dict[cond_name].keys())
        rows = [1, 1, 1, 1]
        cols = [1, 2, 3, 4]

        is_showlegend = False
        for test_name, row, col in zip(test_names, rows, cols):
            epochs = list(test_dict[cond_name][test_name].keys())
            errors = []
            n = 20
            for epoch in epochs[-n:]:
                runs = test_dict[cond_name][test_name][epoch]
                for run in runs:
                    errors.append(np.log(np.mean(run)))
            errors = np.array(errors)

            fig.add_trace(go.Box(
                x=s[idx]*np.ones_like(errors),
                y=errors,
                marker=dict(color=colours[0][idx % len(colours[0])]),
                name=idx+1,
                legendgroup=idx+1,
                showlegend=is_showlegend,
                notched=True
            ), row=row, col=col)
            is_showlegend = False

    fig.update_yaxes(title_text='Error (log MSE)', col=1)
    fig.update_xaxes(title_text="Model", row=1)
    fig.update_layout(
        legend_title="Models",
        width=BIG_FIG_SIZE,
        height=FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="right",
        x=1
    ))
    fig['layout'].update(margin=dict(l=0, r=0, b=0))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)

    fig.write_image(os.path.join(result_dir, "results.pdf"))
    time.sleep(2)
    fig.write_image(os.path.join(result_dir, "results.pdf"))
