import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
import time
import pandas as pd

NUM_GROUPS_PER_PLOT = 5
FIG_SIZE = 400
BIG_FIG_SIZE = 400


def colour_assign(model_name):
    colour_dict = {
        'Other regularization': '#5C7065',
        'Other instance conditioning': '#79B473',
        'Other group encoder': '#433E0E',
        'CxVAE': '#FB5607',
        'Other SOTA': '#C0DA16',
        'def': '#000000',
        '0': '#FB5607',
        '1': '#C0DA16',
        '2': '#9B52DE',
        '3': '#FFBE0B',
        '4': '#FF006E',
        '5': '#3A86FF',
        '6': '#6BD08B'
    }
    if model_name in colour_dict.keys():
        return colour_dict[model_name]
    return colour_dict['def']


def group_assign(model_name):
    name_dict = {
        'True_ours_None': 'CxVAE',
        'True_nemeth_None': 'Other regularization',
        'True_None_None': 'CxVAE',
        'False_ours_None': 'Other instance conditioning',
        'True_ours_mul': 'Other group encoder',
        'True_ours_med': 'Other group encoder',
        'False_None_mul': 'Other SOTA',
        'False_None_med': 'Other SOTA',
        'False_nemeth_med': 'Other SOTA',
        'True': 'CxVAE',
        'False': 'Other SOTA'
    }
    if model_name in name_dict.keys():
        return name_dict[model_name]
    return model_name


def name_assign(model_name):
    name_dict = {
        'True_ours_None': 'CxVAE',
        'True_nemeth_None': 'A',
        'True_None_None': 'CxVAE',
        'False_ours_None': 'C',
        'True_ours_mul': 'D',
        'True_ours_med': 'E',
        'False_None_mul': 'AdaGVAE',
        'False_None_med': 'GVAE',
        'False_nemeth_med': 'COCO-FUNIT',
        'True': 'CxVAE',
        'False': 'GVAE'
    }
    if model_name in name_dict.keys():
        return name_dict[model_name]
    return model_name


def shape_assign(x):
    shapes = [
        'x-thin-open', 'circle-open-dot', 'square-open-dot',
        'triangle-up-open-dot', 'diamond-open-dot', 'cross-thin-open']
    return shapes[x % len(shapes)]


def plot_data(others, result_path):
    num_rows = 1
    num_cols = 1
    name = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        vertical_spacing=0.04, horizontal_spacing=0.01)
    for row in range(num_rows):
        for col in range(num_cols):
            group_idx = num_cols * row + col + 0
            for other, idx in zip(others, range(len(others))):
                fig.add_trace(go.Scatter(
                    x=other[group_idx][:, 0], y=other[group_idx][:, 1],
                    legendgroup="1", showlegend=True, name="School "+name[idx],
                    mode="markers",
                    marker_color=colour_assign(str(idx)), marker_size=10,
                    marker_symbol=shape_assign(idx+2)),
                    row=row+1, col=col+1)
    fig.update_xaxes(row=3)
    fig.update_layout(
        width=BIG_FIG_SIZE,
        height=int(BIG_FIG_SIZE),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        bordercolor="Black",
        borderwidth=0.5
    ))
    fig['layout'].update(margin=dict(l=1, r=1, b=0, t=1))
    fig.update_xaxes(showline=True, linewidth=1, showticklabels=False,
                     linecolor='black', mirror=True, title="Maths Test Score")
    fig.update_yaxes(showline=True, linewidth=1, showticklabels=False,
                     linecolor='black', mirror=True, title="Reading Test Score")
    fig.write_image(result_path + "_data.pdf")
    time.sleep(2)
    fig.write_image(result_path + "_data.pdf")


def plot_trans(x, trans, y, others, result_path, is_trans=True):
    num_rows = 1
    num_cols = 1
    name = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        vertical_spacing=0.04, horizontal_spacing=0.01)
    for row in range(num_rows):
        for col in range(num_cols):
            group_idx = num_cols * row + col + 0
            fig.add_trace(go.Histogram2dContour(
                x=y[group_idx][:, 0], y=y[group_idx][:, 1],
                line_color='lightblue', colorscale=["white", "lightblue"],
                showlegend=True, name='Typical School', showscale=False),
                row=row+1, col=col+1)
            if is_trans:
                for a, b in zip(x[group_idx], trans[group_idx]):
                    fig.add_trace(go.Scatter(
                        x=[a[0], b[0]], y=[a[1], b[1]],
                        legendgroup="2", showlegend=False, mode="lines",
                        marker_line_color="gray", line_color="black",
                        line_width=0.5), row=row+1, col=col+1)
            for other, idx in zip(others, range(len(others))):
                fig.add_trace(go.Scatter(
                    x=other[group_idx][:, 0], y=other[group_idx][:, 1],
                    legendgroup="1", showlegend=True, name="School "+name[idx],
                    mode="markers",
                    marker_color=colour_assign(str(idx)), marker_size=10,
                    marker_symbol=shape_assign(idx+2)),
                    row=row+1, col=col+1)
            if is_trans:
                fig.add_trace(go.Scatter(
                    x=trans[group_idx][:, 0], y=trans[group_idx][:, 1],
                    legendgroup="1", showlegend=True, name="A -> Typical",
                    mode="markers", marker_color=colour_assign('def'),
                    marker_size=10, marker_symbol=shape_assign(1)),
                    row=row+1, col=col+1)
    fig.update_xaxes(range=[-5, 5])
    fig.update_yaxes(range=[-5, 5])
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        width=BIG_FIG_SIZE,
        height=BIG_FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        bordercolor="Black",
        borderwidth=0.5
    ))
    fig['layout'].update(margin=dict(l=1, r=1, b=0, t=1))
    fig.update_xaxes(showline=True, linewidth=1, showticklabels=False,
                     linecolor='black', mirror=True, title="Maths Test Score")
    fig.update_yaxes(showline=True, linewidth=1, showticklabels=False,
                     linecolor='black', mirror=True, title="Reading Test Score")
    name = "_trans.pdf" if is_trans else "_data.pdf"
    fig.write_image(result_path + name)
    time.sleep(2)
    fig.write_image(result_path + name)


def moving_avg(a, n):
    s = np.cumsum(a)
    s[n:] = s[n:] - s[:-n]
    a[n-1:] = s[n-1:] / n
    return a


def plot_results(df, result_dir, skip):
    titles = ['a) Reconstruction Error', "b) Translation Error",
              "c) Mutual Information Gap"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        vertical_spacing=0.05, horizontal_spacing=0.04)

    df = df.loc[df['epoch'] > skip]
    model_names = pd.unique(df['model_name'])

    rows = [1, 1, 1, 1]
    cols = [1, 2, 3, 4]
    test_names = ['rec_error', 'trans_error', 'u_error']
    seen = []
    for plt_idx in range(3):
        test_df = df.loc[df['test_name'] == test_names[plt_idx]]
        ascending = False if plt_idx == 3 else True
        """
        group_df = test_df.groupby('model_name')['value'].mean()\
            .reset_index().sort_values(by=['value'], ascending=ascending)
        """
        group_df = test_df.groupby('model_name')['value'].mean().reset_index()
        model_names = group_df['model_name']
        for model_name in model_names:
            model_df = test_df.loc[test_df['model_name'] == model_name]
            fig.add_trace(go.Violin(
                x=model_df['model_name'].apply(name_assign),
                y=model_df['value'],
                marker_color=colour_assign(group_assign(model_name)),
                name=group_assign(model_name),
                legendgroup=group_assign(model_name),
                showlegend=(group_assign(model_name) not in seen)
            ), row=rows[plt_idx], col=cols[plt_idx])
            seen.append(group_assign(model_name))
        """    
        fig.add_trace(go.Scatter(
            x=group_df['model_name'].apply(name_assign),
            y=group_df['value'],
            mode='lines+markers',
            marker_color=colour_assign('def'),
            name='Mean over 100 runs',
            showlegend=("green" not in seen)
        ), row=rows[plt_idx], col=cols[plt_idx])
        seen.append("green")
        """

    # fig.update_xaxes(title_text='Models', col=2)
    fig.update_layout(
        width=3*BIG_FIG_SIZE,
        height=BIG_FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.1,
        xanchor="left",
        x=0.01,
        bordercolor="Black",
        borderwidth=0.5
    ))
    fig['layout'].update(margin=dict(l=0, r=0, b=0, t=30))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)

    fig.write_image(os.path.join(result_dir, "results.pdf"))
    time.sleep(2)
    fig.write_image(os.path.join(result_dir, "results.pdf"))


def plot_uv_ratio(df, result_dir, skip):
    titles = ['a) Reconstruction', "b) Multiple Imputation",
              "c) MIG"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        vertical_spacing=0.05, horizontal_spacing=0.04)

    df = df.loc[df['epoch'] > skip]

    model_names = pd.unique(df['model_name'])
    uv_ratios = pd.unique(df['uv_ratio'])

    rows = [1, 1, 1, 1]
    cols = [1, 2, 3, 4]
    test_names = ['rec_error', 'trans_error', 'u_error', 'v_error']
    seen = []
    for plt_idx in range(3):
        test_df = df.loc[df['test_name'] == test_names[plt_idx]]
        ascending = False if plt_idx == 3 else True
        group_df = test_df.groupby([
            'model_name', 'uv_ratio'
        ])[['value', 'inst_cond']].mean().reset_index()\
            .sort_values(by=['value'], ascending=ascending)
        for model_name in model_names:
            for uv_ratio in uv_ratios:
                model_df = test_df.loc[test_df['model_name'] == model_name]
                name = str(model_df['inst_cond'].values[0])
                model_df = model_df.loc[model_df['uv_ratio'] == uv_ratio]
                fig.add_trace(go.Box(
                    x=model_df['uv_ratio'],
                    y=model_df['value'],
                    notched=False, boxpoints=False,
                    boxmean=True,
                    marker_color=colour_assign(group_assign(name)),
                    name=name_assign(name),
                    legendgroup=group_assign(name),
                    showlegend=(group_assign(name) not in seen)
                ), row=rows[plt_idx], col=cols[plt_idx])
                seen.append(group_assign(name))
        temp_df = group_df.loc[group_df['inst_cond'] == True].sort_values(
            by=['uv_ratio'], ascending=ascending)
        fig.add_trace(go.Scatter(
            x=temp_df['uv_ratio'],
            y=temp_df['value'],
            mode='lines+markers',
            marker_color=colour_assign(group_assign('True')),
            name='Mean over 100 runs',
            showlegend=False
        ), row=rows[plt_idx], col=cols[plt_idx])
        seen.append("green")
        temp_df = group_df.loc[group_df['inst_cond'] == False].sort_values(
            by=['uv_ratio'], ascending=ascending)
        fig.add_trace(go.Scatter(
            x=temp_df['uv_ratio'],
            y=temp_df['value'],
            mode='lines+markers',
            marker_color=colour_assign(group_assign('False')),
            name='Mean over 100 runs',
            showlegend=False
        ), row=rows[plt_idx], col=cols[plt_idx])
    fig.update_xaxes(
        title_text='Ratio of strength between U and V in the generation of the exam-score dataset', col=2)
    fig.update_xaxes(
        tickvals=[0.01, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.99])
    fig.update_layout(
        width=2*BIG_FIG_SIZE,
        height=BIG_FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.12,
        xanchor="left",
        x=0
    ))
    fig['layout'].update(margin=dict(l=0, r=0, b=0, t=1))
    fig.update_xaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)
    fig.write_image(os.path.join(result_dir, "uv_ratio.pdf"))
    time.sleep(2)
    fig.write_image(os.path.join(result_dir, "uv_ratio.pdf"))


def plot_xy_ratio(df, result_dir, skip):
    titles = ['a) Reconstruction Error', "b) Translation Error",
              "c) Mutual Information Gap"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        vertical_spacing=0.05, horizontal_spacing=0.04)

    df = df.loc[df['epoch'] > skip]

    model_names = pd.unique(df['model_name'])
    xy_ratios = pd.unique(df['xy_ratio'])

    rows = [1, 1, 1, 1]
    cols = [1, 2, 3, 4]
    test_names = ['rec_error', 'trans_error', 'u_error', 'v_error']
    seen = []
    for plt_idx in range(3):
        test_df = df.loc[df['test_name'] == test_names[plt_idx]]
        ascending = False if plt_idx == 3 else True
        group_df = test_df.groupby([
            'model_name', 'xy_ratio'
        ])[['value', 'inst_cond']].mean().reset_index()\
            .sort_values(by=['value'], ascending=ascending)
        for model_name in model_names:
            for xy_ratio in xy_ratios:
                model_df = test_df.loc[test_df['model_name'] == model_name]
                name = str(model_df['inst_cond'].values[0])
                model_df = model_df.loc[model_df['xy_ratio'] == xy_ratio]
                fig.add_trace(go.Box(
                    x=model_df['xy_ratio'],
                    y=model_df['value'],
                    notched=False, boxpoints=False,
                    boxmean=True,
                    marker_color=colour_assign(group_assign(name)),
                    name=name_assign(name),
                    legendgroup=group_assign(name),
                    showlegend=(group_assign(name) not in seen)
                ), row=rows[plt_idx], col=cols[plt_idx])
                seen.append(group_assign(name))
        temp_df = group_df.loc[group_df['inst_cond'] == True].sort_values(
            by=['xy_ratio'], ascending=ascending)
        fig.add_trace(go.Scatter(
            x=temp_df['xy_ratio'],
            y=temp_df['value'],
            mode='lines+markers',
            marker_color=colour_assign(group_assign('True')),
            name='Mean over 100 runs',
            showlegend=False
        ), row=rows[plt_idx], col=cols[plt_idx])
        seen.append("green")
        temp_df = group_df.loc[group_df['inst_cond'] == False].sort_values(
            by=['xy_ratio'], ascending=ascending)
        fig.add_trace(go.Scatter(
            x=temp_df['xy_ratio'],
            y=temp_df['value'],
            mode='lines+markers',
            marker_color=colour_assign(group_assign('False')),
            name='Mean over 100 runs',
            showlegend=False
        ), row=rows[plt_idx], col=cols[plt_idx])

    fig.update_xaxes(
        title_text='Amount of conditional shift', col=2)
    fig.update_xaxes(
        tickvals=[0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1])
    fig.update_layout(
        width=3*BIG_FIG_SIZE,
        height=BIG_FIG_SIZE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.1,
        xanchor="left",
        x=0.01,
        bordercolor="Black",
        borderwidth=0.5
    ))
    fig['layout'].update(margin=dict(l=0, r=0, b=0, t=30))
    fig.update_xaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey',
                     linecolor='black', mirror=True)
    fig.write_image(os.path.join(result_dir, "xy_ratio.pdf"))
    time.sleep(2)
    fig.write_image(os.path.join(result_dir, "xy_ratio.pdf"))
