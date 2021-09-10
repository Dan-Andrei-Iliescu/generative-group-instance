import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_load import load


def plot_data(data):
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}],
               [{}, {}]],
        subplot_titles=('Q', 'Dayl(s)', 'PRCP(mm/day)',
                        'SRAD(W/m2)', 'Tmax(C)', 'Tmin(C)', 'Vp(Pa)'),
        horizontal_spacing=0.05,
        vertical_spacing=0.05)

    colour_cycle = ['rgb(31, 119, 180)',
                    'rgb(255, 127, 14)',
                    'rgb(44, 160, 44)',
                    'rgb(214, 39, 40)',
                    'rgb(148, 103, 189)',
                    'rgb(140, 86, 75)',
                    'rgb(227, 119, 194)',
                    'rgb(127, 127, 127)',
                    'rgb(188, 189, 34)',
                    'rgb(23, 190, 207)']

    basins = list(data.keys())
    num_basins = 5
    for idx in range(num_basins):
        basin = basins[idx]
        df = data[basin]
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['Q'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['Dayl(s)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['PRCP(mm/day)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['SRAD(W/m2)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['Tmax(C)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['Tmin(C)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df['t'], y=df['Vp(Pa)'], fill='tozeroy',
            legendgroup=basin, name=basin, line=dict(color=colour_cycle[idx]),
            showlegend=False), row=4, col=2)

    fig.update_layout(showlegend=True, title_text="River Flow Dataset")
    fig.show()


if __name__ == "__main__":
    data = load()
    plot_data(data)
