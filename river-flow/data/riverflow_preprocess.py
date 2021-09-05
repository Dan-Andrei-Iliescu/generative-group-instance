import pandas as pd
import numpy as np


def load():
    df = np.load('river-flow/data/maurer.pickle', allow_pickle=True)
    df_attributes = np.load(
        'river-flow/data/attributes.pickle', allow_pickle=True)
    basin_list = pd.read_csv('river-flow/data/basin_list.txt', header=None)

    basin_list = basin_list[0].apply(
        lambda x: '0' + str(x) if len(str(x)) < 8 else str(x))

    ymd = df['01013500'].groupby(['Year', 'Mnth']).size().reset_index()
    ymd['d_cumsum'] = 0
    for y in ymd['Year']:
        temp = ymd.loc[ymd['Year'] == y, 0].cumsum()
        ymd.loc[ymd['Year'] == y, 'd_cumsum'] = np.concatenate(
            ([0], temp[:-1]))

    for k, v in df.items():
        df[k].reset_index(inplace=True)
        df[k] = pd.DataFrame.merge(
            v, ymd, right_on=['Year', 'Mnth'], left_on=['Year', 'Mnth'])
        df[k]['n_day'] = df[k]['d_cumsum'] + df[k]['Day']
        df[k]['t'] = df[k]['Year'] + (df[k]['n_day'] - 1) / 366

    df_filtered = {}
    for (b) in basin_list:
        if (df[b].shape[0] < 10593):
            basin_list = basin_list[basin_list != b]
            continue
        df_filtered[b] = df[b][[
            'Date', 't', 'Dayl(s)', 'PRCP(mm/day)', 'SRAD(W/m2)', 'Tmax(C)', 'Tmin(C)', 'Vp(Pa)', 'Q']]

    # Choose type of transform, i.e., 'standardize' or 'normalize'
    dist = 'gaussian'

    if dist == 'gaussian':
        transform = 'standardize'
        log_P = True
        log_Q = True

    if dist == 'gamma':
        transform = 'normalize'
        log_P = True
        log_Q = True
        gamma_shift = 1e-3

    divide_by_area = True
    cols = ['t', 'Dayl(s)', 'PRCP(mm/day)', 'SRAD(W/m2)',
            'Tmax(C)', 'Tmin(C)', 'Vp(Pa)', 'Q']
    epsilon = 1e-3

    x_maxs, x_mins, x_means, x_stds = [], [], [], []
    for k, v in df_filtered.items():
        #     # Scale streamflow values by catchment area
        if divide_by_area:
            v['Q'] = v['Q']/df_attributes[k]['area_geospa_fabric'].values

        # Calculate mean (after scaling by area)
    #     v['Q_mu'] = v['Q'].mean()

    #     Log-transform precipitation
        if log_P:
            v['PRCP(mm/day)'] = np.log(v['PRCP(mm/day)'] + epsilon)

        # Log-transform streamflow
        if log_Q:
            v['Q'] = np.log(v['Q'] + epsilon)

    #     x_maxs.append(v[cols].max().values)
    #     x_mins.append(v[cols].min().values)
        x_means.append(v[cols].values)
        x_stds.append(v[cols].values)

    # x_max = np.concatenate(x_maxs).reshape(-1,len(cols)).max(axis=0)
    # x_min = np.concatenate(x_mins).reshape(-1,len(cols)).min(axis=0)
    x_mean = np.concatenate(x_means, axis=0).mean(axis=0)
    x_std = np.concatenate(x_stds, axis=0).std(axis=0)

    for k, v in df_filtered.items():
        for i, col in enumerate(cols):
            if transform == 'normalize':

                v[col] = (v[col] - x_min[i]) / (x_max[i] - x_min[i])

                if dist == "gamma":
                    v['Q'] = v['Q'] + gamma_shift

                def rev_transform(x):
                    x = x * (x_max[0] - x_min[0]) + x_min[0]
                    if log_Q:
                        x = np.exp(x) - epsilon
                    if dist == "gamma":
                        x = x - gamma_shift
                    return x

                def rev_transform_tensor(x):
                    x = x * (x_max[0] - x_min[0]) + x_min[0]
                    if log_Q:
                        x = torch.exp(x) - epsilon
                    if dist == "gamma":
                        x = x - gamma_shift
                    return x

            elif transform == 'standardize':

                v[col] = (v[col] - x_mean[i]) / x_std[i]

                # WARNING -- NO GAMMA SHIFT

                def rev_transform(x):
                    x = x * x_std[0] + x_mean[0]
                    if log_Q:
                        x = np.exp(x) - epsilon
                    if dist == "gamma":
                        x = x - gamma_shift
                    return x

                def rev_transform_tensor(x):
                    x = x * x_std[0] + x_mean[0]
                    if log_Q:
                        x = torch.exp(x) - epsilon
                    if dist == "gamma":
                        x = x - gamma_shift
                    return x

            else:
                print("No transform has been applied.")

    return df_filtered


if __name__ == "__main__":
    df = load()
    print(df)
