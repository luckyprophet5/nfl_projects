import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

team_map = {'Arizona Cardinals':'ARI', 'Atlanta Falcons':'ATL', 'Baltimore Ravens':'BAL', 'Buffalo Bills':'BUF', 'Carolina Panthers':'CAR', 'Chicago Bears':'CHI', 'Cincinnati Bengals':'CIN',
'Cleveland Browns':'CLE', 'Dallas Cowboys':'DAL', 'Denver Broncos':'DEN', 'Detroit Lions':'DET', 'Green Bay Packers':'GB', 'Houston Texans':'HOU', 'Indianapolis Colts':'IND', 'Jacksonville Jaguars':'JAX', 
'Kansas City Chiefs':'KC', 'Las Vegas Raiders':'LV', 'Los Angeles Chargers':'LAC', 'Los Angeles Rams':'LA', 'Miami Dolphins':'MIA', 'Minnesota Vikings':'MIN', 'New England Patriots':'NE', 'New Orleans Saints':'NO',
'New York Giants':'NYG', 'New York Jets':'NYJ', 'Philadephia Eagles':'PHI', 'Pittsburgh Steelers':'PIT', 'San Francisco 49ers':'SF', 'Seattle Seahawks':'SEA', 'Tampa Bay Buccaneers':'TB', 'Tennessee Titans':'TEN',
'Washington Commanders':'WAS', 'Washington Football Team':'WAS', 'Oakland Raiders':'LV', 'Washington Redskins':'WAS', 'San Diego Chargers':'LAC', 'St. Louis Rams':'LA'}


def fit(df, x, y):
    z = np.polyfit(df[x].fillna(0).astype('float'), df[y].fillna(0).astype('float'), 1)
    p = np.poly1d(z)
    return p

def residual(df, x, y):
    p = fit(df, x, y)
    return (df[y] - p(df[x]))

def plot_fit(df, x, y, linestyle='r--'):
    p = fit(df, x, y)
    plt.plot(df.sort_values(x)[x],p(df.sort_values(x)[x]),linestyle)

def label_plot(labeled_subset,x, y, label_col, x_offset=.01, y_offset=.01):
    '''labeled_subset - rows to label; x - x axis col; y - y axis col; label_col - column to use for labels'''
    for _, point in labeled_subset.iterrows():
        plt.text(point[x]+x_offset,point[y]+y_offset,point[label_col])

def shift_df(columns, groupby, data):
    cols = [groupby, 'season'] + columns 
    data = data[cols]
    ldata = data.groupby(by=groupby).shift(-1)
    data.columns = [groupby, 'prev_season'] + [f'prev_{col}' for col in columns]
    new_data = pd.concat((data, ldata), axis=1).dropna(subset=['season'])
    return new_data


# I stole this code from @EthanCDouglas and modified it a bit
def year_to_year_corr(columns, groupby, data):
    new_data = shift_df(columns, groupby, data).drop(columns=['prev_season','season'])
    tot_corr = new_data.corr(method='pearson')
    num_corr_cols = len(columns)
    corr = tot_corr.iloc[num_corr_cols:,num_corr_cols:]
    pred = tot_corr.iloc[0:num_corr_cols, num_corr_cols:]
    return new_data,corr,pred

def weighted_avg(data, col, weight, groupby):
    data[f'weighted_{col}'] = data[weight]*data[col]
    aggregated = data\
        .groupby(groupby)\
        .agg({
            f'weighted_{col}':'sum',
            weight:'sum'
        })\
        .assign(**{col:(lambda x:x[f'weighted_{col}']/x[weight])})
    return aggregated[col]

def get_ordinal_ending(n):
    if n%100 in [11,12,13]:
        return 'th'
    ordinal_mapping = {0:'th',1:'st',2:'nd',3:'rd',**{num:'th' for num in range(4,10)}}
    return ordinal_mapping[n%10]

def to_ordinal(n):
    return str(int(n)) + get_ordinal_ending(n)

def pctile_format(n):
    return to_ordinal(int(round(n,2)*100))

def get_field_over_time_dict(df, field, characteristic):
    if characteristic == 'mean':
        return {
            f'average_{field}':{
                df.season.min():df.loc[df.season.isin(range(df.season.min(),df.season.min()+3))][field].mean(),
                **{season:df.loc[df.season.isin(range(season-1,season+2))][field].mean() for season in range(df.season.min()+1,df.season.max())},
                df.season.max():df.loc[df.season.isin(range(df.season.max()-2,df.season.max()))][field].mean()
            }
        }
    elif characteristic == 'std':
        return {
            f'{field}_std':{
                df.season.min():df.loc[df.season.isin(range(df.season.min(),df.season.min()+3))][field].std(),
                **{season:df.loc[df.season.isin(range(season-1,season+2))][field].std() for season in range(df.season.min()+1,df.season.max())},
                df.season.max():df.loc[df.season.isin(range(df.season.max()-2,df.season.max()))][field].std()
            }
        }
    return {}

def season_era_adjust(df, field):
    era_avgs = get_field_over_time_dict(df, field, 'mean')
    era_stds = get_field_over_time_dict(df, field, 'std')

    ret = df.merge(pd.DataFrame(era_avgs).reset_index().rename(columns={'index':'season'}), on='season', how='left')
    ret = ret.merge(pd.DataFrame(era_stds).reset_index().rename(columns={'index':'season'}), on='season', how='left')
    ret[f'{field}+'] = ret[field]-ret[f'average_{field}']
    ret[f'{field}_stdevs_above_mean'] = ret[f'{field}+']/ret[f'{field}_std']
    return ret.drop(columns=[f'average_{field}',f'{field}_std'])

def games_era_adjust(games, seasons, field):
    era_avgs = get_field_over_time_dict(seasons, field, 'mean')
    era_stds = get_field_over_time_dict(seasons, field, 'std')

    ret = games.merge(pd.DataFrame(era_avgs).reset_index().rename(columns={'index':'season'}), on='season', how='left')
    ret = ret.merge(pd.DataFrame(era_stds).reset_index().rename(columns={'index':'season'}), on='season', how='left')
    ret[f'{field}+'] = ret[field]-ret[f'average_{field}']
    ret[f'{field}_stdevs_above_mean'] = ret[f'{field}+']/ret[f'{field}_std']
    return ret.drop(columns=[f'average_{field}',f'{field}_std'])

def columns_gradient(df, gradient='PRGn', filter_=True, start=0, end=0, min_max_map={}, aux_cols={}, formats={}, show_cols=[], gradient_cols=[]):
    subset = df.copy()
    if isinstance(filter_,pd.Series):
        subset = df.loc[filter_]
    if start!=0 or end!=0:
        subset = subset[start:end]
    gmap = subset.copy()
    for primary,aux in aux_cols.items():
        if primary in subset.columns and aux in subset.columns and pd.api.types.is_numeric_dtype(subset[primary]):
            if primary in formats and aux in formats:
                subset[primary] = subset\
                    .apply(lambda x:f"{formats[primary]} ({formats[aux]})".format(x[primary], x[aux]), axis=1)
            elif primary in formats:
                subset[primary] = subset\
                    .apply(lambda x:f"{formats[primary]} ({x[aux]})".format(x[primary]), axis=1)
            elif aux in formats:
                subset[primary] = subset\
                    .apply(lambda x:f"{x[primary]} ({formats[aux]})".format(x[aux]), axis=1)
            else:
                subset[primary] = subset\
                    .apply(lambda x:f"{x[primary]} ({x[aux]})", axis=1)
    for col in subset.columns:
        if col in formats and col not in aux_cols:
            subset[col] = subset[col].apply(lambda x:formats[col].format(x))
    style_ = subset.style
    if len(show_cols) > 0:
        style_ = subset[show_cols].style
    if len(gradient_cols) == 0:
        for col in show_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                gradient_cols+=[col]
    for col in gradient_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if col in min_max_map:
            if 'min' in min_max_map[col]:
                min_val = min_max_map[col]['min']
            if 'max' in min_max_map[col]:
                max_val = min_max_map[col]['max']
        style_ = style_.background_gradient(subset=[col], cmap=cm.get_cmap(gradient), vmin=min_val, vmax=max_val, gmap=gmap[col])
    return style_

def plot_yequalsx(ax, linestyle='r--'):
    # https://stackoverflow.com/a/25497638
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linestyle=linestyle)
