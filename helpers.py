import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

team_map = {'Arizona Cardinals':'ARI', 'Atlanta Falcons':'ATL', 'Baltimore Ravens':'BAL', 'Buffalo Bills':'BUF', 'Carolina Panthers':'CAR', 'Chicago Bears':'CHI', 'Cincinnati Bengals':'CIN',
'Cleveland Browns':'CLE', 'Dallas Cowboys':'DAL', 'Denver Broncos':'DEN', 'Detroit Lions':'DET', 'Green Bay Packers':'GB', 'Houston Texans':'HOU', 'Indianapolis Colts':'IND', 'Jacksonville Jaguars':'JAX', 
'Kansas City Chiefs':'KC', 'Las Vegas Raiders':'LV', 'Los Angeles Chargers':'LAC', 'Los Angeles Rams':'LA', 'Miami Dolphins':'MIA', 'Minnesota Vikings':'MIN', 'New England Patriots':'NE', 'New Orleans Saints':'NO',
'New York Giants':'NYG', 'New York Jets':'NYJ', 'Philadephia Eagles':'PHI', 'Pittsburgh Steelers':'PIT', 'San Francisco 49ers':'SF', 'Seattle Seahawks':'SEA', 'Tampa Bay Buccaneers':'TB', 'Tennessee Titans':'TEN',
'Washington Commanders':'WAS', 'Washington Football Team':'WAS', 'Oakland Raiders':'LV', 'Washington Redskins':'WAS', 'San Diego Chargers':'LAC', 'St. Louis Rams':'LA'}


PBP_DTYPES = {
    'play_id':'uint16',
    'old_game_id':'uint32',
    'week':'uint8',
    'yardline_100':'Int8',
    'quarter_seconds_remaining':'Int16',
    'half_seconds_remaining':'Int16',
    'game_seconds_remaining':'Int16',
    'quarter_end':'uint8', # bool
    'drive':'Int8',
    'sp':'uint8', # bool
    'qtr':'uint8',
    'down':'Int8',
    'goal_to_go':'uint8', # bool
    'ydstogo':'uint8',
    'ydsnet':'Int8',
    'yards_gained':'Int8',
    'shotgun':'uint8', # bool
    'no_huddle':'uint8', # bool
    'qb_dropback':'Int8', # bool
    'qb_kneel':'Int8', # bool
    'qb_spike':'Int8', # bool
    'qb_scramble':'Int8', # bool
    'air_yards':'Int8',
    'yards_after_catch':'Int8',
    'kick_distance':'Int8',
    'home_timeouts_remaining':'uint8',
    'away_timeouts_remaining':'uint8',
    'timeout':'Int8', # bool
    'posteam_timeouts_remaining':'Int8',
    'defteam_timeouts_remaining':'Int8',
    'total_home_score':'uint8',  # will have to hope no one scores >255 points
    'total_away_score':'uint8',
    'posteam_score':'Int8', # >127 points, even
    'defteam_score':'Int8',
    'score_differential':'Int8',
    'posteam_score_post':'Int8',
    'defteam_score_post':'Int8',
    'score_differential_post':'Int8',
#     no_score_prob                           float64
# opp_fg_prob                             float64
# opp_safety_prob                         float64
# opp_td_prob                             float64
# fg_prob                                 float64
# safety_prob                             float64
# td_prob                                 float64
# extra_point_prob                        float64
# two_point_conversion_prob               float64
# ep                                      float64
# epa                                     float64
# total_home_epa                          float64
# total_away_epa                          float64
# total_home_rush_epa                     float64
# total_away_rush_epa                     float64
# total_home_pass_epa                     float64
# total_away_pass_epa                     float64
# air_epa                                 float64
# yac_epa                                 float64
# comp_air_epa                            float64
# comp_yac_epa                            float64
# total_home_comp_air_epa                 float64
# total_away_comp_air_epa                 float64
# total_home_comp_yac_epa                 float64
# total_away_comp_yac_epa                 float64
# total_home_raw_air_epa                  float64
# total_away_raw_air_epa                  float64
# total_home_raw_yac_epa                  float64
# total_away_raw_yac_epa                  float64
# wp                                      float64
# def_wp                                  float64
# home_wp                                 float64
# away_wp                                 float64
# wpa                                     float64
# vegas_wpa                               float64
# vegas_home_wpa                          float64
# home_wp_post                            float64
# away_wp_post                            float64
# vegas_wp                                float64
# vegas_home_wp                           float64
# total_home_rush_wpa                     float64
# total_away_rush_wpa                     float64
# total_home_pass_wpa                     float64
# total_away_pass_wpa                     float64
# air_wpa                                 float64
# yac_wpa                                 float64
# comp_air_wpa                            float64
# comp_yac_wpa                            float64
# total_home_comp_air_wpa                 float64
# total_away_comp_air_wpa                 float64
# total_home_comp_yac_wpa                 float64
# total_away_comp_yac_wpa                 float64
# total_home_raw_air_wpa                  float64
# total_away_raw_air_wpa                  float64
# total_home_raw_yac_wpa                  float64
# total_away_raw_yac_wpa                  float64
# punt_blocked                            float64
# first_down_rush                         float64
# first_down_pass                         float64
# first_down_penalty                      float64
# third_down_converted                    float64
# third_down_failed                       float64
# fourth_down_converted                   float64
# fourth_down_failed                      float64
# incomplete_pass                         float64
# touchback                                 int64
# interception                            float64
# punt_inside_twenty                      float64
# punt_in_endzone                         float64
# punt_out_of_bounds                      float64
# punt_downed                             float64
# punt_fair_catch                         float64
# kickoff_inside_twenty                   float64
# kickoff_in_endzone                      float64
# kickoff_out_of_bounds                   float64
# kickoff_downed                          float64
# kickoff_fair_catch                      float64
# fumble_forced                           float64
# fumble_not_forced                       float64
# fumble_out_of_bounds                    float64
# solo_tackle                             float64
# safety                                  float64
# penalty                                 float64
# tackled_for_loss                        float64
# fumble_lost                             float64
# own_kickoff_recovery                    float64
# own_kickoff_recovery_td                 float64
# qb_hit                                  float64
# rush_attempt                            float64
# pass_attempt                            float64
# sack                                    float64
# touchdown                               float64
# pass_touchdown                          float64
# rush_touchdown                          float64
# return_touchdown                        float64
# extra_point_attempt                     float64
# two_point_attempt                       float64
# field_goal_attempt                      float64
# kickoff_attempt                         float64
# punt_attempt                            float64
# fumble                                  float64
# complete_pass                           float64
# assist_tackle                           float64
# lateral_reception                       float64
# lateral_rush                            float64
# lateral_return                          float64
# lateral_recovery                        float64
}

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
