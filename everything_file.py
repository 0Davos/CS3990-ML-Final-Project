'''
File with all the code we've amassed
'''


# %% - Base Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



# %% - Initial Cleaning - gets us cleaned_golf.csv


csv_file_path = 'clean_merged_playerdata_with_weather.csv'

df = pd.read_csv(csv_file_path)

# drop columns we definitely don't need
df = df.drop(["bet_type", "tie_rule", "open_time", "close_time", 
              "p1_outcome_text", "p2_outcome_text", "p3_outcome_text", 
              "book", "event_completed", "event_name", "odds", 
             
             'p1_player_name', 'p2_player_name', 'p3_player_name',
             
             'dg_id_p1', 'fin_text_p1', 'fin_text_p2', 'fin_text_p3',
             'course_name_p1', 'teetime_p2', 'teetime_p3', 'wx_teetime',
             'wx_datetime_hour',
             'wx_date_from_close', 'wx_conditions', 'wx_icon', 'wx_datetimeEpoch',
             'tour_p1', 'season'], axis=1)

# rename columns we'd like to keep
df = df.rename(columns={'teetime_p1':'teetime'})

df = df.drop(["teetime"], axis=1)


# preciptype can either only be nan or 'rain'
df['wx_preciptype'] = df['wx_preciptype'].fillna(0)
df['wx_preciptype'] = df['wx_preciptype'].apply(lambda x: 1 if x != 0 else x)
    
    

# Create one outcome column
df['outcome'] = (
    df[['p1_outcome', 'p2_outcome', 'p3_outcome']]
    .fillna(0) # turn all na's or NaNs to 0
    .idxmax(axis=1) 
    .str.extract(r'p(\d+)_outcome') # pull out 1, 2, 3
    .astype(float) # convert to floats
)
# Then remove the other outcome column
df = df.drop(['p1_outcome', 'p2_outcome', 'p3_outcome'], axis=1)

# create a csv to check changes
df.to_csv('cleaned_golf.csv', index=False)


# %% - Calculate Rolling averages, create cleaned_golf_rolling_averages.csv

df = pd.read_csv('cleaned_golf.csv')
# reorganize based on time (year, event_id, round_num)
df = df.sort_values(['year', 'event_id', 'round_num']).reset_index(drop=True)

# Player names are under a dg_id column, so we will group by that
# Each row has three different dg_ids, and their respective player stats: p1_dg_id, p2_dg_id, p3_dg_id
# Player stats in each row, which represents a match, are: 
# round_score_p1, sg_putt_p1,sg_arg_p1,sg_app_p1,sg_ott_p1,sg_t2g_p1,sg_total_p1,driving_dist_p1,driving_acc_p1,gir_p1,scrambling_p1,prox_rgh_p1, prox_fw_p1,great_shots_p1,poor_shots_p1,eagles_or_better_p1,birdies_p1,pars_p1,bogies_p1,doubles_or_worse_p1
# each player has the same stats but with _p2 and _p3 suffixes respectively
# Calculate average player stats and append to the dataframe


# List of stat names (without p1/p2/p3 suffixes)
player_stats = [
    'round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total',
    'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw',
    'great_shots', 'poor_shots', 'eagles_or_better', 'birdies', 'pars',
    'bogies', 'doubles_or_worse'
]

# We will build a long-format table of all players across the dataset
rows = []

for idx, row in df.iterrows():
    # For each match, create 3 entries (one per player)
    for i in [1, 2, 3]:
        player_data = {
            'row_idx': idx,
            'player_num': i,
            'dg_id': row[f'p{i}_dg_id'],
            'year': row['year'],
            'event_id': row['event_id'],
            'round_num': row['round_num']
        }
        for stat in player_stats:
            player_data[stat] = row[f'{stat}_p{i}']
        rows.append(player_data)

# Convert into long-form dataframe: one row per player per match
long_df = pd.DataFrame(rows)
long_df = long_df.sort_values(['dg_id', 'year', 'event_id', 'round_num']).reset_index(drop=True)

# Compute rolling avgs for each player across only rounds before current date
for stat in player_stats:
    long_df[f'{stat}_avg'] = long_df.groupby('dg_id')[stat].transform(
        lambda x: x.shift(1).expanding().mean()
    )


# Merge back to original dataframe
for player_num in [1, 2, 3]:
    player_df = long_df[long_df['player_num'] == player_num].copy()
    
    for stat in player_stats:
        df[f'{stat}_p{player_num}_avg'] = df.index.map(
            player_df.set_index('row_idx')[f'{stat}_avg']
        )

# Drop original stat columns
columns_to_drop = []
for player_num in [1, 2, 3]:
    for stat in player_stats:
        columns_to_drop.append(f'{stat}_p{player_num}')

df = df.drop(columns=columns_to_drop)

df.to_csv("cleaned_golf_rolling_averages.csv", index=False)




# %% - PCA - redo with cleaned_golf_rolling averages
# %% - Create Correlation Matrix?
# %% - Random Forest
# %% - Log Regression
# %% - Neural Network
