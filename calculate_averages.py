## Golf Data Average Player Stats Calculations. Append to cleaned_golf.csv

import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_golf.csv')

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
        player_id = row[f"p{i}_dg_id"]

        entry = {"dg_id": player_id}
        for stat in player_stats:
            entry[stat] = row[f"{stat}_p{i}"]

        rows.append(entry)

# Convert into long-form dataframe: one row per player per match
long_df = pd.DataFrame(rows)

# Compute GLOBAL AVERAGES for each player across ALL rounds
player_avg = long_df.groupby("dg_id")[player_stats].mean().reset_index()

# Merge these global averages back into your original dataframe
for i in [1,2,3]:
    df = df.merge(player_avg, how="left", left_on=f"p{i}_dg_id", right_on="dg_id", suffixes=("", f"_p{i}_avg"))
    df = df.drop(columns=["dg_id"])  # remove the extra merge column

# Save updated file
df.to_csv("cleaned_golf_global_averages.csv", index=False)
