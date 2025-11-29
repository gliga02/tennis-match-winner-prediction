import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_player_surface_stats(player_surface_stats, player_name, surface):
    df_player = player_surface_stats[player_surface_stats['player_name'] == player_name]

    df_surface = df_player[df_player['surface'] == surface]

    if df_surface.shape[0] > 0 and df_surface['n_matches'].iloc[0] >= 5:
        return df_surface.iloc[0]
    
    df_all = df_player.groupby('player_name').agg({
        'player_1stIn_pct': 'mean',
        'player_1stWon_pct': 'mean',
        'player_2ndWon_pct': 'mean',
        'player_bp_saved_pct': 'mean',
        'avg_rank': 'mean',
        'avg_age': 'mean'
    }).reset_index()

    df_all['surface'] = surface

    return df_all.iloc[0]

def build_feature_vector(p1, p2, surface, feature_columns):
    data = {}

    data['player_1stIn_pct'] = p1['player_1stIn_pct']
    data['opp_1stIn_pct'] = p2['player_1stIn_pct']

    data['player_1stWon_pct'] = p1['player_1stWon_pct']
    data['opp_1stWon_pct']    = p2['player_1stWon_pct']

    data['player_2ndWon_pct'] = p1['player_2ndWon_pct']
    data['opp_2ndWon_pct']    = p2['player_2ndWon_pct']

    data['player_bp_saved_pct'] = p1['player_bp_saved_pct']
    data['opp_bp_saved_pct']    = p2['player_bp_saved_pct']

    data['rank_diff'] = p1['avg_rank'] - p2['avg_rank']
    data['age_diff'] = p1['avg_age'] - p2['avg_age']

    data['hand_diff'] = 0

    data['best_of'] = 3
    data['minutes'] = 120

    data['surface_Grass'] = 1 if surface == 'Grass' else 0
    data['surface_Hard'] = 1 if surface == 'Hard' else 0

    data['tourney_level_D'] = 0
    data['tourney_level_F'] = 0
    data['tourney_level_G'] = 0
    data['tourney_level_M'] = 0
    data['tourney_level_O'] = 0

    row = pd.DataFrame([data])[feature_columns]

    return row


def predict_match(player1, player2, surface, model, player_surface_stats, feature_columns):
    p1_stats = get_player_surface_stats(player_surface_stats, player1, surface)
    p2_stats = get_player_surface_stats(player_surface_stats, player2, surface)

    X_row = build_feature_vector(p1_stats, p2_stats, surface, feature_columns)

    proba = model.predict_proba(X_row)[0][1]

    return {
        "player1": player1,
        "player2": player2,
        "surface": surface,
        "p1_win_prob": float(proba),
        "p2_win_prob": float(1 - proba),
        "p1_stats": p1_stats.to_dict(),
        "p2_stats": p2_stats.to_dict()
    }

def plot_serve_comparison(p1_stats, p2_stats, player1, player2):
    metrics = ['player_1stWon_pct', 'player_2ndWon_pct', 'player_bp_saved_pct']
    labels  = ['1st Serve Won %', '2nd Serve Won %', 'Break Points Saved %']

    p1 = np.array([p1_stats[m] for m in metrics]) * 100
    p2 = np.array([p2_stats[m] for m in metrics]) * 100


    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the loop

    p1_closed = np.concatenate((p1, [p1[0]]))
    p2_closed = np.concatenate((p2, [p2[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, p1_closed, linewidth=2, linestyle='-', label=player1)
    ax.fill(angles, p1_closed, alpha=0.25)

    ax.plot(angles, p2_closed, linewidth=2, linestyle='--', label=player2)
    ax.fill(angles, p2_closed, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.set_yticklabels(['40%', '50%', '60%', '70%', '80%', '90%'])
    ax.set_ylim(40, 100)

    ax.set_title("Serve Performance Profile", pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

    ax.grid(True, linestyle=':', linewidth=0.7)
    fig.tight_layout()

    return fig