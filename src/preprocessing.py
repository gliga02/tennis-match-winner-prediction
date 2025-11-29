import pandas as pd

def transform_matches(df):
    winner_df = pd.DataFrame({
        'player_id': df['winner_id'],
        'player_name': df['winner_name'],
        'player_hand': df['winner_hand'],
        'player_age': df['winner_age'],

        'opp_id': df['loser_id'],
        'opp_name': df['loser_name'],
        'opp_hand': df['loser_hand'],
        'opp_age': df['loser_age'],

        'player_ace': df['w_ace'],
        'player_df': df['w_df'],
        'player_svpt': df['w_svpt'],
        'player_1stIn': df['w_1stIn'],
        'player_1stWon': df['w_1stWon'],
        'player_2ndWon': df['w_2ndWon'],
        'player_SvGms': df['w_SvGms'],
        'player_bpSaved': df['w_bpSaved'],
        'player_bpFaced': df['w_bpFaced'],

        'opp_ace': df['l_ace'],
        'opp_df': df['l_df'],
        'opp_svpt': df['l_svpt'],
        'opp_1stIn': df['l_1stIn'],
        'opp_1stWon': df['l_1stWon'],
        'opp_2ndWon': df['l_2ndWon'],
        'opp_SvGms': df['l_SvGms'],
        'opp_bpSaved': df['l_bpSaved'],
        'opp_bpFaced': df['l_bpFaced'],

        'surface': df['surface'],
        'tourney_level': df['tourney_level'],
        'best_of': df['best_of'],
        'minutes': df['minutes'],

        'player_rank': df['winner_rank'],
        'opp_rank': df['loser_rank'],

        'label': 1
    })

    loser_df = pd.DataFrame({
        'player_id': df['loser_id'],
        'player_name': df['loser_name'],
        'player_hand': df['loser_hand'],
        'player_age': df['loser_age'],
        
        'opp_id': df['winner_id'],
        'opp_name': df['winner_name'],
        'opp_hand': df['winner_hand'],
        'opp_age': df['winner_age'],
        
        'player_ace': df['l_ace'],
        'player_df': df['l_df'],
        'player_svpt': df['l_svpt'],
        'player_1stIn': df['l_1stIn'],
        'player_1stWon': df['l_1stWon'],
        'player_2ndWon': df['l_2ndWon'],
        'player_SvGms': df['l_SvGms'],
        'player_bpSaved': df['l_bpSaved'],
        'player_bpFaced': df['l_bpFaced'],
        
        'opp_ace': df['w_ace'],
        'opp_df': df['w_df'],
        'opp_svpt': df['w_svpt'],
        'opp_1stIn': df['w_1stIn'],
        'opp_1stWon': df['w_1stWon'],
        'opp_2ndWon': df['w_2ndWon'],
        'opp_SvGms': df['w_SvGms'],
        'opp_bpSaved': df['w_bpSaved'],
        'opp_bpFaced': df['w_bpFaced'],
        
        'surface': df['surface'],
        'tourney_level': df['tourney_level'],
        'best_of': df['best_of'],
        'minutes': df['minutes'],
        
        'player_rank': df['loser_rank'],
        'opp_rank': df['winner_rank'],

        'label': 0
    })

    return pd.concat([winner_df, loser_df], ignore_index=True)