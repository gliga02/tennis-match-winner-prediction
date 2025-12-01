import sys
import joblib
import pandas as pd
import gradio as gr

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.prediction import predict_match, plot_serve_comparison

models_dir = BASE_DIR / "models"
data_dir = BASE_DIR / "data"

rf_model = joblib.load(models_dir / "rf_full.joblib")
feature_columns = joblib.load(models_dir / "feature_columns_full.joblib")

player_surface_stats = pd.read_csv(data_dir / "player_surface_stats.csv")

player_list = sorted(player_surface_stats['player_name'].unique())
surfaces = ["Hard", "Clay", "Grass"]

def gradio_predict(player1, player2, surface):
    if player1 == player2:
        return "Choose two different players.", None
    
    result = predict_match(
        player1 = player1,
        player2 = player2,
        surface=surface,
        model=rf_model,
        player_surface_stats=player_surface_stats,
        feature_columns=feature_columns
    )

    p1 = result['player1']
    p2 = result['player2']
    p1_prob = result['p1_win_prob']
    p2_prob = result['p2_win_prob']
    p1_stats = result['p1_stats']
    p2_stats = result['p2_stats']

    summary = (
        f"### Prediction\n"
        f"- **Surface:** {surface}\n"
        f"- **{p1} win probability: {p1_prob:.3f}**\n"
        f"- **{p2} win probability: {p2_prob:.3f}**\n\n"
        f"### Key serve stats:\n"
        f"- {p1} 1st serve points won: **{p1_stats['player_1stWon_pct']:.3f}**\n"
        f"- {p2} 1st serve points won: **{p2_stats['player_1stWon_pct']:.3f}**\n"
        f"- {p1} 2nd serve points won: **{p1_stats['player_2ndWon_pct']:.3f}**\n"
        f"- {p2} 2nd serve points won: **{p2_stats['player_2ndWon_pct']:.3f}**\n"
        f"- {p1} BP saved: **{p1_stats['player_bp_saved_pct']:.3f}**\n"
        f"- {p2} BP saved: **{p2_stats['player_bp_saved_pct']:.3f}**\n"
    )

    fig = plot_serve_comparison(
        p1_stats=p1_stats,
        p2_stats=p2_stats,
        player1=p1,
        player2=p2
    )

    return summary, fig


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(player_list, label="Player 1", value="Novak Djokovic"),
        gr.Dropdown(player_list, label="Player 2", value="Pete Sampras"),
        gr.Radio(surfaces, label="Surface", value="Grass")
    ],

    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Plot(label="Serve comparison")
    ],

    title="Tennis Match Winner Prediction",

    description=(
        "Select two ATP Players and Surface.\n"
        "The model uses historical match statistics to estimate "
        "win probabilities and compare their serve performance."
    ),

    allow_flagging="never",

    theme="gradio/soft",

    css="""
        .gradio-container {
            background: #f7f7f9 !important;
            font-family: Inter, sans-serif;
        }

        h1, h2, h3 {
            font-weight: 600 !important;
        }

        .gradio-input {
            border-radius: 10px !important;
        }
    """
)

if __name__ == "__main__":
    demo.launch()