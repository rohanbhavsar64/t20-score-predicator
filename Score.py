import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace with your actual dataset)
# df = pd.read_csv("match.csv")

def create_per_over_score_prediction_chart():
    st.title("IPL Per-Over Score Prediction Chart")

    # Select match ID (you can customize this based on your dataset)
    match_id = st.selectbox("Select Match ID", df["match_id"].unique())

    # Filter data for the selected match
    match_data = df[df["match_id"] == match_id]

    # Calculate cumulative runs per over
    match_data["cumulative_runs"] = match_data.groupby("over")["total_runs_x"].cumsum()

    # Create the per-over score prediction chart
    plt.figure(figsize=(10, 6))
    plt.plot(match_data["over"], match_data["cumulative_runs"], marker="o", label="Cumulative Runs")
    plt.xlabel("Over")
    plt.ylabel("Cumulative Runs")
    plt.title(f"Per-Over Score Prediction Chart for Match {match_id}")
    plt.grid(True)
    st.pyplot(plt)

if __name__ == "__main__":
    create_per_over_score_prediction_chart()
