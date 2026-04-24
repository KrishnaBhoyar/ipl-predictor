import streamlit as st
import pandas as pd
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# config
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f1f5f9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Title */
h1 {
    text-align: center;
    color: #38bdf8;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}

/* Button hover */
div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* Section headings */
h2 {
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# load model
model = pickle.load(open("model.pkl", "rb"))

# teams
teams = ["CSK","MI","RCB","KKR","DC","RR","SRH","PBKS","LSG","GT"]

team_logos = {
    "CSK": "CSK.PNG",
    "MI": "MI.PNG",
    "RCB": "RCB.PNG",
    "KKR": "KKR.PNG",
    "DC": "DC.PNG",
    "RR": "RR.PNG",
    "SRH": "SRH.PNG",
    "PBKS": "PBKS.PNG",
    "LSG": "LSG.PNG",
    "GT": "GT.PNG"
}

# load logo
def load_logo(team):
    path = os.path.join(ASSETS_DIR, team_logos.get(team))
    if os.path.exists(path):
        return Image.open(path)
    return None

# chart
def show_chart(team1, team2, p1, p2):
    fig, ax = plt.subplots()
    bars = ax.bar([team1, team2], [p1, p2])
    ax.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{int(h)}%", ha='center')

    st.pyplot(fig)

# UI
st.title("🏏 IPL Match Win Predictor")

# sidebar
team1 = st.sidebar.selectbox("Team 1", teams)
team2 = st.sidebar.selectbox("Team 2", teams)

if team1 == team2:
    st.error("Select different teams")
    st.stop()

toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2])
toss_decision = st.sidebar.radio("Toss Decision", ["Bat", "Field"])

# display teams
col1, col2 = st.columns(2)

with col1:
    logo1 = load_logo(team1)
    if logo1:
        st.image(logo1, width=120)
    st.subheader(team1)

with col2:
    logo2 = load_logo(team2)
    if logo2:
        st.image(logo2, width=120)
    st.subheader(team2)

st.markdown("## 🔮 Prediction")

# predict
if st.button("Predict Winner"):

    try:
        # create input
        input_df = pd.DataFrame({
            "home_team": [team1],
            "away_team": [team2],
            "toss_winner": [toss_winner],
            "toss_decision": [toss_decision]
        })

        # one-hot encode
        input_encoded = pd.get_dummies(input_df)

        # match training columns
        model_columns = model.feature_names_in_

        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # reorder
        input_encoded = input_encoded[model_columns]

        # prediction
        prob = model.predict_proba(input_encoded)

        p1 = int(prob[0][1] * 100)
        p2 = 100 - p1

        winner = team1 if p1 > p2 else team2
        st.success(f"🏆 Predicted Winner: {winner}")

        # results
        col3, col4 = st.columns(2)

        with col3:
            if logo1:
                st.image(logo1, width=100)
            st.metric(team1, f"{p1}%")

        with col4:
            if logo2:
                st.image(logo2, width=100)
            st.metric(team2, f"{p2}%")


    except Exception as e:
        st.error(f"Prediction error: {e}")