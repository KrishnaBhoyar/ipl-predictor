import streamlit as st
import pandas as pd
import os
from PIL import Image
import pickle

# config
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

# style
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1 {
    text-align: center;
    color: #38bdf8;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
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

# logo loader
def load_logo(team):
    path = os.path.join(ASSETS_DIR, f"{team}.png")
    if os.path.exists(path):
        return Image.open(path)
    return None


# title
st.title("🏏 IPL Match Win Predictor")
st.markdown("---")

# sidebar
team1 = st.sidebar.selectbox("Team 1", teams)
team2 = st.sidebar.selectbox("Team 2", teams)

venue = st.sidebar.selectbox(
    "Venue",
    ["Mumbai", "Delhi", "Chennai", "Bangalore", "Hyderabad"]
)

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
        st.image(logo1, width=150)
    st.subheader(team1)

with col2:
    logo2 = load_logo(team2)
    if logo2:
        st.image(logo2, width=150)
    st.subheader(team2)

st.markdown("## 🔮 Prediction")

# prediction
if st.button("Predict Winner"):

    try:
        with st.spinner("Analyzing match..."):

            # IMPORTANT: venue not used in model yet (safe)
            input_df = pd.DataFrame({
                "home_team": [team1],
                "away_team": [team2],
                "toss_winner": [toss_winner],
                "toss_decision": [toss_decision]
            })

            # encoding
            input_encoded = pd.get_dummies(input_df)

            model_columns = model.feature_names_in_

            for col in model_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            input_encoded = input_encoded[model_columns]

            # prediction
            prob = model.predict_proba(input_encoded)

            p1 = int(prob[0][1] * 100)
            p2 = 100 - p1

            winner = team1 if p1 > p2 else team2

            # confidence
            confidence = "High" if max(p1, p2) > 70 else "Medium" if max(p1, p2) > 55 else "Low"

            st.success(f"""
🏆 {winner} has {max(p1,p2)}% chance of winning  
Confidence Level: {confidence}
""")

            col3, col4 = st.columns(2)

            with col3:
                if logo1:
                    st.image(logo1, width=120)
                st.metric(team1, f"{p1}%")

            with col4:
                if logo2:
                    st.image(logo2, width=120)
                st.metric(team2, f"{p2}%")


            # insights
            st.markdown("### 📊 Match Insights")

            if toss_winner == winner:
                st.write("✔ Toss advantage impacted the prediction")

            if toss_decision == "Field":
                st.write("✔ Chasing teams generally perform better")

            st.write(f"📍 Match Venue: {venue}")

    except Exception as e:
        st.error(f"Prediction error: {e}")