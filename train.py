import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("Cricket_data.csv")

# 2. Check columns
print("Columns in dataset:")
print(df.columns)

# 3. CLEAN DATA (IMPORTANT - do first)
df = df.dropna(subset=[
    'home_team', 'away_team', 'toss_won', 'decision', 'venue_name', 'winner'
])

# 4. Select features
X = df[['home_team', 'away_team', 'toss_won', 'decision', 'venue_name']]
y = df['winner']

# 5. Convert categorical → numeric
X = pd.get_dummies(X)
y = pd.get_dummies(y).idxmax(axis=1)

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model (improved)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")