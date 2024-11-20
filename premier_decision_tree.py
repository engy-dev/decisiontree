import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('Fantasy Premier League.csv')

# Select relevant features
features = ['minutes', 'expected_goals', 'expected_assists', 'form_rank_type', 'influence_rank_type', 'creativity_rank_type', 'threat_rank_type']
X = data[features]
y = data['goals_scored'].apply(lambda x: 1 if x > 0 else 0)  # 1 if scored, 0 if not

# Handle missing values
X = X.fillna(0)

# Encode categorical variables
le = LabelEncoder()
for column in X.select_dtypes(include=['object']):
    X[column] = le.fit_transform(X[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth for better visualization
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Function to predict if a player will score
def predict_goal(minutes, expected_goals, expected_assists, form_rank_type, influence_rank_type, creativity_rank_type, threat_rank_type):
    prediction = clf.predict([[minutes, expected_goals, expected_assists, form_rank_type, influence_rank_type, creativity_rank_type, threat_rank_type]])
    return "Likely to score" if prediction[0] == 1 else "Unlikely to score"

# Example usage
player_prediction = predict_goal(90, 0.5, 0.2, 100, 150, 200, 50)
print(f"Prediction: {player_prediction}")

# Generate decision tree diagram
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=features, class_names=['No Goal', 'Goal'], filled=True, rounded=True)
plt.savefig('premier_league_goal_prediction_tree.png', dpi=300, bbox_inches='tight')
plt.close()

print("Decision tree diagram saved as 'premier_league_goal_prediction_tree.png'")