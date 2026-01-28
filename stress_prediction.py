import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Encode categorical data
encoder = LabelEncoder()
data['exercise'] = encoder.fit_transform(data['exercise'])
data['stress_level'] = encoder.fit_transform(data['stress_level'])

# Features and target
X = data[['study_hours', 'sleep_hours', 'screen_time', 'exercise']]
y = data['stress_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Sample prediction
sample = [[5, 7, 4, 1]]  # study, sleep, screen, exercise
result = model.predict(sample)

print("Predicted Stress Level:", result[0])
