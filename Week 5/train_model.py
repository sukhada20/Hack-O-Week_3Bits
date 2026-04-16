import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("hvac_data.csv")

# Convert text labels to numbers
le_lab = LabelEncoder()
df["lab_id"] = le_lab.fit_transform(df["lab_id"])

le_cool = LabelEncoder()
df["cooling_needed"] = le_cool.fit_transform(df["cooling_needed"])

# Features and target
X = df[["lab_id", "occupancy", "temperature"]]
y = df["cooling_needed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Test one prediction
sample = [[1, 30, 29]]  # Lab2, 30 students, 29°C
prediction = model.predict(sample)
result = le_cool.inverse_transform(prediction)

print("Sample prediction:", result[0])
