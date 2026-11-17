
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset/diabetes.csv")

print("Original Dataset:")
print(data.head())

# Check missing values
print("Missing Values:")
print(data.isnull().sum())

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_with_zero] = data[columns_with_zero].replace(0, np.nan)

print("\nMissing Values After Replacing Zeros:")
print(data.isnull().sum())

# Separate input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fill missing values using median
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# Check the results
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nMissing Values in Training Data:")
print(X_train.isnull().sum())

print("\nMissing Values in Testing Data:")
print(X_test.isnull().sum())

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")


# Define Multiple Algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

# Store algorithm results
results = []

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and Evaluate Each Algorithm
for name, algorithm in models.items():

    print("\nTraining:", name)

    algorithm.fit(X_train, y_train)

    y_pred = algorithm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy * 100
    })

    print("Accuracy:", accuracy * 100, "%")

    # Select the algorithm with highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = algorithm
        best_model_name = name

# Compare All Algorithms
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)
# Save algorithm comparison results
results_df.to_csv("models/model_comparison.csv", index=False)

print("\n========== ALGORITHM COMPARISON ==========")
print(results_df.to_string(index=False))

# Display Best Algorithm
print("\nBest Algorithm:", best_model_name)
print("Best Accuracy:", best_accuracy * 100, "%")

# Detailed Evaluation of Best Algorithm
best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))


# Save Best Model and All Trained Models
os.makedirs("models", exist_ok=True)

# Save the best model
joblib.dump(best_model, "models/diabetes_model.pkl")

# Save all five trained models
joblib.dump(models, "models/all_models.pkl")

# Save preprocessing objects
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save comparison results
results_df.to_csv("models/model_comparison.csv", index=False)

print("\nAll Models and Preprocessing Files Saved Successfully!")

print("\nBest Model Saved Successfully!")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model and preprocessing objects
joblib.dump(best_model, "models/diabetes_model.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and preprocessing objects saved successfully!")