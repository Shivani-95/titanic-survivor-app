import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load test data
df = pd.read_csv("test.csv")

# Save PassengerId for final output
passenger_ids = df['PassengerId']

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Fare'].fillna(df['Fare'].mean(), inplace=True)

# Drop unwanted columns
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Encode Sex
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Handle Embarked
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Scale Age and Fare
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Select same features as training
X_test = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Predict
predictions = model.predict(X_test)

# Create output DataFrame
output = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# Save to CSV
output.to_csv("submission.csv", index=False)
print("✅ Prediction complete. File saved as 'submission.csv'")
