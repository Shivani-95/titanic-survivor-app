import pandas as pd

# Step 1: Load CSV
df = pd.read_csv("train.csv")

# Step 2: Show top 5 rows
print("Top 5 Rows:")
print(df.head())

# Step 3: Show info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill Embarked with most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(['Cabin'], axis=1, inplace=True)


from sklearn.preprocessing import LabelEncoder

# Encode 'Sex' column (male=1, female=0)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encode 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scale 'Age' and 'Fare'
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


from sklearn.model_selection import train_test_split

# Select input features (X) and target (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrain and Test Set Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import joblib
joblib.dump(model, "titanic_model.pkl")
