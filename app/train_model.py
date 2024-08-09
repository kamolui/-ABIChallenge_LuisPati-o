import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

train_data = pd.read_csv('app/data/train.csv')

train_data = train_data[['Pclass', 'Age', 'Parch', 'SibSp', 'Fare', 'Survived', 'Sex']].dropna()

train_data['Sex'] = train_data['Sex'].apply(lambda x: 0 if x == 'female' else 1)

# Separate features and target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# save
joblib.dump(model, "app/models/model.pkl") 