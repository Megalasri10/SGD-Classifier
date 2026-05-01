# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and split it into training and testing data.

2.Create and train the SGD Classifier using the training data.

3.Predict the Iris flower species using the trained model.

4.Evaluate the model using accuracy and classification metrics.

## Program:
```
/*
Developed by: Megala M S
RegisterNumber: 212225040230
# EX 07 - Implementation of Logistic Regression Using SGD Classifier

# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2: Create Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Define Features and Target
X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

# Step 4: Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 5: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create SGD Classifier Model
model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    learning_rate='optimal',
    random_state=42
)

# Step 7: Train the Model
model.fit(X_train, y_train)

# Step 8: Predict Test Data
y_pred = model.predict(X_test)

# Step 9: Evaluate the Model
print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 10: Predict for New Student Data
new_data = pd.DataFrame({
    'Hours_Studied': [7],
    'Attendance': [85]
})

# Scale New Data
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("\nNew Student Data:")
print(new_data)

if prediction[0] == 1:
    print("Prediction: Student Passed")
else:
    print("Prediction: Student Failed")

# Step 11: Visualization
plt.figure(figsize=(6, 4))

plt.scatter(df['Hours_Studied'], df['Pass'])

plt.xlabel("Hours Studied")
plt.ylabel("Pass / Fail")
plt.title("Logistic Regression using SGD Classifier")

plt.grid(True)
plt.show()
*/
```

## Output:
<img width="693" height="878" alt="image" src="https://github.com/user-attachments/assets/9c000bac-ca9e-46a2-a7f3-7375e399a6bc" />
<img width="817" height="492" alt="image" src="https://github.com/user-attachments/assets/156ce74e-9416-46d6-ad01-ab6f83e8939e" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
