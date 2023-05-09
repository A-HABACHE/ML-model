import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the data
df = pd.read_csv('User_Data.csv')

# Split the data into features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_std, y_train)

# Predict on the test data
y_pred = clf.predict(X_test_std)

# # Compute the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# Save the trained model and scaler for future use
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
