# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**  
   Import the dataset to initiate the analysis.

2. **Explore Data**  
   Examine the dataset to identify patterns, distributions, and relationships.

3. **Select Features**  
   Determine the most important features to enhance model accuracy and efficiency.

4. **Split Data**  
   Separate the dataset into training and testing sets for effective validation.

5. **Train Model**  
   Use the training data to build and train the model.

6. **Evaluate Model**  
   Measure the model’s performance on the test data with relevant metrics.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Priyadharshini.P
RegisterNumber:  212223240128
*/

# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the provided URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
data = pd.read_csv(url)

# Step 2: Explore the dataset
# Display the first few rows and column names to verify the structure
print(data.head())
print(data.columns)

# Step 3: Select features and target variable
# Drop 'id' and other non-feature columns, using 'diagnosis' as the target
X = data.drop(columns=['Class'])  # Remove any irrelevant columns like 'id'
y = data['Class']  # The target column indicating benign or malignant diagnosis

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize and train the Decision Tree model
# Create a Decision Tree Classifier and fit it on the training data
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
# Predict on the test set and evaluate the results
y_pred = model.predict(X_test)

# Print the accuracy and classification metrics for the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Confusion Matrix
# Generate a heatmap of the confusion matrix for better visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
<img width="671" alt="Screenshot 2024-11-14 at 11 32 31 AM" src="https://github.com/user-attachments/assets/b4928d71-0dfd-485a-9ccc-0a5a75222352">


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
