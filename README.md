# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**: Import the dataset using `pandas` and load the tumor data from the given file path.

2. **Check for Missing Values**: Check if there are any missing values in the dataset using `isnull().sum()`.

3. **Handle Missing Values**: If there are any missing values, remove them using `dropna()`.

4. **Separate Features and Target Variable**: 
   - Separate the feature variables `X` (all columns except 'Class').
   - Set the target variable `y` as the 'Class' column.

5. **Split Data**: Divide the dataset into training (80%) and testing (20%) sets using `train_test_split()`.

6. **Initialize Decision Tree Classifier**: Initialize the `DecisionTreeClassifier` model with a fixed random state.

7. **Train the Model**: Fit the `DecisionTreeClassifier` model on the training data.

8. **Make Predictions**: Use the trained model to predict on the test data.

9. **Evaluate the Model**: 
   - **Confusion Matrix**: Calculate and print the confusion matrix.
   - **Classification Report**: Print the classification report with precision, recall, and f1-score.
   - **Accuracy Score**: Print the accuracy of the model on the test data.

10. **Visualize the Confusion Matrix**: Create a heatmap using `seaborn` to visualize the confusion matrix with appropriate labels for "Benign" and "Malignant" classes.

11. **Display Decision Tree Rules**: Print out the decision tree rules using `export_text()` to show the classification logic.

12. **Visualize the Decision Tree**: Plot the decision tree using `plot_tree()` to visually represent how the decision-making process works.
 

## Program:
```python
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Priyadharshini.P
RegisterNumber: 21223240128
*/
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\admin\Downloads\tumor.csv"  # Replace with your file path if necessary
data = pd.read_csv(file_path)

# Check for missing values
print("Checking for missing values...")
print(data.isnull().sum())

# Handle missing values (if any exist)
if data.isnull().values.any():
    data = data.dropna()

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Display the decision tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))  # Adjust figure size as needed
plot_tree(clf, feature_names=X.columns, class_names=["Benign", "Malignant"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
```


## Output:

![image](https://github.com/user-attachments/assets/fe1ae58b-adda-4852-8687-a719c02b269c)


![image](https://github.com/user-attachments/assets/b5f70b4b-4551-462d-a486-d9434d25257e)

![image](https://github.com/user-attachments/assets/05d084b9-0835-4766-816d-432a9217fdb5)

![image](https://github.com/user-attachments/assets/d1dca6b3-62bc-4a0e-bbb4-3aa8a9f5ae2c)





## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
stallation / Jupyter notebook

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
