# Decision-Tree-and-Random-Forest
# Heart Disease Prediction Model 

**Overview**
This project aims to build a machine learning model that predicts whether a person has heart disease based on several health features. We will use Decision Trees and Random Forests, two powerful algorithms, to train the model and make predictions. The project will guide you through the process of:
Training a Decision Tree and visualizing it.
Evaluating the model using accuracy, cross-validation, and other metrics.
Training a Random Forest and comparing its performance.
Interpreting the results, such as feature importance.

**Requirements**

To get started, you'll need a few Python libraries. Install them using pip:
pip install pandas scikit-learn matplotlib seaborn

**1. Preparing the Data**
Before we train our model, we need to load the dataset. The dataset contains information such as:
Age
Cholesterol levels
Blood pressure
Presence of heart disease (our target variable)
We split the data into features (X) and the target variable (y). Features are the health-related information, and the target is whether the person has heart disease (1 = Yes, 0 = No).

**Example:**

import pandas as pd

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Split data into features and target
X = data.drop('target', axis=1)  # Features (everything except the target column)
y = data['target']  # Target (heart disease or not)
2. Training a Decision Tree
A Decision Tree is a model that splits the data into smaller groups based on the features, aiming to make predictions by following those splits.

**Unpruned Decision Tree:** A tree without any constraints can easily overfit the data (memorize it), but it may not generalize well to new data.

**Pruned Decision Tree:** A tree with certain constraints (like limiting the depth) helps prevent overfitting and makes the model generalize better.

**Example:**
from sklearn.tree import DecisionTreeClassifier
# Create the decision tree model
dt_model = DecisionTreeClassifier(max_depth=5)  # Limit tree depth to prevent overfitting

# Train the model
dt_model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = dt_model.score(X_train, y_train)
test_accuracy = dt_model.score(X_test, y_test)

**3. Visualizing the Decision Tree**

We can visualize the tree to see how it makes decisions based on the data. The tree will show the features and the decisions it uses to predict heart disease.
**Example:**
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=X.columns)
plt.show()

**4. Training a Random Forest**
A Random Forest is an ensemble of multiple Decision Trees. By averaging the predictions of many trees, it usually performs better than a single Decision Tree.

**Example:**
from sklearn.ensemble import RandomForestClassifier

# Create the random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)
5. Evaluating Model Performance
Accuracy
Accuracy tells us the percentage of correct predictions out of all predictions made. A high test accuracy (e.g., 99%) indicates that the model is doing well.

**Cross-Validation**
Cross-validation helps us test the model’s performance on different subsets of data. It provides a more reliable estimate of how the model will perform on unseen data.

**Example:**

from sklearn.model_selection import cross_val_score

# Perform cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.2f}")
Feature Importance
Feature importance tells us which features (e.g., age, cholesterol) are most important for making predictions.

Example:
importances = rf_model.feature_importances_
print("Feature Importances:", importances)
**6. Final Thoughts**
After training both models (Decision Tree and Random Forest), you should:

Compare their performance (accuracy, cross-validation score).

Evaluate the feature importances to understand which factors are influencing the predictions most.

Ensure that the model generalizes well to unseen data (test set), without overfitting to the training data.

**When is the model ready?**
If the model performs well on both the training and test data (e.g., high accuracy, stable performance across folds).

If the model generalizes well (i.e., it performs well on cross-validation and does not overfit).

If the results are interpretable and meaningful (i.e., the features used are reasonable and make sense in a real-world context).

# Next Steps
**Deploying the Model:** Once the model is ready, you can deploy it as a service (e.g., using Flask or FastAPI) so that it can be used for predictions in real time.

**Improving the Model:** If needed, try tuning hyperparameters further, experimenting with different models, or collecting more data to improve accuracy.

# Happy coding! 🎉
