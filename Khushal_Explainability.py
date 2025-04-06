#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd

# Import additional required libraries for machine learning and visualization
import numpy as np
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the training dataset
train = pd.read_excel('credit_train.xlsx')

# Display basic information about the dataset
train.info()

# Provide descriptive statistics of the dataset
train.describe()

# Extract features (drop the target variable 'approved')
x = train.drop('approved', axis=1)

# Display extracted features
x

# Extract target variable 'approved'
y = train['approved']

# Display extracted target variable
y

# Function to train and evaluate a Decision Tree classifier
def DT_explain(train_file, test_file):
    # Load training data
    train = pd.read_excel(train_file)
    x = train.drop('approved', axis=1)
    y = train['approved']
    
    # Initialize Decision Tree classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)  # Train the model
    
    # Load test data
    test = pd.read_excel(test_file)
    x_test = test.drop('approved', axis=1)
    y_test = test['approved']

    # Evaluate on training data
    print('Train Result')
    predictions = clf.predict(x)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Evaluate on test data
    print('Test Result')
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Visualize the decision tree
    plt.figure(figsize=(30, 20))  # Set figure size
    plot_tree(clf, filled=True, feature_names=x.columns, class_names=['Not Approved', 'Approved'], fontsize=8)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png", dpi=200, bbox_inches='tight')  # Save the tree plot
    plt.show()

# Run the Decision Tree function with train and test data
DT_explain('credit_train.xlsx', 'credit_test.xlsx')

# Function to train and evaluate a Logistic Regression model
def LR_explain(train_file, test_file):
    train = pd.read_excel(train_file)
    x = train.drop('approved', axis=1)
    y = train['approved']
    
    # Initialize Logistic Regression with a higher max_iter to avoid convergence warnings
    clf = LogisticRegression(max_iter=1000)
    x_old = x  # Keep a copy of original data before normalization
    x = normalize(x)  # Normalize the feature data
    std = StandardScaler()
    # x = std.fit_transform(x)  # Optional standardization (commented out)

    clf.fit(x, y)  # Train the model
    
    test = pd.read_excel(test_file)
    x_test = test.drop('approved', axis=1)
    x_test = normalize(x_test)  # Normalize the test data
    # x_test = std.transform(x_test)  # Optional standardization (commented out)
    y_test = test['approved']

    # Evaluate on training data
    print('Train Result')
    predictions = clf.predict(x)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Evaluate on test data
    print('Test Result')
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Print feature weights
    feature_names = x_old.columns
    feature_weights = clf.coef_[0]
    print("Feature Weights:")
    for feature, weight in zip(feature_names, feature_weights):
        print(f"{feature}: {weight:.4f}")

# Run Logistic Regression function with train and test data
LR_explain('credit_train.xlsx', 'credit_test.xlsx')

# Function to train and evaluate a Multi-layer Perceptron (MLP) classifier
def MLP_explain(train_file, test_file):
    train = pd.read_excel(train_file)
    x = train.drop('approved', axis=1)
    y = train['approved']

    # Initialize MLP with a hidden layer of 4 and 2 neurons
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=1, max_iter=1000)
    std = StandardScaler()
    x_old = x  # Keep original data
    x = std.fit_transform(x)  # Standardize data
    
    clf.fit(x, y)  # Train the model

    test = pd.read_excel(test_file)
    x_test = test.drop('approved', axis=1)
    x_test = std.transform(x_test)  # Standardize test data
    y_test = test['approved']

    # Evaluate on training data
    print('Train Result')
    predictions = clf.predict(x)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Evaluate on test data
    print('Test Result')
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Print model weights and biases
    for i, layer_weights in enumerate(clf.coefs_):
        print(f"Layer {i+1} Weights (shape {layer_weights.shape}):")
        print(layer_weights)
    for i, layer_biases in enumerate(clf.intercepts_):
        print(f"Layer {i+1} Biases (shape {len(layer_biases)}):")
        print(layer_biases)

    # Print feature-weight mapping
    print("Feature to First Layer Weights:")
    for feature, weights in zip(x_old, clf.coefs_[0]):
        print(f"{feature}: {weights}")

# Run the MLP function with train and test data
MLP_explain('credit_train.xlsx', 'credit_test.xlsx')