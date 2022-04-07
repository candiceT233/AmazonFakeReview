import os
import numpy as np
import pandas as pd

# BASE_PATH = '/content/drive/MyDrive/cs579/CS579-Project2/final_code/datasets'
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_PATH = './datasets'
FILENAME = 'amazon_ml_2.csv'
FEATURES = ['goodness', 'fairness']
LABEL = ['is_trust']


def load_data(filename=FILENAME):
    df = pd.read_csv(
        os.path.join(BASE_PATH, filename),
    )
    # datasets = pd.DataFrame(df).head(4000)
    datasets = pd.DataFrame(df)
    X = datasets[FEATURES]
    y = datasets[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def evaluate_model(y_test, y_pred, name='knn'):
    # evaluate model
    # Model Accuracy: how often is the classifier correct?
    accuracy = accuracy_score(y_test, y_pred)
    # Model Precision: what percentage of positive tuples are labeled as such?
    precision = metrics.precision_score(y_test, y_pred)
    # Model Recall: what percentage of positive tuples are labelled as such?
    recall = metrics.recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(name.center(50, '*'))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    results = {
        'name': name,
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3)
    }
    print('results:', results)
    return results
