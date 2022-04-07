from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from load_data import load_data, evaluate_model


def ensembleBaggingClassifier_model(filename):
    X_train, X_test, y_train, y_test = load_data(filename=filename)
    # initialized a 10-fold cross-validation fold
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    # instantiated a Decision Tree Classifier with 100 trees and wrapped it in a Bagging-based Ensemble
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)
    result = evaluate_model(y_test, y_pred, name="Ensemble BaggingClassifier")
    result['filename'] = filename
    return result
