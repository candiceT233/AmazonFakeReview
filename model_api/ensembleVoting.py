from random import seed

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from load_data import load_data, evaluate_model


def ensembleVoting_model(filename):
    X_train, X_test, y_train, y_test = load_data(filename=filename)
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    model = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    result = evaluate_model(model, y_pred, name="Ensemble Voting")
    result['filename'] = filename
    return result
