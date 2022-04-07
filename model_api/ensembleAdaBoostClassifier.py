from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

from load_data import load_data, evaluate_model


def ensembleAdaBoostClassifier_model(filename):
    seed = 7
    num_trees = 70
    X_train, X_test, y_train, y_test = load_data(filename=filename)
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    result = evaluate_model(y_test, y_pred, name="Ensemble AdaBoostClassifier")
    result['filename'] = filename
    return result
