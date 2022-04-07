from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from load_data import load_data, evaluate_model


def knn_model(filename):
    params = {
        'n_neighbors': range(1, 15, 2),
        'p': [1, 2],
        'weights': ['uniform', 'distance']
    }

    knn_model = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=params,
        cv=5,
        n_jobs=5,
        verbose=1,
    )
    X_train, X_test, y_train, y_test = load_data(filename=filename)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    result = evaluate_model(y_test, y_pred, name='knn')
    result['filename'] = filename
    return result
