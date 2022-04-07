# Import svm model
from sklearn import svm

from load_data import load_data, evaluate_model


def svm_model(filename):
    X_train, X_test, y_train, y_test = load_data(filename=filename)
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    result = evaluate_model(y_test, y_pred, name='svm')
    result['filename'] = filename
    return result
