from helper_functions import read_data
from model_functions import train, feature_importances, confusion_matrix_evaluate, roc_curve_evaluate
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Read in data
    X, y = read_data()

    # Split into a training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Train a model
    model, vectorizer = train(X_train, y_train, X_test, y_test)

    # Plot feature importances
    feature_importances(model, vectorizer)

    # Plot confusion matrix
    confusion_matrix_evaluate(model, X_test, y_test)

    # Trains a OneVsRest classifier and plots ROC curve.
    roc_curve_evaluate(X_train, y_train, X_test, y_test, kfold=False)
