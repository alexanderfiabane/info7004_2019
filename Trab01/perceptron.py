import sys
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import multiprocessing
cores = multiprocessing.cpu_count()

def main(data):
    # X, y = load_digits(return_X_y=True)

    # loads data
    print("Loading data...")
    X_data, y_data = load_svmlight_file(data)

    # splits data
    print("Spliting data... ")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    clf = Perceptron(tol=1e-3, random_state=0, n_jobs=cores, early_stopping=True)
    clf.fit(X_train, y_train)
    # Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
    #       fit_intercept=True, max_iter=None, n_iter=None, n_iter_no_change=5,
    #       n_jobs=cores, penalty=None, random_state=0, shuffle=True, tol=0.001,
    #       validation_fraction=0.1, verbose=0, warm_start=False)
    print(clf.score(X_train, y_train))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: perceptron.py <data>")

    main(sys.argv[1])