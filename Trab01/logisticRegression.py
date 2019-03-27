import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from operator import itemgetter
from sklearn.linear_model import LogisticRegression

def main(representacao):
    # loads data
    print("Loading data...")
    X_data, y_data = load_svmlight_file(representacao)

    best_fits = []
    # splits data
    print("Spliting data... ")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2',
                       random_state=None, tol=0.0001)
    print(classifier.score(X_test, y_test))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: logisticRegression.py <name_file>")

    main(sys.argv[1])
    #main()#representacao_train_d2v_v1000_w20_mc3