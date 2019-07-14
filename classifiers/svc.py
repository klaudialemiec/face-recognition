from sklearn.svm import SVC


def train_and_test(x_train, y_train, x_test):
    svc = SVC(C=1.0, kernel="linear", probability=True)
    svc.fit(x_train, y_train)
    return svc.predict(x_test)
