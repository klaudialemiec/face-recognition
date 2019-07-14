from sklearn.neighbors import KNeighborsClassifier


def train_and_test(x_train, y_train, x_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def tran_and_test_with_threshold(x_train, y_train, x_test, k, threshold):
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    classifier.fit(x_train, y_train)
    results = []

    for x in x_test:
        if 'numpy.ndarray' in str(type(x)):
            distances, neighbors = classifier.kneighbors(X=x, n_neighbors=k, return_distance=True)
        else:
            distances, neighbors = classifier.kneighbors(X=[x], n_neighbors=k, return_distance=True)
        distances = distances[0].tolist()
        neighbors = neighbors[0].tolist()

        min_distance = min(distances)
        min_distance_idx = distances.index(min_distance)

        if min_distance > threshold:
            results.append('Unknown')
        else:
            # tmp = classifier.predict_proba([x])
            # print(max(tmp[0]))
            # distance_avg = np.mean(distances)
            # if distance_avg > threshold:
            #     results.append('Unknown')
            # else:
            results.append(y_train[neighbors[min_distance_idx]])
    return results

