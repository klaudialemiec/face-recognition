from classifiers.distance_calculator import calculate_and_find_min_distance


def train_and_test(x_train, y_train, x_test, threshold):
    labels_pred = []
    y_pred = calculate_and_find_min_distance(x_test, x_train, threshold=threshold)
    for y in y_pred:
        if y == -1:
            labels_pred.append('Unknown')
        else:
            labels_pred.append(y_train[y[0]])
    return labels_pred
