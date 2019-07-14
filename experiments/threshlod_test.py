from sklearn.metrics import f1_score, accuracy_score
from data_loaders.embedder_loader import load_embeddings, load_embeddings_as_lists
from classifiers.distance_calculator import calculate_distance
import numpy as np
import matplotlib.pyplot as plt


distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise

embedded, names = load_embeddings_as_lists("..\\data\\embeddings\\embeddings_norm2.pkl")
num = len(embedded)

for i in range(num - 1):
    for j in range(1, num):
        tmp1 = embedded[i]
        tmp2 = embedded[j]
        distance = calculate_distance([embedded[i]], [embedded[j]], 'euclidean')[0][0]
        distances.append(distance)
        identical.append(1 if names[i] == names[j] else 0)
    print(i)

print('after for')
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.01, 1.0, 0.01)

f1_scores = [f1_score(identical, distances <= t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances <= t) for t in thresholds]

print('after acc')

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend()
plt.show()
