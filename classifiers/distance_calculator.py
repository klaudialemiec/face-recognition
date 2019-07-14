from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import unravel_index


def calculate_distance(embeddings1, embeddings2, distance_metric='euclidean'):
    if distance_metric == 'euclidean':
        dist = euclidean_distances(embeddings1, embeddings2)
    elif distance_metric == 'cosine':
        dist = cosine_similarity(embeddings1, embeddings2)
    else:
        raise '[ERROR] Undefined distance metric %d' % distance_metric
    return dist


def calculate_and_find_min_distance(embeddings1, embeddings2, distance_metric='euclidean'):
    distances = calculate_distance(embeddings1, embeddings2, distance_metric)
    results = [unravel_index(d.argmin(), d.shape) for d in distances]
    return results
