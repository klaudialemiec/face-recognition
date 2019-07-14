import pickle
import os
from sklearn.model_selection import train_test_split

DEFAULT_PATH = "data/embeddings/embeddings_norm2.pkl"


def save_all_embeddings(embeddings, path=DEFAULT_PATH):
    with open(path, 'wb') as save_file:
        pickle.dump(embeddings, save_file, pickle.HIGHEST_PROTOCOL)


def add_to_existing_embeddings(new_user_embeddings, path_to_existing=DEFAULT_PATH):
    if os.path.isfile(path_to_existing):
        with open(path_to_existing, 'r+b') as embeddings_file:
            data = pickle.load(embeddings_file)
            data[new_user_embeddings[0]] = new_user_embeddings[1]
            pickle.dump(data, embeddings_file, pickle.HIGHEST_PROTOCOL)


def load_embeddings(path=DEFAULT_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_embeddings_and_split(path=DEFAULT_PATH):
    embeddings, names = load_embeddings_as_lists(path)
    trainX, testX, trainY, testY = train_test_split(embeddings, names, train_size=0.8, random_state=24, stratify=names)
    return trainX, testX, trainY, testY


def load_embeddings_as_lists(path=DEFAULT_PATH):
    embeddings_dict = load_embeddings(path)
    names = []
    embeddings = []
    for user, embs in embeddings_dict.items():
        if len(embs) > 1:
            embeddings.extend(embs)
            names.extend([user] * len(embs))
    return embeddings, names
