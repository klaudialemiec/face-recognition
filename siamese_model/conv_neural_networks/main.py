from siamese_model.conv_neural_networks.siamse_model import create_model
from keras.models import Model
from keras.layers import Input
from keras.models import model_from_json
from siamese_model.siamese_network import prepare_triplet
from siamese_model.conv_neural_networks.TripleLossLayer import TripleLossLayer
from data_loaders import embedder_loader


def create_and_train_model(image_shape, data):
    model = create_model()

    # Input for anchor, positive and negative images
    in_anchor = Input(shape=image_shape)
    in_positive = Input(shape=image_shape)
    in_negative = Input(shape=image_shape)

    # Output for anchor, positive and negative embedding vectors
    # The nn4_small siamese_model instance is shared (Siamese network)
    emb_anchor = model(in_anchor)
    emb_positive = model(in_positive)
    emb_negative = model(in_negative)

    # Layer that computes the triplet loss from anchor, positive and negative embedding vectors
    triplet_loss_layer = TripleLossLayer(alpha=0.3, name='triplet_loss_layer')([emb_anchor, emb_positive, emb_negative])

    # Model that can be trained with anchor, positive negative images
    model_train = Model([in_anchor, in_positive, in_negative], triplet_loss_layer)
    generator = prepare_triplet(data)

    model_train.compile(loss=None, optimizer='adam',)
    model_train.fit_generator(generator, epochs=10, steps_per_epoch=100)
    return model_train


def save_model_to_file(model, model_path, weigths_path):
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weigths(weigths_path)


def load_model_from_file(model_path, weigths_path):
    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weigths(weigths_path)
    return loaded_model


def evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def main():

    data = embedder_loader.load_embeddings("../../data/embedder/images_and_embeddings.pkl")

    generator = prepare_triplet(data)
    i, _ = next(generator)

    model = create_and_train_model((160, 160, 3), data)
    save_model_to_file(model, "siamese_model.json", "siamese_model.h5")
    model = load_model_from_file("siamese_model.json", "siamese_model.h5")
    evaluate(model)

import numpy as np
a_batch = np.random.rand(4, 96, 96, 3)
print(a_batch)
main()
