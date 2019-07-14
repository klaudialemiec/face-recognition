import tensorflow as tf
from classifiers import distance_calculator
from numpy import unravel_index
from data_loaders import embedder_loader
from keras.layers import Input
from keras.models import Model
from random import randrange
import random
from siamese_model.triplet_loss_layer import TripletLossLayer
from keras.models import model_from_json
import numpy as np

IMAGE_SHAPE = (160, 160, 3)


def prepare_triplets(users_images_embeddings):
    all_embeddings = []

    for (user_images, embeddings) in users_images_embeddings.values():
        all_embeddings.extend(zip(user_images, embeddings))

    for user, user_data in users_images_embeddings.items():
        user_images, user_images_embeddings = user_data

        self_distances = distance_calculator.calculate_distance(user_images_embeddings, user_images_embeddings, 'euclidean')
        anchor_idx, positive_idx = unravel_index(self_distances.argmax(), self_distances.shape)
        anchor = user_images[anchor_idx]
        positive = user_images[positive_idx]

        all_embeddings_without_current_users = [x for x in all_embeddings if x[1] not in user_images_embeddings]

        all_embeddings_wo_users = [x[1] for x in all_embeddings_without_current_users]

        other_distances = distance_calculator.calculate_distance([user_images_embeddings[anchor_idx]], all_embeddings_wo_users, 'euclidean')
        negative_idx = other_distances.argmin()
        negative = all_embeddings_without_current_users[negative_idx][0]

        # print(distance_calculator.calculate_distance([user_images_embeddings[anchor_idx]], [all_embeddings_wo_users[negative_idx]], 'euclidean'))
        # print(distance_calculator.calculate_distance([user_images_embeddings[anchor_idx]], [user_images_embeddings[positive_idx]], 'euclidean'))

        yield anchor, positive, negative


def prepare_triplet(users_images_embeddings):
    all_embeddings = []

    for (user_images, embeddings) in users_images_embeddings.values():
        all_embeddings.extend(zip(user_images, embeddings))

    while True:
        users = random.sample(users_images_embeddings.items(), 4)

        anchors = np.zeros(shape=(4, 160, 160, 3))
        positives = np.zeros(shape=(4, 160, 160, 3))
        negatives = np.zeros(shape=(4, 160, 160, 3))


        i = 0;
        for user, user_data in users:
            user_images, user_images_embeddings = user_data

            self_distances = distance_calculator.calculate_distance(user_images_embeddings, user_images_embeddings,
                                                                    'euclidean')
            anchor_idx, positive_idx = unravel_index(self_distances.argmax(), self_distances.shape)
            anchor = user_images[randrange(len(user_images))]
            anchors[i] = anchor
            positive = user_images[randrange(len(user_images))]
            positives[i] = positive

            all_embeddings_without_current_users = [x for x in all_embeddings if x[1] not in user_images_embeddings]

            all_embeddings_wo_users = [x[1] for x in all_embeddings_without_current_users]

            other_distances = distance_calculator.calculate_distance([user_images_embeddings[anchor_idx]],
                                                                     all_embeddings_wo_users, 'euclidean')
            negative_idx = other_distances.argmin()
            negative = all_embeddings_without_current_users[randrange(len(all_embeddings_without_current_users))][0]
            negatives[i] = negative
            i += 1
        yield [anchor, positive, negative], None


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


def learn(model, embeddings):
    # Input for anchor, positive and negative images
    in_anchor = Input(shape=IMAGE_SHAPE)
    in_positive = Input(shape=IMAGE_SHAPE)
    in_negative = Input(shape=IMAGE_SHAPE)

    # Output for anchor, positive and negative embedding vectors
    # The nn4_small siamese_model instance is shared (Siamese network)
    emb_anchor = model(in_anchor)
    emb_positive = model(in_positive)
    emb_negative = model(in_negative)

    # Layer that computes the triplet loss from anchor, positive and negative embedding vectors
    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')(
        [emb_anchor, emb_positive, emb_negative])

    # Model that can be trained with anchor, positive negative images
    model_train = Model([in_anchor, in_positive, in_negative], triplet_loss_layer)
    generator = prepare_triplets(embeddings)

    model_train.compile(loss=None, optimizer='adam')
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


embeddings = embedder_loader.load_embeddings("../data/embedder/images_and_embeddings.pkl")
generator = prepare_triplets(embeddings)
import cv2
for i in range(0, 5):
    anchor, positive, negative = next(generator)
    cv2.imshow("Anchor " + str(i), anchor)
    cv2.imshow("Positive " + str(i), positive)
    cv2.imshow("Negative " + str(i), negative)
cv2.waitKey(0)


# from image_prepocessing import preprocessing
# import cv2
#
# images, _ = images_loader.load_images()
# #embedder = embedder_loader.load_embeddings("../data/embedder/images_and_embeddings.pkl")\
# images = next(iter(images.values()))
#
# tmp = images[0]
# tmp_p = preprocessing.preprocess_image(tmp, 160, 160)[0]
# cv2.imshow("pps", tmp_p)
# siamese_model = create_model()
# emb = siamese_model(Input(tmp_p))
# print(emb)


#trained_model = learn(siamese_model, embedder)
#save_model_to_file(trained_model, "./dupms/siamese_model.json", "./dupms/siamese_model.h5")
