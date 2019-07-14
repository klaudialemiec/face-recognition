from classifiers import svc, knn, distance
from image_prepocessing.preprocessing import detect_and_paint_face
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from siamese_model.facenet import FacenetModel
from image_prepocessing import preprocessing, image_modifiers
from data_loaders import embedder_loader, images_loader
from embedder import embeddings_handler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import glob
import cv2


def threshold_test(x_tran, y_train, x_test, threshold):
    return distance.train_and_test(x_tran, y_train, x_test, threshold=threshold)


def knn_train_and_test(x_train, y_train, x_test):
    return knn.train_and_test(x_train, y_train, x_test)


def svc_train_and_test(x_train, y_train, x_test):
    return svc.train_and_test(x_train, y_train, x_test)


def print_result(y_pred, test_images_path):
    test_images = glob.glob(test_images_path + '*/*.jpg')
    i = 0
    for image_path, label in zip(test_images, y_pred):
        image = cv2.imread(image_path)
        result = detect_and_paint_face(image, label)
        if result is not None:
            cv2.imshow(label + str(i), result)
        i += 1
    cv2.waitKey(0)


def show_result(y_pred, faces, image, window_name):
    result = preprocessing.paint_detected_faces(image.copy(), faces, y_pred)
    cv2.imshow(window_name, result)


def evaluate(y_true, y_pred):
    encoder = LabelEncoder()
    encoder.fit(np.concatenate((y_true, y_pred), axis=None))
    labels_true = encoder.transform(y_true)
    labels_pred = encoder.transform(y_pred)
    return accuracy_score(labels_true, labels_pred), f1_score(labels_true, labels_pred, average='macro')


def launch_known_faces_experiment():
    with tf.Session() as sess:
        model = FacenetModel(sess)
        x_train, y_train = embedder_loader.load_embeddings_as_lists("..\\data\\embeddings\\embeddings_norm2.pkl")

        names, images = images_loader.load_images_as_list("..\\data\\test\\")
        images_processed = preprocessing.preprocess_images(images, 160, 160)

        images_processed_not_empty = []
        for idx, image in enumerate(images_processed):
            if image:
                images_processed_not_empty.append(image)
            else:
                print("Face not detected in image " + names[idx])

        x_test = [embeddings_handler.embeddings_for_faces(i, model) for i in images_processed_not_empty]
        y_pred = knn.tran_and_test_with_threshold(x_train, y_train, x_test, 1, 0.84)
        print(y_pred)
        print(evaluate(names, y_pred))

        for idx2, image2 in enumerate(images):
            faces = preprocessing.detect_faces(image2)
            show_result(y_pred, faces, image2, str(idx2))
        cv2.waitKey(0)


def launch_unknown_faces_experiment():
    with tf.Session() as sess:
        model = FacenetModel(sess)
        x_train, y_train = embedder_loader.load_embeddings_as_lists("..\\data\\embeddings\\embeddings_norm2.pkl")

        names, images = images_loader.load_images_as_list("..\\data\\test\\")
        images_processed = preprocessing.preprocess_images(images, 160, 160)

        images_processed_not_empty = []
        for idx, image in enumerate(images_processed):
            if image:
                images_processed_not_empty.append(image)
            else:
                print("Face not detected in image " + names[idx])

        x_test = embeddings_handler.embeddings_for_faces(images_processed_not_empty, model)

        y_test = ["Unknown"] * len(x_test)
        y_pred = knn.tran_and_test_with_threshold(x_train, y_train, x_test, 1, 0.84)
        print(evaluate(y_test, y_pred))


def launch_standard_experiment():
    print("[INFO] Starting standard experiment: standard preprocessing, whole embeddings base, kNN & SVC classfiers")
    x_train, x_test, y_train, y_test = embedder_loader.load_embeddings_and_split("data/embeddings/embeddings_norm2.pkl")
    print("[INFO] Embeddings base loaded. Containing " + str(len(x_train)) + " embeddings.")
    print("[INFO] Test embeddings loaded. Containing " + str(len(x_test)) + " embeddings.")

    print("[INFO] Running k-NN classifier...")
    y_pred_knn = knn.train_and_test(x_train, y_train, x_test, 1)
    knn_acc, knn_f1 = evaluate(y_test, y_pred_knn)
    print("[INFO] k-NN classifier score: accuracy - " + str(knn_acc) + ", f1 - " + str(knn_f1))

    print("[INFO] Running SVC classifier...")
    y_pred_svc = svc.train_and_test(x_train, y_train, x_test)
    svc_acc, svc_f1 = evaluate(y_test, y_pred_svc)
    print("[INFO] SVC classifier score: accuracy - " + str(svc_acc) + ", f1 - " + str(svc_f1))


def launch_modified_experiment(modifier):
    print("[INFO] Starting experiment: " + modifier + " modifier, kNN classifier")

    persons, images = images_loader.load_images_as_list("data/lfw/", 4)
    print("[INFO] Images loaded.")
    print(len(images))

    print("[INFO] Preprocessing...")
    if modifier == 'grayscale':
        images = [image_modifiers.image_rgb_to_grayscale(image) for image in images]
    elif modifier == 'blur':
        images = [image_modifiers.blur_image(image) for image in images]
    elif modifier == 'equalize':
        images = [image_modifiers.equalize_histogram(image) for image in images]
    elif modifier == 'gamma':
        images = [image_modifiers.adjust_gamma(image) for image in images]
    elif modifier == 'gamma_grayscale':
        images = [image_modifiers.adjust_gamma(image) for image in images]
        images = [image_modifiers.image_rgb_to_grayscale(image) for image in images]
    elif modifier == 'histogram_grayscale':
        images = [image_modifiers.image_rgb_to_grayscale(image) for image in images]
        images = [image_modifiers.equalize_histogram(image) for image in images]

    left_persons, preprocessed_faces = [], []
    for person, image in zip(persons, images):
        faces = [preprocessing.resize_image(image, 160, 160)]
        #faces = preprocessing.preprocess_image(image, 160, 160)
        if len(faces) > 0:
            left_persons.append(person)
            preprocessed_faces.append(faces[0])
    print("[INFO] Preprocessing finished.")
    print(len(preprocessed_faces))

    print("[INFO] Loading model...")
    sess = tf.Session()
    model = FacenetModel(sess)
    print("[INFO] Model loaded.")
    faces_embeddings = [embeddings_handler.embedding_for_face(face, model) for face in preprocessed_faces]
    sess.close()

    base_embeddings, test_embeddings, base_persons, test_persons = \
        train_test_split(faces_embeddings, left_persons, train_size=0.8, stratify=left_persons)
    print("[INFO] Embeddings base created. Containing " + str(len(base_embeddings)) + " embeddings.")
    print("[INFO] Test embeddings created. Containing " + str(len(test_embeddings)) + " embeddings.")

    persons_pred_knn = knn.train_and_test(base_embeddings, base_persons, test_embeddings, 1)
    knn_acc, knn_f1 = evaluate(test_persons, persons_pred_knn)
    print("[INFO] k-NN classifier score: accuracy - " + str(knn_acc) + ", f1 - " + str(knn_f1))

    persons_pred_svc = svc.train_and_test(base_embeddings, base_persons, test_embeddings)
    svc_acc, svc_f1 = evaluate(test_persons, persons_pred_svc)
    print("[INFO] SVC classifier score: accuracy - " + str(svc_acc) + ", f1 - " + str(svc_f1))