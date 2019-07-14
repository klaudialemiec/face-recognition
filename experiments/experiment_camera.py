from classifiers import svc, knn, distance
from image_prepocessing.preprocessing import paint_detected_faces, start_video
from siamese_model.facenet import FacenetModel
from data_loaders import embedder_loader
from image_prepocessing import preprocessing
from embedder import embeddings_handler
import tensorflow as tf
import cv2


def threshold_test(x_tran, y_train, x_test, threshold):
    return distance.train_and_test(x_tran, y_train, x_test, threshold=threshold)


def knn_train_and_test(x_train, y_train, x_test):
    return knn.train_and_test(x_train, y_train, x_test)


def svc_train_and_test(x_train, y_train, x_test):
    return svc.train_and_test(x_train, y_train, x_test)


def show_result(y_pred, faces, image):
    result = paint_detected_faces(image.copy(), faces, y_pred)
    cv2.imshow('Result', result)
    cv2.waitKey(0)


def launch_camera_experiment():
    image, faces = start_video()
    if len(faces) > 0:
        print("[INFO] Detected " + str(len(faces)) + " faces.")
        with tf.Session() as sess:
            model = FacenetModel(sess)
            x_train, y_train = embedder_loader.load_embeddings_as_lists("..\\data\\embeddings\\embeddings_norm2.pkl")
            cv2.imshow('Marked faces', preprocessing.mark_faces(image.copy(), faces))

            image_processed = preprocessing.preprocess_image(image, 160, 160)
            for idx, img in enumerate(image_processed):
                cv2.imshow('After preprocessing ' + str(img), image_processed[idx])

            x_test = embeddings_handler.embeddings_for_faces(image_processed, model)

            y_pred = knn.tran_and_test_with_threshold(x_train, y_train, x_test, 2, 0.84)
            show_result(y_pred, faces, image)
    else:
        print("[WARNING] Camera cannot detect any face.")
