from image_prepocessing.preprocessing import start_video
from siamese_model.facenet import FacenetModel
from data_loaders import embedder_loader
from image_prepocessing import preprocessing
from embedder import embeddings_handler
import tensorflow as tf
import cv2


def launch_add_person_experiment_camera(person_name):
    image, faces = start_video()
    add_person(image, faces, person_name)


def launch_add_person_experiment_file(person_name, path):
    image = cv2.imread(path)
    faces = preprocessing.detect_faces(image)
    add_person(image, faces, person_name)


def add_person(image, faces, person_name):
    if len(faces) > 0:
        print("[INFO] Detected " + str(len(faces)) + " faces. Adding only most middle one.")
        cv2.imshow('Found faces', preprocessing.mark_faces(image.copy(), faces))

        face = preprocessing.preprocess_image(image, 160, 160)[0]
        cv2.imshow('After preprocessing', face)

        session = tf.Session()
        model = FacenetModel(session)
        embedding = embeddings_handler.embedding_for_face(face, model)
        session.close()
        embedder_loader.add_to_existing_embeddings((person_name, embedding))
        print("[INFO] Face added to base.")
    else:
        print("[WARNING] Camera cannot detect any face.")
