import tensorflow as tf
import os

MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "\\pretrained_facenet\\"


class FacenetModel:

    def __init__(self, session):
        saver = tf.train.import_meta_graph(MODEL_PATH + "model-20180408-102900.meta")
        saver.restore(session, MODEL_PATH + "model-20180408-102900.ckpt-90")
        self.session = session

    def create_images_embeddings(self, images):

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array = self.session.run(embeddings, feed_dict=feed_dict)
        return emb_array
