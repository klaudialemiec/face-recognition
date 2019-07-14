from keras.layers import Layer
from siamese_model import siamese_network


class TripleLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripleLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        return siamese_network.triplet_loss([], inputs, alpha=self.alpha)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
