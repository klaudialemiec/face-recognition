from keras import backend as K
from keras.layers import Layer


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        positive_dist = K.sum(K.square(anchor - positive), axis=-1)
        negative_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(positive_dist - negative_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
