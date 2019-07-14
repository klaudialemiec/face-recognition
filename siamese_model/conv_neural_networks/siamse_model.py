from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam


# We have 2 inputs, 1 for each picture
# left_input = Input((160, 160, 3))
# right_input = Input((160, 160, 3))

# We will use 2 instances of 1 network for this task
def create_model():
    convnet = Sequential([
        Conv2D(5, 3, input_shape=(160, 160, 3)),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(5, 3),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7, 2),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7, 2),
        Activation('relu'),
        Flatten(),
        Dense(128),
        Activation('sigmoid')
    ])
    return convnet


# # Connect each 'leg' of the network to each input
# # Remember, they have the same weights
# encoded_l = convnet(left_input)
# encoded_r = convnet(right_input)
#
# # Getting the L1 Distance between the 2 encodings
# L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
#
# # Add the distance function to the network
# L1_distance = L1_layer([encoded_l, encoded_r])
#
# prediction = Dense(1, activation='sigmoid')(L1_distance)
# siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
#
# optimizer = Adam(0.001, decay=2.5e-4)
# #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
# siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
#
# siamese_net.fit([left_input,right_input], targets,
#           batch_size=16,
#           epochs=30,
#           verbose=1,
#           validation_data=([test_left,test_right],test_targets))
