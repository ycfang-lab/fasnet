from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def FasNet(input_num):
    """ for binary classification
    """
    input_video = Input(batch_shape=(None, input_num, 2048))
    encoded_video = Bidirectional(LSTM(128, return_sequences=True))(input_video)
    dense_video = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(Flatten()(encoded_video))
    dense_video = Dropout(0.5)(dense_video)
    real = Dense(1, activation='sigmoid', name='real')(dense_video)

    model = Model(inputs=[input_video], outputs=[real])

    model.compile(
        loss=['binary_crossentropy'],
        optimizer=Adam(lr=0.0001, decay=1e-6),
        metrics=['accuracy']
    )
    return model
