from tensorflow import keras
from tensorflow.keras import layers

def get_model(study_type):
    if study_type == 'cells':
        convHead1_channels = 5
    else:
        convHead1_channels = 4

    input = layers.Input(shape=(None, 128, 128, convHead1_channels))

    x = layers.ConvLSTM2D(
        filters = 32,
        kernel_size=(7,7),
        padding="same",
        return_sequences=True,
        activation="relu"
        )(input)

    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters = 64,
        kernel_size=(5,5),
        padding = "same",
        return_sequences = True,
        activation = "relu"
        )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters = 128,
        kernel_size=(3,3),
        padding = "same",
        return_sequences = True,
        activation = "relu"
        )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters = 128,
        kernel_size = (1,1),
        padding = "same",
        return_sequences = False,
        activation = "relu"
        )(x)

    x = layers.Dropout(rate=0.25)(x)


    output_1 = layers.Conv2D(filters=convHead1_channels, kernel_size=(3, 3), activation='sigmoid',
                       padding='same', name='convHead1')(x)

    output_2 = layers. Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid',
                       padding='same', name='convHead2')(x)

    model = keras.models.Model(input, [output_1,output_2])

    print(model.summary())
    return model
