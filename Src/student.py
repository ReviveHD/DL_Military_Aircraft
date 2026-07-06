from tensorflow.keras import layers, models


def residual_block(x, filters, stride=1):
    shortcut = x

    # Pfad 1
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut anpassen, falls Filteranzahl oder Größe sich ändert
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_aeroconv_residual(input_shape=(224, 224, 3), num_classes=28):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)

    # Initialer Conv-Block
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual Blöcke
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # Dropout etwas reduziert
    x = layers.Dense(512, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation=None)(x)
    return models.Model(inputs, outputs, name="AeroConv_Residual")