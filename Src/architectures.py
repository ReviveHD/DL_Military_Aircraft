from tensorflow.keras import layers, models


# ==============================================================================
# AeroConv 1: Die Baseline (Die Flatten-Falle)
# ==============================================================================
def build_aeroconv1(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_1")
    model.add(layers.Input(shape=input_shape))

    # Pixelwerte auf [0, 1] normalisieren
    model.add(layers.Rescaling(1. / 255))

    # Klassischer VGG-artiger Backbone
    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    # DIE FLATTEN-FALLE: Verursacht extreme Parameter-Explosion und Overfitting
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# ==============================================================================
# AeroConv 2: Die Rettung durch GAP
# ==============================================================================
def build_aeroconv2(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_2")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1. / 255))

    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    # Reduziert Dimensionen über den Mittelwert; verhindert Parameter-Explosion
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# ==============================================================================
# AeroConv 3: Einführung von Dropout
# ==============================================================================
def build_aeroconv3(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_3")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1. / 255))

    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    # Regularisierung zur Vermeidung von Overfitting
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# ==============================================================================
# AeroConv 4: Einführung von Batch Normalization
# ==============================================================================
def build_aeroconv4(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_4")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1. / 255))

    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        # Normalisiert Layer-Ausgaben; stabilisiert und beschleunigt das Training
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# ==============================================================================
# AeroConv 5 bis 9: Einführung von Data Augmentation
# ==============================================================================
def build_aeroconv5(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_5")
    model.add(layers.Input(shape=input_shape))

    # Bildvariationen zur künstlichen Vergrößerung des Datensatzes
    model.add(layers.Rescaling(1. / 255))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomContrast(0.1))

    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    for units, drop_rate in [(512, 0.5), (256, 0.4)]:
        model.add(layers.Dense(units, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def build_aeroconv6(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_6")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Rescaling(1. / 255))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomContrast(0.1))
    model.add(layers.RandomZoom(0.1))

    # Filterkapazität erhöht (Start bei 32 statt 16) gegen das Underfitting von V5
    for filters in [32, 64, 128, 256, 512]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    for units, drop_rate in [(512, 0.5), (256, 0.4)]:
        model.add(layers.Dense(units, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def build_aeroconv7(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_7")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Rescaling(1. / 255))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomContrast(0.1))
    model.add(layers.RandomZoom(0.3))  # Aggressiver Zoom (Merkmale fielen aus dem Bild)

    for filters in [32, 64, 128, 256, 512]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    for units, drop_rate in [(512, 0.5), (256, 0.4)]:
        model.add(layers.Dense(units, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def build_aeroconv8(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_8")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Rescaling(1. / 255))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomContrast(0.1))
    model.add(layers.RandomZoom(0.3))

    # Initialer 7x7 Filter zur Erfassung makroskopischer Flugzeugkonturen
    model.add(layers.Conv2D(32, (7, 7), padding='same', activation='relu'))

    for filters in [32, 64, 128, 256, 512]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    for units, drop_rate in [(512, 0.5), (256, 0.4)]:
        model.add(layers.Dense(units, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def build_aeroconv9(input_shape=(256, 256, 3), num_classes=28):
    model = models.Sequential(name="AeroConv_9")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Rescaling(1. / 255))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomContrast(0.1))
    model.add(layers.RandomZoom(0.3))

    # Filter-Stapelung am Netzanfang für multi-skalierte Geometrien
    model.add(layers.Conv2D(32, (7, 7), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))

    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    for filters in [64, 128, 256, 512]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    for units, drop_rate in [(512, 0.5), (256, 0.4)]:
        model.add(layers.Dense(units, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# ==============================================================================
# AeroConv 10: Die finale State-of-the-Art Architektur
# ==============================================================================

# Squeeze-and-Excitation (Kanal-Aufmerksamkeit)
def se_block(x, filters, ratio=16):
    # Squeeze: Globales Pooling komprimiert räumliche Dimensionen auf 1x1
    se = layers.GlobalAveragePooling2D()(x)

    # Excitation: Lernt die dynamische Gewichtung (Wichtigkeit) der Kanäle
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    se = layers.Reshape((1, 1, filters))(se)
    # Kanäle elementweise mit gelernten Gewichten skalieren
    return layers.Multiply()([x, se])


# Hauptbaustein: Kombiniert Asymmetrische Faltung, SE-Block und Residuals
def aero_block(x, filters, stride=1):
    shortcut = x

    # Standard-Faltung
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Asymmetrische Faltung: Faktorisiert 5x5 in sequentielles 1x5 und 5x1 (Spart 60% Parameter)
    x = layers.Conv2D(filters, (1, 5), padding='same', use_bias=False)(x)
    x = layers.Conv2D(filters, (5, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Kanal-Attention anwenden
    x = se_block(x, filters)

    # Anpassung des Shortcuts bei Dimensionsänderungen (Linear Projection)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Skip-Connection: Verhindert das Verschwinden des Gradienten
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# Definition des Gesamtmodells via Functional API
def build_aeroconv10(input_shape=(256, 256, 3), num_classes=28):
    inputs = layers.Input(shape=input_shape)

    # Integrierte Data Augmentation (Schutz vor Overfitting)
    x = layers.Rescaling(1. / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = layers.RandomZoom(0.2)(x)

    # Stem: Initiale Bildreduktion
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Body: Stapelung der Aero-Blöcke mit ansteigender Kanaltiefe
    x = aero_block(x, filters=64, stride=1)
    x = aero_block(x, filters=128, stride=2)
    x = aero_block(x, filters=256, stride=2)
    x = aero_block(x, filters=512, stride=2)
    x = aero_block(x, filters=512, stride=2)

    x = layers.GlobalAveragePooling2D()(x)

    # Klassifikationskopf mit starkem Dropout wegen hoher Netzkapazität
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.7)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="AeroConv10")
    return model