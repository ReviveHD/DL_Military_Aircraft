import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50,
    ResNet101,
    ResNet152
)
from tensorflow.keras.applications.resnet import preprocess_input

def build_dynamic_resnet(r_variant='50', input_shape=(300, 300, 3), num_classes=28):
    # 1. Dictionary für die automatische Modellauswahl
    model_selector = {
        '50': ResNet50,
        '101': ResNet101,
        '152': ResNet152
    }

    # Wähle die Klasse aus (Standard ist ResNet50, falls was falsch übergeben wird)
    BaseModelClass = model_selector.get(str(r_variant), ResNet50)
    print(f"Lade Architektur: ResNet{r_variant} (V1)...")

    # 2. Das vortrainierte Basis-Modell laden
    base_model = BaseModelClass(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 3. Backbone einfrieren
    base_model.trainable = False

    # 4. Deinen neuen Klassifikationskopf aufbauen
    inputs = layers.Input(shape=input_shape)

    # ResNet V1 Preprocessing
    x = preprocess_input(inputs)

    # Data Augmentation
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.2)(x)

    # Das gefrorene Basis-Modell anwenden (training=False für stabile BatchNormalization)
    x = base_model(x, training=False)

    # Dimensionen reduzieren
    x = layers.GlobalAveragePooling2D()(x)

    # Klassifikator mit Dropout gegen Overfitting
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=f"Transfer_ResNet{r_variant}_V1")
    return model, base_model