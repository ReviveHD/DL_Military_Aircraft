import datetime
import os
import random
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW

# HINWEIS: Mixed Precision bewusst deaktiviert zur Absicherung der WSL2-VRAM-Stabilität.
from Src.architectures import build_aeroconv10

# --- 1. HYPERPARAMETER & STRUKTUR-SETUP ---
data_dir = "data"
batch_size = 16
img_height = 224
img_width = 224

# --- 2. PIPELINE STRATIFICATION: DETERMINISTISCHER SPLIT ---
print("\n--- Sammle und splitte Bilder deterministisch (Seed: 45) ---")


def create_deterministic_datasets(data_dir, img_height, img_width, batch_size, seed=45):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    # Alle Bildpfade sammeln
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(class_to_idx[class_name])

    # Deterministisch mischen
    combined = list(zip(all_paths, all_labels))
    random.seed(seed)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    # 80% Train, 10% Val, 10% Test
    total = len(all_paths)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_paths, train_labels = all_paths[:train_end], all_labels[:train_end]
    val_paths, val_labels = all_paths[train_end:val_end], all_labels[train_end:val_end]
    test_paths, test_labels = all_paths[val_end:], all_labels[val_end:]

    def parse_image(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32)
        return image, label

    def build_tf_dataset(paths, labels, is_training):
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        if is_training:
            ds = ds.shuffle(buffer_size=2000, seed=seed)

        ds = ds.map(parse_image, num_parallel_calls=4)

        ds = ds.batch(batch_size)
        return ds.prefetch(buffer_size=2)

    train_ds = build_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, is_training=False)
    test_ds = build_tf_dataset(test_paths, test_labels, is_training=False)

    print(f"Erfolgreich aufgeteilt: {len(train_paths)} Train | {len(val_paths)} Val | {len(test_paths)} Test")
    return train_ds, val_ds, test_ds, class_names, train_labels


train_ds, val_ds, test_ds, class_names, raw_train_labels = create_deterministic_datasets(
    data_dir, img_height, img_width, batch_size, seed=45
)

# --- 3. COST-SENSITIVE LEARNING (Klassengewichtung) ---
print("\n--- Berechne Klassengewichte (Class Weights) ---")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(raw_train_labels),
    y=raw_train_labels
)
global_weight = dict(enumerate(class_weights))
print("Gewichte erfolgreich berechnet!")

num_classes = len(class_names)
print(f"Gefundene Klassen: {num_classes}")

# --- 4. MODELL-INSTANZIIERUNG (AeroConv10) ---
print("\n--- Initialisiere AeroConv10 (From Scratch) ---")
model = build_aeroconv10(num_classes=num_classes, input_shape=(img_height, img_width, 3))
model.summary()

# --- 5. OPTIMIERUNGS-WÄCHTER (Callbacks) ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='Model/best_model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.keras',
    monitor='val_loss',
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

callbacks_list = [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]

# --- 6. MODELL-KOMPILIERUNG ---
model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --- 7. TRAININGS-PHASE ---
print("\n" + "=" * 50)
print("STARTE TRAINING VON GRUND AUF (Mit Augmentation)")
print("=" * 50)

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks_list,
    class_weight=global_weight,
)

# --- 8. ABSCHLUSSPRÜFUNG (Evaluierung auf Testdaten) ---
print("\n--- Training beendet. Starte Abschlussprüfung auf Testdaten ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest-Genauigkeit (Accuracy): {test_accuracy * 100:.2f}%\n")


# --- 9. PLOT TRAINING HISTORY ---
def plot_training(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


print("Displaying training history...")
plot_training(history)

# --- 10. GENERIERUNG DER WAHRHEITSMATRIX ---
print("\n--- Erstelle Wahrheitsmatrix (Confusion Matrix) ---")
y_true = []
y_pred = []

print("Sammle Vorhersagen für die Testdaten...")
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Vorhergesagte Klasse (Das sagt die KI)', fontsize=12)
plt.ylabel('Wahre Klasse (Das ist es wirklich)', fontsize=12)
plt.title(f'Wahrheitsmatrix - Test Accuracy: {test_accuracy * 100:.1f}%', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Detaillierter Klassifikations-Report ---")
print(classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names,
                            zero_division=0))