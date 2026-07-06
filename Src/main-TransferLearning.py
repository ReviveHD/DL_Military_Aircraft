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

# Import deiner neuen ResNet Architektur
from Src.ResNet import build_dynamic_resnet

# ==============================================================================
# --- 1. HYPERPARAMETER & STRUKTUR-SETUP ---
# ==============================================================================
data_dir = "data"
batch_size = 16
img_height = 224
img_width = 224

# ==============================================================================
# --- 2. PIPELINE STRATIFICATION: DETERMINISTISCHER SPLIT ---
# ==============================================================================
print("\n--- Sammle und splitte Bilder deterministisch (Seed: 45) ---")


def create_deterministic_datasets(data_dir, img_height, img_width, batch_size, seed=45):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(class_to_idx[class_name])

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


# --- COST-SENSITIVE LEARNING (Klassengewichtung) ---
print("Berechne Klassengewichte (Class Weights)...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(raw_train_labels),
    y=raw_train_labels
)
global_weight = dict(enumerate(class_weights))
num_classes = len(class_names)
print("Gewichte erfolgreich berechnet!")

# ==============================================================================
# --- 3. MODELL-INSTANZIIERUNG (ResNet101) ---
# ==============================================================================
model, base_model = build_dynamic_resnet(input_shape=(img_height, img_width, 3), num_classes=num_classes,
                                         r_variant='101')

# --- OPTIMIERUNGS-WÄCHTER ---
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='Model/best_resnet_model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.keras',
    monitor='val_loss', save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
log_dir = "logs/fit/resnet_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks_phase2 = [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]

# ==============================================================================
# --- 4. PHASE 1: FEATURE EXTRACTION (Backbone eingefroren) ---
# ==============================================================================
print("\n" + "=" * 50)
print("STARTE PHASE 1: Feature Extraction (Backbone eingefroren)")
print("=" * 50)

model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

epochs_phase1 = 10
history_phase1 = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs_phase1,
    callbacks=[early_stopping, tensorboard_callback], class_weight=global_weight,
)

# ==============================================================================
# --- 5. PHASE 2: PARTIAL FINE-TUNING (Blöcke conv4 und conv5 aufgetaut) ---
# ==============================================================================
print("\n" + "=" * 50)
print("STARTE PHASE 2: Partial Fine-Tuning (Blöcke conv4 und conv5 aufgetaut)")
print("=" * 50)

# Backbone öffnen
base_model.trainable = True

# 1. Definiere, ab welchem conv4-Block du auftauen willst.
# ResNet101 hat in conv4 die Blöcke 1 bis 23. (1 = alle conv4 Blöcke auftauen, 23 = nur einen conv4 Block auftauen)
start_block = 1
target_blocks = [f"conv4_block{i}" for i in range(start_block, 24)]

# 2. Auftau-Logik
for layer in base_model.layers:
    # Wenn die Schicht zu conv5 gehört ODER zu einem unserer ausgewählten conv4-Blöcke
    if layer.name.startswith('conv5') or any(layer.name.startswith(b) for b in target_blocks):
        layer.trainable = True
    else:
        # Alles darunter (conv1, conv2, conv3 und conv4_block1 bis start_block) bleibt eingefroren!
        layer.trainable = False

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Es wurden exakt {trainable_count} Schichten für das Fine-Tuning aufgetaut.")

# Neu kompilieren mit winziger Lernrate (1e-5)
model.compile(
    optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

epochs_phase2 = 40
history_phase2 = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs_phase2,
    initial_epoch=history_phase1.epoch[-1] + 1,
    callbacks=callbacks_phase2, class_weight=global_weight,
)

# ==============================================================================
# --- 6. ABSCHLUSSPRÜFUNG & PLOTS ---
# ==============================================================================
print("\n--- Training beendet. Starte Abschlussprüfung auf Testdaten ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest-Genauigkeit (Accuracy): {test_accuracy * 100:.2f}%\n")


def plot_training(hist1, hist2):
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    epochs = range(1, len(acc) + 1)
    phase1_end = len(hist1.history['accuracy'])

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.axvline(x=phase1_end, color='k', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.axvline(x=phase1_end, color='k', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


print("Displaying training history...")
plot_training(history_phase1, history_phase2)

# ==============================================================================
# --- 7. GENERIERUNG DER WAHRHEITSMATRIX ---
# ==============================================================================
print("\n--- Erstelle Wahrheitsmatrix (Confusion Matrix) ---")
y_true, y_pred = [], []
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Vorhergesagte Klasse', fontsize=12)
plt.ylabel('Wahre Klasse', fontsize=12)
plt.title(f'Confusion Matrix - ResNet101 Transfer Learning (Test Acc: {test_accuracy * 100:.1f}%)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Detaillierter Klassifikations-Report ---")
print(classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names,
                            zero_division=0))