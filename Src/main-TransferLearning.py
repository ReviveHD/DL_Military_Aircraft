import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf
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
# --- 2. PIPELINE STRATIFICATION: PHASE I (80% Training, 20% Temporärer Rest) ---
# ==============================================================================
print("\n--- Lade Datensatz ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=45,
    image_size=(img_height, img_width), batch_size=batch_size
)

val_temp_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=45,
    image_size=(img_height, img_width), batch_size=batch_size
)

# ==============================================================================
# --- 3. PIPELINE STRATIFICATION: PHASE II (10% Val / 10% Test) ---
# ==============================================================================
val_batches = tf.data.experimental.cardinality(val_temp_ds)
test_ds = val_temp_ds.take(val_batches // 2)
val_ds = val_temp_ds.skip(val_batches // 2)

# --- COST-SENSITIVE LEARNING (Klassengewichtung) ---
print("Berechne Klassengewichte (Class Weights)...")
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
global_weight = dict(enumerate(class_weights))
class_names = train_ds.class_names
num_classes = len(class_names)

# ==============================================================================
# --- 4. PERFORMANCE OPTIMIERUNG ---
# ==============================================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# --- 5. MODELL-INSTANZIIERUNG (ResNet101) ---
# ==============================================================================
model, base_model = build_dynamic_resnet(input_shape=(img_height, img_width, 3), num_classes=num_classes, r_variant='101')

# --- OPTIMIERUNGS-WÄCHTER ---
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='Model/best_resnet_model_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")+'.keras',
    monitor='val_loss', save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
log_dir = "logs/fit/resnet_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks_phase2 = [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]

# ==============================================================================
# --- 6. PHASE 1: FEATURE EXTRACTION (Backbone eingefroren) ---
# ==============================================================================
print("\n" + "="*50)
print("STARTE PHASE 1: Feature Extraction (Backbone eingefroren)")
print("="*50)

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
# --- 7. PHASE 2: PARTIAL FINE-TUNING (Blöcke conv4 und conv5 aufgetaut) ---
# ==============================================================================
print("\n" + "="*50)
print("STARTE PHASE 2: Partial Fine-Tuning (Blöcke conv4 und conv5 aufgetaut)")
print("="*50)

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

epochs_phase2 = 50
history_phase2 = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs_phase2,
    initial_epoch=history_phase1.epoch[-1] + 1,
    callbacks=callbacks_phase2, class_weight=global_weight,
)

# ==============================================================================
# --- 8. ABSCHLUSSPRÜFUNG & PLOTS ---
# ==============================================================================
print("\n--- Training beendet. Starte Abschlussprüfung auf Testdaten ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test-Genauigkeit (Accuracy): {test_accuracy * 100:.2f}%")

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
# --- 9. GENERIERUNG DER WAHRHEITSMATRIX ---
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
plt.title('Confusion Matrix - ResNet101 Transfer Learning', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Detaillierter Klassifikations-Report ---")
print(classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0))