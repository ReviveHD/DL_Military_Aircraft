import datetime
import os
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
from Src.architectures import (
    build_aeroconv1, build_aeroconv2, build_aeroconv3, build_aeroconv4,
    build_aeroconv5, build_aeroconv6, build_aeroconv7, build_aeroconv8,
    build_aeroconv9, build_aeroconv10
)

# --- 1. HYPERPARAMETER & STRUKTUR-SETUP ---
data_dir = "data"
batch_size = 16
img_height = 224  # Optimierte Dimension zur Gewährleistung der VRAM-Stabilität bei Batch Size 16
img_width = 224

# --- 2. PIPELINE STRATIFICATION: PHASE I (80% Training, 20% Temporärer Rest) ---
print("Lade Trainingsdaten (80%)...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=45,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("Lade restliche Daten (20%)...")
val_temp_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=45,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# --- 3. PIPELINE STRATIFICATION: PHASE II (Restliche 20% halbieren in 10% Val / 10% Test) ---
# Garantiert ein mathematisch sauberes und repräsentatives Drei-Wege-Splitting
val_batches = tf.data.experimental.cardinality(val_temp_ds)
test_ds = val_temp_ds.take(val_batches // 2)
val_ds = val_temp_ds.skip(val_batches // 2)

# --- COST-SENSITIVE LEARNING (Klassengewichtung) ---
print("Berechne Klassengewichte (Class Weights)...")
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
global_weight = dict(enumerate(class_weights))

class_names = train_ds.class_names
print(f"Gefundene Klassen: {class_names}")
print(f"Anzahl Trainings-Batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Anzahl Validierungs-Batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Anzahl Test-Batches: {tf.data.experimental.cardinality(test_ds)}")

# --- 4. PERFORMANCE OPTIMIERUNG VIA TENSORFLOW DATA API ---
AUTOTUNE = tf.data.AUTOTUNE
# Prefetching entkoppelt das Laden der Daten von der GPU-Verarbeitung (Verhinderung von Hardware-Bottlenecks)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- 5. MODELL-INSTANZIIERUNG (AeroConv10) ---
num_classes = len(class_names)
model = build_aeroconv10(num_classes=num_classes, input_shape=(img_height, img_width, 3))
model.summary()

# --- OPTIMIERUNGS-WÄCHTER (Callbacks für Kapitel 3.3) ---
# Verhindert Overfitting durch vorzeitigen Abbruch bei Stagnation des Validierungs-Verlusts
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

# Sichert den absolut besten Gewichts-Zustand zeitsynchronisiert auf die Festplatte
model_checkpoint = ModelCheckpoint(
    filepath='Model/best_model_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")+'.keras',
    monitor='val_loss',
    save_best_only=True
)

# Verringert die Schrittweite dynamisch, um präzise in enge Täler der Loss-Landschaft zu steuern
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

# Schreibt Telemetriedaten für die interaktive Weboberfläche in Ubuntu / WSL2
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

callbacks_list = [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]

# --- 6. MODELL-KOMPILIERUNG ---
# AdamW mit entkoppelter Gewichtungsdekadenz und starrem Gradient-Clipping gegen Gradienten-Explosion
model.compile(
    optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- 7. TRAININGS-PHASE ---
epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks_list,
    class_weight=global_weight,
)

# --- 8. ABSCHLUSSPRÜFUNG (Evaluierung auf isolierten Testdaten für Kapitel 4.3) ---
print("\n--- Training beendet. Starte Abschlussprüfung auf Testdaten ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test-Genauigkeit (Accuracy): {test_accuracy * 100:.2f}%")

# Plot-Funktion für die im Paper untereinander gesetzten Konvergenzkurven
def plot_training(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

print("Displaying training history...")
plot_training(history)

# --- 9. GENERIERUNG DER WAHRHEITSMATRIX (Für den LaTeX-Anhang) ---
print("\n--- Erstelle Wahrheitsmatrix (Confusion Matrix) ---")

y_true = []
y_pred = []

print("Sammle Vorhersagen für die Testdaten...")
# Ein einziger, sequentieller Durchlauf durch test_ds, um Index-Verschiebungen auszuschließen
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Matrix-Berechnung über scikit-learn
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

# Heatmap-Generierung via Seaborn
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Vorhergesagte Klasse (Das sagt die KI)', fontsize=12)
plt.ylabel('Wahre Klasse (Das ist es wirklich)', fontsize=12)
plt.title('Welche Flugzeuge verwechselt das Modell?', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Ausgabe des detaillierten Reports für die große F1-Score Tabelle im Anhang
print("\n--- Detaillierter Klassifikations-Report ---")
print(classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0))