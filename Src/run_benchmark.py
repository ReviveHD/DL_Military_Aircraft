import datetime
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import load_img

# Importierung des evolutionären Architektur-Lineups
from Src.architectures import (
    build_aeroconv1, build_aeroconv2, build_aeroconv3, build_aeroconv4,
    build_aeroconv5, build_aeroconv6, build_aeroconv7, build_aeroconv8,
    build_aeroconv9, build_aeroconv10
)

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
data_dir = "/mnt/c/Users/Kaspar/PycharmProjects/DL_Military_Aircraft/data"
batch_size = 16
img_height = 224
img_width = 224

# Zentrales Verzeichnis für die automatisiert generierten Messergebnisse
results_dir = "Benchmark_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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

    # OOM-freundliche Parsing-Funktion
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

        # Parallelisiertes Laden der Bilder (begrenzt gegen RAM-Überlastung)
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

num_classes = len(class_names)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(raw_train_labels),
    y=raw_train_labels
)
global_weight = dict(enumerate(class_weights))


# --- 3. AUTOMATED BENCHMARK PIPELINE ---
# Iterationsliste für den automatisierten Modell-Vergleich
architecture_functions = [
    build_aeroconv1, build_aeroconv2, build_aeroconv3, build_aeroconv4,
    build_aeroconv5, build_aeroconv6, build_aeroconv7, build_aeroconv8,
    build_aeroconv9, build_aeroconv10
]

benchmark_summary = []

for build_fn in architecture_functions:
    model_name = build_fn.__name__.replace("build_", "")
    print(f"\n" + "=" * 50)
    print(f"🚀 STARTE TRAINING FÜR: {model_name.upper()}")
    print("=" * 50 + "\n")

    # Modellinstanziierung basierend auf den vordefinierten Bildmaßen
    model = build_fn(num_classes=num_classes, input_shape=(img_height, img_width, 3))

    # Konsistente Kompilierungsparameter mit Gradient-Clipping zur numerischen Stabilität
    model.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Lokale Callbacks zur dynamischen Trainingssteuerung
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0)

    # Persistenz-Sicherung: Speichert nur den Zustand mit dem minimalen Validierungsverlust
    model_path = os.path.join(results_dir, f"{model_name}_best.keras")
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=0)

    # Modelltraining unter Berücksichtigung der berechneten Klassengewichte
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=global_weight,
        verbose=1
    )

    # Evaluierung auf dem gänzlich ungesehenen Test-Split
    print(f"\nEvaluierung von {model_name} auf Testdaten...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])

    # Protokollierung der Metriken für die finale Benchmark-Tabelle
    benchmark_summary.append({
        'Model': model_name.upper(),
        'Train_Acc': round(best_train_acc * 100, 2),
        'Val_Acc': round(best_val_acc * 100, 2),
        'Test_Acc': round(test_accuracy * 100, 2)
    })

    # --- 4. POST-MORTEM EVALUATION & METRIC GENERATION ---
    y_true = []
    y_pred = []

    # Iterative Batch-Vorhersage auf den Testdaten zur Generierung der Vorhersage-Vektoren
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4.1 Export des klassenspezifischen Metrik-Reports (F1, Precision, Recall) als TXT-Datei
    report_str = classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names,
                                       zero_division=0)
    report_path = os.path.join(results_dir, f"{model_name}_F1_Report.txt")
    with open(report_path, "w") as text_file:
        text_file.write(f"Classification Report für {model_name.upper()}\n")
        text_file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        text_file.write("=" * 50 + "\n")
        text_file.write(report_str)

    # 4.2 Visualisierung und Export der Wahrheitsmatrix (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Wahre Klasse')
    plt.title(f'Confusion Matrix: {model_name.upper()} (Test Acc: {test_accuracy * 100:.2f}%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Abspeichern im Ergebnisordner
    cm_path = os.path.join(results_dir, f"{model_name}_ConfusionMatrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()  # Speicherbereinigung: Schließt die Grafik, um RAM-Anstauungen im Loop zu verhindern


# --- 5. DATA AGGREGATION & EXPORT ---
print("\n" + "=" * 50)
print("ALLE MODELLE TRAINIERT! SPEICHERE ZUSAMMENFASSUNG...")
print("=" * 50)

# Transformation der Ergebnisliste in ein Pandas-DataFrame für den CSV-Export
df_results = pd.DataFrame(benchmark_summary)
csv_path = os.path.join(results_dir, "Benchmark_Zusammenfassung.csv")
df_results.to_csv(csv_path, index=False, sep=";")

# Konsolen-Ausgabe der finalen Performance-Matrix (wichtig für LaTeX Tabelle \ref{tab:results_evolution})
print(df_results.to_string(index=False))
print(f"\nAlle Detail-Reports (F1-Scores) und Bilder liegen im Ordner: '{results_dir}'")
