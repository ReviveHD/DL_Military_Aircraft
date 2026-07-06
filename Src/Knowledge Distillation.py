import os

from sklearn.utils import compute_class_weight

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime
import random
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from Src.Distiller import Distiller

from Src.student import build_aeroconv_residual

import numpy as np
import matplotlib.pyplot as plt
import os


def extract_misclassified_images(model, test_ds, class_names, output_dir="misclassified_samples"):
    os.makedirs(output_dir, exist_ok=True)

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = y_batch.numpy()

        # Finde Indices, wo Vorhersage falsch ist
        mask = (pred_classes != true_classes)
        errors = np.where(mask)[0]

        for idx in errors:
            img = x_batch[idx].numpy().astype("uint8")
            true_label = class_names[true_classes[idx]]
            pred_label = class_names[pred_classes[idx]]

            # Speichere Bild
            plt.imshow(img)
            plt.title(f"True: {true_label} | Pred: {pred_label}")
            plt.axis('off')
            plt.savefig(f"{output_dir}/{true_label}_as_{pred_label}_{idx}.png")
            plt.close()

    print(f"Fehlerhafte Bilder wurden in '{output_dir}' gespeichert.")

# ==============================================================================
# --- 1. PIPELINE & DATENLADEN (DETERMINISTISCHER SPLIT) ---
# ==============================================================================
if __name__ == "__main__":
    data_dir = "data"
    batch_size = 16
    img_height = 224
    img_width = 224

    print("\n--- Sammle und splitte Bilder deterministisch (Seed: 45) ---")

    def create_deterministic_datasets(data_dir, img_height, img_width, batch_size, seed=45):
        # 1. Klassen auslesen und alphabetisch sortieren
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        all_paths = []
        all_labels = []

        # 2. Alle Bildpfade sammeln
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_paths.append(os.path.join(class_dir, img_name))
                    all_labels.append(class_to_idx[class_name])

        # 3. Zwingend deterministisch mischen (Python random ist hier sehr stabil)
        combined = list(zip(all_paths, all_labels))
        random.seed(seed)
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)

        # 4. Gnadenlos in 80% Train, 10% Val, 10% Test zerschneiden
        total = len(all_paths)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        train_paths, train_labels = all_paths[:train_end], all_labels[:train_end]
        val_paths, val_labels = all_paths[train_end:val_end], all_labels[train_end:val_end]
        test_paths, test_labels = all_paths[val_end:], all_labels[val_end:]

        # 5. OOM-freundliche Parsing-Funktion
        def parse_image(filename, label):
            image_string = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.resize(image, [img_height, img_width])
            image = tf.cast(image, tf.float32)
            return image, label

        # 6. Ressourcen-schonende Dataset-Pipeline bauen
        def build_tf_dataset(paths, labels, is_training):
            ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

            if is_training:
                ds = ds.shuffle(buffer_size=2000, seed=seed)

            ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)


            ds = ds.batch(batch_size)


            return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        train_ds = build_tf_dataset(train_paths, train_labels, is_training=True)
        val_ds = build_tf_dataset(val_paths, val_labels, is_training=False)
        test_ds = build_tf_dataset(test_paths, test_labels, is_training=False)

        print(f"Erfolgreich aufgeteilt: {len(train_paths)} Train | {len(val_paths)} Val | {len(test_paths)} Test")
        return train_ds, val_ds, test_ds, class_names


    # Wir rufen unsere neue, kugelsichere Funktion auf!
    train_ds, val_ds, test_ds, class_names = create_deterministic_datasets(
        data_dir, img_height, img_width, batch_size, seed=45
    )
    num_classes = len(class_names)

    # ==============================================================================
    # --- 2. TEACHER & STUDENT INITIALISIEREN ---
    # ==============================================================================
    print("\n--- Lade Teacher Modell (ResNet101) ---")


    def load_teacher_with_logits(model_path):
        base_teacher = tf.keras.models.load_model(model_path)
        base_teacher.layers[-1].activation = keras.activations.linear
        base_teacher.trainable = False
        return base_teacher

    # Teacher laden
    teacher_model = tf.keras.models.load_model('Model/best_resnet_model_20260623-1915.keras')
    teacher_model.layers[-1].activation = keras.activations.linear
    teacher_model.trainable = False

    # Student erstellen
    student_model = build_aeroconv_residual(input_shape=(img_height, img_width, 3), num_classes=num_classes)
    student_model.layers[-1].activation = keras.activations.linear

    # Füge das vor der Distiller-Initialisierung ein:
    all_labels_flat = []
    for _, labels in train_ds.unbatch():
        all_labels_flat.append(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels_flat), y=all_labels_flat)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Initialisiere Distiller
    distiller = Distiller(teacher=teacher_model, student=student_model, class_weight_dict=class_weight_dict)
    distiller.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.9,
    temperature=3.0
    )

    # ==============================================================================
    # --- 3. DISTILLATION TRAINING STARTEN ---
    # ==============================================================================
    print("\n" + "=" * 50)
    print("STARTE KNOWLEDGE DISTILLATION")
    print("=" * 50)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        callbacks=[reduce_lr, early_stopping]
    )

    # ==============================================================================
    # --- 4. FINALES SCHÜLER-MODELL SPEICHERN & EVALUIEREN ---
    # ==============================================================================
    distiller.student.save('Model/best_model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.keras')
    print("\n--- Distilled Student Model  erfolgreich gespeichert! ---")

    student_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("\n--- Finale Evaluation des destillierten Schülers auf Testdaten ---")
    test_loss, test_accuracy = student_model.evaluate(test_ds)
    print(f"\nStudent Test-Genauigkeit (Accuracy): {test_accuracy * 100:.2f}%")

    print("\n--- Erstelle Wahrheitsmatrix (Confusion Matrix) ---")
    y_true = []
    y_pred = []

    print("Sammle Vorhersagen für die Testdaten...")
    for x_batch, y_batch in test_ds:
        preds = student_model.predict(x_batch, verbose=0)
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

    extract_misclassified_images(student_model, test_ds, class_names)




