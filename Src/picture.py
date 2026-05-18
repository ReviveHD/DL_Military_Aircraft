import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img

# 1. Pfaddefinition zum Datenverzeichnis
data_dir = "/mnt/c/Users/Kaspar/PycharmProjects/DL_Military_Aircraft/data"

# 2. Zielklassen für die visuelle Repräsentation im Paper (Kapitel 2.2)
classes_to_show = ['F16', 'Mig29', 'Rafale', 'Su57']

# 3. Konfiguration der Plot-Fläche (2x2 Grid)
plt.figure(figsize=(10, 8))

for i, class_name in enumerate(classes_to_show):
    class_dir = os.path.join(data_dir, class_name)

    # Dateisystem abfragen und nur gültige Bildformate filtern
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Stochastische Auswahl für eine unvoreingenommene Stichprobe
    random_image = random.choice(images)
    img_path = os.path.join(class_dir, random_image)

    # Bild via Keras-Utility in den Speicher laden
    img = load_img(img_path)

    # Positionierung im 2x2-Subplot-Raster
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)

    # Akademische Formatierung der Titelzeilen
    plt.title(f"Class: {class_name}", fontsize=16, fontweight='bold')
    plt.axis('off')  # Ausblenden der Pixel-Achsen für ein sauberes LaTeX-Layout

# 4. Optimierung der Abstände und hochauflösender Export
plt.tight_layout()

# dpi=300 garantiert Druckqualität; bbox_inches='tight' verhindert abgeschnittene Ränder
save_path = "../dataset_samples.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Erfolg! Die Kollage wurde als '{save_path}' gespeichert.")