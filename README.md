# Fine-Grained Military Aircraft Classification with AeroConv10

Dieses Repository enthält den vollständigen Quellcode für ein iterativ entwickeltes Deep-Learning-Projekt zur feingranularen Klassifikation (*Fine-Grained Visual Classification*, FGVC) von militärischen Luftfahrzeugen. Das finale Modell **AeroConv10** wurde vollständig *from scratch* (ohne Transfer Learning) in TensorFlow/Keras implementiert und erreicht eine Test-Genauigkeit von **62,35 %** auf 28 visuell hochgradig ähnlichen Klassen.

## 🚀 Features & Architekturevolution
* **AeroConv1 bis AeroConv4:** Von der klassischen sequenziellen Baseline (VGG-Style) über die Behebung der Parameter-Explosion mittels *Global Average Pooling (GAP)* bis zur Stabilisierung durch *Batch Normalization* und *Dropout*.
* **AeroConv5 bis AeroConv9:** Integration asynchroner Daten-Augmentation und die systematische Untersuchung von Kapazitätsgrenzen und Rezeptivfeld-Skalierungen.
* **AeroConv10 (Ultimate):** Bruch mit dem sequenziellen Paradigma. Fusion aus **Residual Connections** (Skip-Connections für stabilen Gradientenfluss), **Squeeze-and-Excitation (SE) Blöcken** (dynamische Kanal-Aufmerksamkeit) und **asymmetrischen Faltungen** (Faktorisierung von $5 \times 5$-Filtern zur drastischen Parametereinsparung).

---

## 📊 Datensatz & Datenkuratierung

### Herkunft der Daten
Die Bilddaten basieren auf frei zugänglichen Repositorien für Flugzeug-Spotter (z. B. Kaggle / Web-Scraping-Datensätze für FGVC). Der ursprüngliche Rohdatensatz umfasste **102 verschiedene zivile und militärische Luftfahrzeugklassen**.

### Wichtiger Hinweis zur Datenverfügbarkeit
> ⚠️ **HINWEIS:** Aufgrund der großen Datenmenge (über 20.000 hochauflösende Bilder) und aus urheberrechtlichen Gründen sind die reinen Bilddaten **nicht** in diesem GitHub-Repository enthalten. Um das Projekt lokal auszuführen, muss die Ordnerstruktur unter `data/` manuell mit entsprechenden Bildern befüllt werden.

### Filterkriterien & Selektion
Um ein stabiles Training von Grund auf (*from scratch*) zu ermöglichen, wurde ein gezieltes Sub-Sampling durchgeführt. Es wurden ausschließlich militärische Klassen selektiert, die eine ausreichende Datendichte von **300 bis 800 Bildern pro Klasse** aufwiesen, um Overfitting in *Few-Shot*-Szenarien zu verhindern. Der final kuratierte Datensatz umfasst **20.113 Bilder**.

### Die 28 selektierten Klassen
Das Modell wurde auf die Erkennung der folgenden 28 spezifischen militärischen Jets, Bomber und Hubschrauber trainiert:

* **Erdkampfflugzeuge & Bomber:** A-10 Warthog, B-1 Lancer, B-2 Spirit, B-52 Stratofortress, Vulcan
* **Jagdflugzeuge & Mehrzweckkampfflugzeuge:** F-4 Phantom, F-14 Tomcat, F-15 Eagle, F-16 Fighting Falcon, F-22 Raptor, F-35 Lightning II, J-10, MiG-29 Fulcrum, MiG-31 Foxhound, Mirage 2000, Rafale, Su-24 Fencer, Su-34 Fullback, Su-57 Felon, Tornado
* **Transportflugzeuge:** C-130 Hercules, C-17 Globemaster
* **Hubschrauber:** AH-64 Apache, Mi-8, Mi-24 Hind, UH-60 Black Hawk
* **Unbemannte Drohnen (UAVs):** MQ-9 Reaper, TB2 Bayraktar

---

## 🛠️ Projektstruktur

```text
├── Benchmark_Results/      # Automatisch generierte F1-Reports und Confusion Matrices
├── Model/                  # Speicherort für die besten .keras Modellgewichte
├── Plots/                  # Generierte Abbildungen für die wissenschaftliche Arbeit
├── Src/
│   └── architectures.py    # Definition aller Modellstufen (AeroConv1 bis AeroConv10)
├── dataset_utility.py      # Skript zur Generierung der Bild-Kollagen
├── main.py                 # Hauptskript für das finale Training von AeroConv10
└── benchmark_runner.py     # Automatisierte Schleife zum Trainieren und Evaluieren von V1-V9
