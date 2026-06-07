# Fine-Grained Military Aircraft Classification with AeroConv10

Dieses Repository enthaelt den vollstaendigen Quellcode fuer ein iterativ entwickeltes Deep-Learning-Projekt zur feingranularen Klassifikation (Fine-Grained Visual Classification, FGVC) von militaerischen Luftfahrzeugen. Das finale Modell AeroConv10 wurde vollstaendig von Grund auf (without Transfer Learning) in TensorFlow/Keras implementiert und erreicht eine Test-Genauigkeit von 62,35 % auf 28 visuell hochgradig aehnlichen Klassen.

## Systemumgebung und Reproduzierbarkeit
Das gesamte Projekt wurde in einer hybriden Umgebung entwickelt und evaluiert. Um Inkompatibilitaeten zu vermeiden und die volle native GPU-Beschleunigung zu nutzen, wurden das Training und die Benchmarks unter Windows Subsystem for Linux (WSL2 / Ubuntu) innerhalb einer isolierten Python-Umgebung (virtualenv, Python 3.12) ausgefuehrt. 

Hinweis zur Hardware-Stabilitaet: Zur Vermeidung von Speicher-Abstuerzen (VRAM-Crashes) unter WSL2 wurde die Bildgroesse auf 224x224 Pixel optimiert und auf den Einsatz von Mixed Precision verzichtet.

## Architekturevolution
* AeroConv1 bis AeroConv4: Von der klassischen sequenziellen Baseline (VGG-Style) ueber die Behebung der Parameter-Explosion mittels Global Average Pooling (GAP) bis zur Stabilisierung durch Batch Normalization und Dropout.
* AeroConv5 bis AeroConv9: Integration asynchroner Daten-Augmentation und die systematische Untersuchung von Kapazitaetsgrenzen und Rezeptivfeld-Skalierungen.
* AeroConv10 (Ultimate): Bruch mit dem sequenziellen Paradigma. Fusion aus Residual Connections (Skip-Connections fuer stabilen Gradientenfluss), Squeeze-and-Excitation (SE) Bloecken (dynamische Kanal-Aufmerksamkeit) und asymmetrischen Faltungen (Faktorisierung von 5x5-Filtern zur drastischen Parametereinsparung).

---

## Datensatz und Datenkuratierung

### Herkunft der Daten
Die Bilddaten basieren auf einem oeffentlich zugaenglichen FGVC-Datensatz.
Link zum originalen Rohdatensatz: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

Der urspruengliche Rohdatensatz umfasste insgesamt 102 verschiedene zivile und militaerische Luftfahrzeugklassen.

### Verfuegbarkeit im Repository
Aufgrund der grossen Datenmenge (ueber 20.000 hochaufloesende Bilder) und aus urheberrechtlichen Gruenden sind die reinen Bilddaten nicht in diesem GitHub-Repository enthalten. Das Gleiche gilt fuer die grossen, trainierten Modellgewichte (.keras / .h5), um die Dateigroessen-Limits von GitHub zu respektieren. Um das Projekt lokal auszufuehren, muss die Ordnerstruktur unter data/ manuell mit den entsprechenden Klassenordnern aus der oben genannten Quelle befuellt werden.

### Filterkriterien und Selektion
Um ein stabiles Training von Grund auf (from scratch) zu ermoeglichen, wurde ein gezieltes Sub-Sampling durchgefuehrt. Es wurden ausschliesslich militaerische Klassen selektiert, die eine ausreichende Datendichte von mindestens 300 Bildern pro Klasse aufwiesen, um Overfitting in Few-Shot-Szenarien zu verhindern. Der final kuratierte Datensatz umfasst 20.113 Bilder.

### Die 28 selektierten Klassen
Das Modell wurde auf die Erkennung der folgenden 28 spezifischen militaerischen Jets, Bomber und Hubschrauber trainiert:

* Erdkampfflugzeuge und Bomber: A-10 Warthog, B-1 Lancer, B-2 Spirit, B-52 Stratofortress, Vulcan
* Jagdflugzeuge und Mehrzweckkampfflugzeuge: F-4 Phantom, F-14 Tomcat, F-15 Eagle, F-16 Fighting Falcon, F-22 Raptor, F-35 Lightning II, J-10, MiG-29 Fulcrum, MiG-31 Foxhound, Mirage 2000, Rafale, Su-24 Fencer, Su-34 Fullback, Su-57 Felon, Tornado
* Transportflugzeuge: C-130 Hercules, C-17 Globemaster
* Hubschrauber: AH-64 Apache, Mi-8, Mi-24 Hind, UH-60 Black Hawk
* Unbemannte Drohnen (UAVs): MQ-9 Reaper, TB2 Bayraktar

---

## Projektstruktur

```text
├── Benchmark_Results/      # Automatisch generierte F1-Reports, Graphen und Confusion Matrices
├── Plots/                  # Generierte Abbildungen und Architektur-Visualisierungen fuers Paper
├── Src/
│   ├── architectures.py    # Definition aller Modellstufen (AeroConv1 bis AeroConv10)
│   ├── main.py             # Hauptskript fuer das finale Training von AeroConv10
│   ├── picture.py          # Skript zur Visualisierung / Bilderzeugung
│   └── run_benchmark.py    # Automatisierte Schleife zum Trainieren und Evaluieren von V1-V9
├── .gitignore              # Ignoriert Trainingsdaten und grosse Modellgewichte (.keras/.h5)
├── README.md               # Diese Projektdokumentation
└── dataset_samples.png     # Beispielhafte Zusammenstellung der Trainingsklassen
