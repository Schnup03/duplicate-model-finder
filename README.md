# Duplicate Model Finder

Erweiterung für die [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), die Modellverzeichnisse nach doppelten Dateien durchsucht und ein bequemes Löschen direkt aus der WebUI ermöglicht.

## Überblick

- Durchsucht die üblichen Modellordner (`models/Stable-diffusion`, `models/Lora`, `models/VAE`).
- Unterstützte Dateiendungen: `.ckpt`, `.safetensors`, `.pt`.
- Berechnet SHA256-Hashes, um identische Dateien zuverlässig zu erkennen.
- Stellt einen eigenen Tab in der WebUI bereit mit Schaltflächen zum Scannen und Löschen.

## Voraussetzungen

- Laufende Installation der [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Python-Umgebung der WebUI (Gradio ist Bestandteil der WebUI-Abhängigkeiten).

## Installation

Klonen Sie das Repository in den `extensions`-Ordner Ihrer WebUI-Installation:

```bash
git clone <repo_url> extensions/duplicate-model-finder
```

## Verwendung

1. Starten Sie die WebUI wie gewohnt.
2. Öffnen Sie den Tab **Duplicate Models**.
3. Klicken Sie auf **Scan for duplicates**, um doppelte Modelle zu ermitteln.
4. Wählen Sie über die Checkboxen die überflüssigen Dateien aus und klicken Sie auf **Delete selected**.

Nach einem Scan werden alle gefundenen Doppelgänger mit ihrem Hash angezeigt. Dateien, die sich nicht löschen ließen, bleiben ausgewählt, sodass ein erneuter Versuch möglich ist.

## Tests

Dieses Projekt verwendet `pytest` für Unit-Tests. Um die Tests auszuführen, aktivieren Sie die Python-Umgebung der WebUI und führen Sie dann aus:

```bash
pytest
```

## Bekannte Hinweise

- Nur Dateien mit den Endungen `.ckpt`, `.safetensors` und `.pt` werden berücksichtigt.
- Die Hash-Berechnung erfolgt in 1-MB-Blöcken, um den Speicherverbrauch niedrig zu halten.

## Performance-Hinweise

- **Size-Pre-Filter:** Vor dem SHA256-Hashing werden Dateien nach Größe gruppiert. Nur Dateien, die ihre Größe mit mindestens einer anderen Datei teilen, werden gehasht. Das spart bei großen Modell-Dateien (typischerweise mehrere GB pro `.safetensors`) erheblich I/O, weil eindeutige Dateien gar nicht erst gelesen werden.
- **Paralleles Hashing:** Das Hashing läuft per Default in einem Thread-Pool mit bis zu 8 Workern, abhängig von der verfügbaren CPU-Anzahl. Beide Optimierungen lassen sich getrennt deaktivieren — `find_duplicates(use_size_prefilter=False, max_workers=1)` für rein sequentielles Verhalten (nützlich für deterministische Tests).
- **Skalierung:** Bei 50 Modell-Dateien à 4 GB reduziert der Size-Pre-Filter die zu hashende Datenmenge typischerweise um >95 %; die Parallelisierung skaliert mit der Anzahl physischer Kerne.
