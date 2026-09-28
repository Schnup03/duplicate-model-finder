# Duplicate Model Finder

Erweiterung für [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) und kompatible WebUI-Forks. Sie findet physisch verschiedene, inhaltlich identische Modelldateien und zeigt sie im Tab **Duplicate Models** an.

Unterstützt werden `.ckpt`, `.safetensors` und `.pt` in den Kategorien `Stable-diffusion`, `Lora` und `VAE`. Der Scan übernimmt den aktuellen Modellstamm der WebUI und zusätzlich konfigurierte Checkpoint-, LoRA- und VAE-Ordner. Außerhalb der WebUI verwendet er die relativen Standardordner `models/Stable-diffusion`, `models/Lora` und `models/VAE`.

## Installation

Im `extensions`-Ordner der WebUI installieren und die WebUI neu starten:

```bash
git clone https://github.com/Schnup03/duplicate-model-finder.git extensions/duplicate-model-finder
```

## Bedienung

Der Tab enthält folgende sichtbaren Aktionen:

| Schaltfläche | Wirkung |
| --- | --- |
| **Scan for duplicates** | Durchsucht die aktiven Modellordner und zeigt nur Gruppen mit mindestens zwei verschiedenen physischen Dateien an. Die Checkbox **Aggressive scan** wählt den alternativen BLAKE2b-/`scandir`-Scan. |
| **Select extra copies** | Wählt in jeder Gruppe die zusätzlichen, löschbaren Kopien aus und lässt mindestens eine Kopie stehen. Einzelne Dateien können auch von Hand gewählt werden. |
| **Cancel scan** | Bricht die laufende Suche kooperativ ab. Eine bereits begonnene Hash-Berechnung kann noch zu Ende laufen. |
| **Move selected to trash** | Verschiebt ausgewählte Dateien in einen wiederherstellbaren Schwesterordner. Die Trefferliste wird danach aktualisiert. |
| **Permanently delete selected** | Löscht nur nach Aktivierung der gesonderten Bestätigungs-Checkbox. Die letzte Kopie einer Treffergruppe bleibt geschützt. |
| **Empty trash** | Löscht nach gesonderter Bestätigung Papierkorbdateien, die **seit dem Aussortieren** mindestens 30 Tage alt sind. Jüngere Dateien bleiben erhalten. |

Dateien außerhalb der konfigurierten Modellordner können über Verzeichnisverknüpfungen sichtbar sein, sind aber in der Lösch-Auswahl gesperrt. Der tatsächliche Dateipfad wird unmittelbar vor einer Dateiaktion gegen die erlaubten Modellordner geprüft. Papierkorbdateien erscheinen nicht als neue Duplikate. Der Scanner erkennt auch zyklische Verzeichnisverknüpfungen.

Wer die Aufbewahrungsfrist für eine automatische Bereinigung beim Scan festlegen möchte, kann `TRASH_AUTO_EMPTY_DAYS` auf eine **positive ganze Zahl** setzen. Ohne diese Einstellung erfolgt keine automatische Bereinigung. Ungültige Werte werden übersprungen und im Status angezeigt. Dateien mit unbekannten, manuell vergebenen Papierkorbnamen bleiben bei einer positiven Frist aus Sicherheitsgründen erhalten; `purge_old_trash(0)` entfernt sie nur bei einem ausdrücklichen programmatischen Aufruf.

## Python-Schnittstelle

```python
from scripts.duplicate_model_finder import find_duplicates, move_files_to_trash

groups = find_duplicates()  # {"<sha256>": [".../a.ckpt", ".../copy.ckpt"]}
status, failed = move_files_to_trash(["models/Stable-diffusion/copy.ckpt"])
```

`find_duplicates` akzeptiert optional `directories`, `extensions`, `use_size_prefilter`, `max_workers`, `stop_event` und `prefer_nvme`. Standardmäßig überspringt die Größenfilterung Dateien, deren Größe einzigartig ist; gleiche Größen allein gelten **nicht** als Duplikat. Die alternative Scanoption verwendet BLAKE2b mit größeren Lese-Blöcken; ihre Hashwerte sind nicht mit den normalen SHA256-Werten austauschbar.

Die Funktionen `move_files_to_trash`, `permanently_delete_files` und `purge_old_trash` nehmen für eigenständige Skripte optional `allowed_roots=[...]` an. Ohne diese Angabe gelten die aktiven WebUI-Modellordner beziehungsweise außerhalb der WebUI die Standardordner. Die endgültige Löschung verlangt zusätzlich `confirm=True`. Dateiverknüpfungen, Ordner außerhalb der erlaubten Wurzeln und Papierkorbdateien werden von den normalen Löschfunktionen abgewiesen.

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

Die CI prüft Python 3.10–3.12 unter Linux, Python 3.12 unter Windows mit echten Junctions sowie die Bedienoberfläche mit Gradio 3.41.2. Der separate Secret-Scan prüft die Git-Historie.

## Grenzen

Diese Erweiterung sucht und bereinigt Duplikate. Eine NVMe-/Archiv-Verwaltung, Laufwerk-übergreifende Modellverschiebungen und Adapter für weitere Programme sind derzeit nicht enthalten. Über Verzeichnisverknüpfungen werden externe Modelle nur lesend in der Trefferliste gezeigt.

## Lizenz

MIT – siehe [LICENSE](LICENSE).
