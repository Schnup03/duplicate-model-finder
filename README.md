# Duplicate Model Finder

Erweiterung für die [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), die Modellverzeichnisse nach doppelten Dateien durchsucht und ein bequemes Löschen direkt aus der WebUI ermöglicht.

## Überblick

- Durchsucht die üblichen Modellordner (`models/Stable-diffusion`, `models/Lora`, `models/VAE`).
- Unterstützte Dateiendungen: `.ckpt`, `.safetensors`, `.pt`.
- Berechnet SHA256-Hashes, um identische Dateien zuverlässig zu erkennen.
- Stellt einen eigenen Tab in der WebUI bereit mit Schaltflächen zum Scannen und soft-/hart-Löschen.
- Verfolgt Symlinks und meldet jede physische Datei nur einmal.
- Bietet einen **Cancel scan**-Button zum kooperativen Abbrechen laufender Scans.

## Voraussetzungen

- Laufende Installation der [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Python-Umgebung der WebUI (Gradio ist Bestandteil der WebUI-Abhängigkeiten).

## Installation

Klonen Sie das Repository in den `extensions`-Ordner Ihrer WebUI-Installation:

```bash
git clone https://github.com/Schnup03/duplicate-model-finder.git extensions/duplicate-model-finder
```

Das Repository ist **öffentlich** — kein GitHub-Login erforderlich.

## Verwendung

1. Starten Sie die WebUI wie gewohnt.
2. Öffnen Sie den Tab **Duplicate Models**.
3. Klicken Sie auf **Scan for duplicates**, um doppelte Modelle zu ermitteln.
4. Während eines Scans können Sie den Vorgang jederzeit mit **Cancel scan** abbrechen — der Scan stoppt kooperativ beim nächsten Verzeichniswechsel.
5. Wählen Sie über die Checkboxen die überflüssigen Dateien aus.
6. Klicken Sie auf **Move selected to trash** (Standardweg, reversibel) **oder** auf **Permanently delete selected** (setzt zusätzlich die Bestätigungs-Checkbox voraus).

Nach einem Scan werden alle gefundenen Doppelgänger mit ihrem Hash angezeigt. Dateien, die sich nicht löschen ließen, bleiben ausgewählt, sodass ein erneuter Versuch möglich ist.

## Konfiguration (Public API)

Alle Funktionen sind auch programmatisch nutzbar, z. B. aus eigenen Skripten oder Tests:

```python
from scripts.duplicate_model_finder import (
    find_duplicates,
    iter_model_files,
    move_files_to_trash,
    permanently_delete_files,
)
import threading

# Standard-Scan
dups = find_duplicates()
# {"<sha256>": ["/path/a.ckpt", "/path/a_copy.ckpt"], ...}

# Scan mit Anpassungen
dups = find_duplicates(
    directories=["models/Stable-diffusion"],
    extensions=(".safetensors",),
    use_size_prefilter=False,    # jede Datei hashen (kein Size-Skip)
    max_workers=1,                # sequentiell (deterministisch für Tests)
)

# Modell-Dateien direkt iterieren — mit optionalem Abbruch
stop = threading.Event()
for path in iter_model_files(stop_event=stop):
    print(path)
    if found_enough:
        stop.set()  # bricht den Walk kooperativ beim nächsten Stop-Check ab

# Sicheres Löschen
status, failed = move_files_to_trash(["models/Stable-diffusion/a.ckpt"])
# "Moved 1 file(s) to .duplicate_model_finder_trash/ (recoverable manually)"

# Irreversibles Löschen — verlangt explizites confirm=True
status, failed = permanently_delete_files(
    ["/tmp/some/leftover.ckpt"], confirm=True,
)
```

| Parameter | Default | Bedeutung |
|-----------|---------|-----------|
| `use_size_prefilter` | `True` | Dateien mit einzigartiger Größe werden nicht gehasht (großer Performance-Win bei vielen einzigartigen Modellen). |
| `max_workers` | `min(8, max(2, cpu_count))` | Größe des Thread-Pools für paralleles Hashing. `1` für sequentiell. |
| `stop_event` | `None` | Optionaler `threading.Event` zum kooperativen Abbruch während des Walks und der Größen-Gruppierung. |
| `directories` | `MODEL_DIRS` | Liste der zu scannenden Verzeichnisse. |
| `extensions` | `(".ckpt", ".safetensors", ".pt")` | Dateiendungen-Filter. |

## Tests

Dieses Projekt verwendet `pytest` für Unit-Tests.

**Empfohlen (mit Dev-Tools):**

```bash
pip install -e ".[dev]"
pytest -v
```

**Ohne Dev-Tools (nur Test-Dependencies):**

```bash
pip install pytest
pytest -v
```

**Coverage lokal messen:**

```bash
pip install pytest-cov
pytest --cov=scripts --cov-report=term-missing tests/
```

## CI

GitHub Actions laufen auf jedem Push zu `main` und jedem Pull-Request:

- **test**: pytest gegen Python **3.10 / 3.11 / 3.12** (Matrix-Build, fail-fast off)
- **secret-scan**: [gitleaks](https://github.com/gitleaks/gitleaks) gegen die **gesamte** Git-Historie

Workflow-Datei: `.github/workflows/ci.yml`. Branch-Schutz-Regeln auf `main` (Recommended: erforderliche Statusprüfungen „test" und „secret-scan" + 1 Review).

Workflow läuft auch auf `workflow_dispatch` für manuelle Re-Runs.

## Screenshots

_Platzhalter — UI-Screenshots werden gerne via PR beigetragen. Die UI ist in `on_ui_tabs()` definiert (Gradio-Blocks mit Scan/Cancel/Move/Permanent-Buttons und Checkbox-Liste)._

## Bekannte Hinweise

- Nur Dateien mit den Endungen `.ckpt`, `.safetensors` und `.pt` werden berücksichtigt.
- Die Hash-Berechnung erfolgt in 1-MB-Blöcken, um den Speicherverbrauch niedrig zu halten.
- **Symlink-Behandlung** (PR #13): `os.walk(..., followlinks=True)` plus Deduplizierung über `os.path.realpath`, sodass dieselbe physische Datei nur einmal gemeldet wird.
- **Cancel-Button** (PR #13): setzt ein `threading.Event`; `iter_model_files` und `find_duplicates` prüfen das Flag zwischen Verzeichnis-Wechseln. Eine bereits laufende SHA256-Berechnung wird aktuell noch zu Ende geführt — die Reaktion erfolgt beim nächsten Checkpoint.
- **Aus PR #13 nicht übernommen** (obsolet oder Trade-off-Konflikt mit Cluster 1/2; siehe [`docs/rationale.md`](docs/rationale.md) für die Begründung):
  - „Select all"-Button für die Checkbox-Liste
  - BLAKE2b-`compute_hash` + 16-MB-Chunks + `os.scandir` (PR #3 hatte das, Cluster 1 hat bewusst SHA256 + 1 MB + `os.walk` gewählt)
  - „Prefer deleting numbered copy filenames"-Logik im Delete-Pfad (refactored durch Cluster 2)
- **TODO / Folge-PRs**: Auto-Retention-Cleanup für `.duplicate_model_finder_trash/`, UI-Fortschrittsbalken für sehr große Scans, evtl. Checkbox „Prefer unnumbered copy" für gezieltere Delete-Auswahl.

## Performance-Hinweise

- **Size-Pre-Filter:** Vor dem SHA256-Hashing werden Dateien nach Größe gruppiert. Nur Dateien, die ihre Größe mit mindestens einer anderen Datei teilen, werden gehasht. Das spart bei großen Modell-Dateien (typischerweise mehrere GB pro `.safetensors`) erheblich I/O, weil eindeutige Dateien gar nicht erst gelesen werden.
- **Paralleles Hashing:** Das Hashing läuft per Default in einem Thread-Pool mit bis zu 8 Workern, abhängig von der verfügbaren CPU-Anzahl. Beide Optimierungen lassen sich getrennt deaktivieren — `find_duplicates(use_size_prefilter=False, max_workers=1)` für rein sequentielles Verhalten (nützlich für deterministische Tests).
- **Skalierung:** Bei 50 Modell-Dateien à 4 GB reduziert der Size-Pre-Filter die zu hashende Datenmenge typischerweise um >95 %; die Parallelisierung skaliert mit der Anzahl physischer Kerne.

## Sicherheit

- **Soft-Delete by Default:** Das Löschen verschiebt Dateien in einen Schwester-Ordner `.duplicate_model_finder_trash/` (mit UTC-Zeitstempel-Präfix zur Vermeidung von Namenskollisionen). Die Dateien bleiben auf der Festplatte und können manuell wiederhergestellt oder endgültig gelöscht werden — kein versehentlicher Datenverlust.
- **Bestätigung für Permanent-Löschung:** Die Buttons „Move selected to trash" und „Permanently delete selected" sind getrennt. Der Permanent-Pfad verlangt zusätzlich eine explizite Bestätigung über die Checkbox „Yes, I really want to permanently delete the selected files" und nutzt intern `permanently_delete_files(paths, confirm=True)`.
- **Trash-Retention (Option C Hybrid, closes #17):** Ein neuer „Empty trash"-Button leert alle `.duplicate_model_finder_trash/`-Ordner innerhalb der `MODEL_DIRS`-Trees. Auch hier ist eine explizite Bestätigungs-Checkbox Pflicht. Die Retention-Frist ist `TRASH_RETENTION_DAYS = 30` Tage (Default, im Code anpassbar). Optional kann die automatische Bereinigung über die Umgebungsvariable `TRASH_AUTO_EMPTY_DAYS` aktiviert werden — wenn nicht gesetzt, ist der UI-Button der einzige Pfad zur dauerhaften Trash-Entleerung. Sub-Directories innerhalb des Trash-Ordners werden bewusst übersprungen, damit manuelle Wiederherstellungs-Strukturen nicht angetastet werden.
- **Permission-Check:** Vor jedem Lösch-Vorgang wird `os.access(path, os.W_OK)` geprüft; nicht beschreibbare Dateien werden übersprungen und im Status-Bericht aufgeführt (gleiche Liste bleibt ausgewählt für Retry).
- **Logging:** Alle Lösch-Vorgänge werden über das `logging`-Modul dokumentiert (Logger-Name `scripts.duplicate_model_finder`). Fehler und verweigerte Aktionen landen auf `WARNING`/`ERROR`-Level, normale Operationen auf `INFO`.
- **Kein Auto-Merge:** PRs erfordern grüne CI + 1 Review (siehe [CONTRIBUTING](CONTRIBUTING.md)).

## Lizenz

MIT — siehe [LICENSE](LICENSE).
