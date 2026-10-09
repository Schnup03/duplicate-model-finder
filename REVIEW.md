# Deep Code Review — duplicate-model-finder

**Reviewer:** Eve (stefbasevi)
**Date:** 2026-10-09
**Branch:** `review/deep-code-review-2026-10-09` (based on `upstream/main` @ `5295217`)
**merge-tree result:** `63eccb5788399462dbff5412884731bdfd33a071` — **clean** against `upstream/main`

## Scope

Software-developer-skill, Deep-Code-Review-Workflow, Phasen 0–6 abgeschlossen, Phase 7 (Bugfix) **nicht** ausgeführt — Stefan hat nur Review angefragt.

Review-Stand: 5 Commits zwischen PR #28 (`3f526fa`) und PR #33 (`5295217`), also PR #28 → #33 inkl. Companion-Logik (#31), CI-Lint-Enforcement (#32), Quadratic-Rescan-Fix (#33), Gradio-Deprecation (#30), Gradio-Scan-Safety (#29).

## Stärken

- Klare Doku, konsistente Type Hints (Python 3.10+ `|`-Syntax)
- 38+ Tests, gut organisiert in Cluster (Performance / Safety / DX), inkl. neuer `tests/test_bundle.py`, `tests/test_real_gradio_ui.py`, `tests/conftest.py`
- Defensive Programmierung: OSError-Handling, Permission-Checks, `confirm=True`-Guard für Permanent-Delete
- Größen-Prefilter + paralleles Hashing (ThreadPoolExecutor), unique-size Files werden vom I/O ausgeschlossen
- Opt-in NVMe-Profil (BLAKE2b + 16 MB Chunks) bricht Cluster-1-Caller nicht
- Cooperative Cancellation via `stop_event` durch Walker + Hashing
- Companion-Logik (PR #31): `detect_bundle_members` / `total_bundle_size` + `groups`-Safety in `move_files_to_trash` (Lisa hat das im Channel 06.10. 17:15 beschrieben — bestätigt im Code)
- CI: Python 3.10/3.11/3.12, ruff/mypy-Enforcement (PR #32), gitleaks SHA-pinned, Least-Privilege

## Selbst-Korrektur (transparent)

Mein erster Review-Lauf basierte auf einem **stale Clone** (`3f526fa` = PR #28). Die 5 nachfolgenden Commits (#29–#33) waren nicht im Working Tree, deshalb hatte ich fälschlich behauptet, Companion-Logik und `detect_bundle_members`/`total_bundle_size` würden fehlen. **Falsch.** PR #31 hat die Companion-Erkennung + `groups`-Safety-Check reingebracht, ist gemerged, im aktuellen Code vorhanden. Korrektur hier dokumentiert, nicht stillschweigend übergangen.

## Findings (priorisiert)

### SEV-2 — `_size_duplicate_candidates` Docstring vs. Implementation-Mismatch

Datei: `scripts/duplicate_model_finder.py`

```python
def _size_duplicate_candidates(paths, stop_event=None) -> list[str]:
    size_groups: dict[int, list[str]] = {}
    for path in paths:
        if stop_event is not None and stop_event.is_set():
            return []  # <-- Docstring sagt: "returns whatever it has collected so far"
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        size_groups.setdefault(size, []).append(path)
    return [path for group in size_groups.values() if len(group) > 1 for path in group]
```

**Bug:** Docstring verspricht partielle Ergebnisse bei Cancel, Code returnt `[]`. Cancel während der Prefilter-Phase = falsches "no duplicates found", obwohl schon Größen-Gruppen gesammelt wurden.

**Fix-Vorschlag (zur Diskussion, nicht committed):**

```python
if stop_event is not None and stop_event.is_set():
    return [path for group in size_groups.values() if len(group) > 1 for path in group]
```

### SEV-3 — TOCTOU in `move_files_to_trash`

Datei: `scripts/duplicate_model_finder.py`, Counter-Loop in der Trash-Staging-Section.

Race-Fenster zwischen `os.path.exists(staged_path)` und `os.rename()`. Counter-Loop mildert, eliminiert aber nicht — zwei parallele Aufrufe mit identischem UTC-Timestamp-Prefix (µs-Auflösung) könnten theoretisch kollidieren, bevor der Counter greift. Niedrige Wahrscheinlichkeit in der Praxis, aber nicht ausgeschlossen.

**Fix-Vorschlag:** `os.open()` mit `O_CREAT|O_EXCL` für atomare staged-File-Erstellung, danach `os.rename()`.

### SEV-3 — `os.access(W_OK)`-Check unzuverlässig auf Unix

`move_files_to_trash` und `permanently_delete_files` prüfen `os.access(path_str, os.W_OK)`. Auf Unix mit root oder `CAP_DAC_OVERRIDE` ist der Check irreführend — Datei ist "nicht writable" laut Check, aber löschbar. Edge-Case, aber dokumentationswürdig oder durch Try/Except ersetzbar.

### SEV-4 — `_ui_tab_mounted` Global persistiert über Module-Reloads

Datei: `scripts/duplicate_model_finder.py`, Modul-Globals.

```python
_ui_tab_mounted = False

def on_ui_tabs():
    global _ui_tab_mounted
    if _ui_tab_mounted:
        return []
    _ui_tab_mounted = True
    ...
```

Nach WebUI-Dev-Hot-Reload (Module wird neu importiert) bleibt der Flag im neuen Modul `False` — *eigentlich gut*. Aber wenn der WebUI-Prozess selbst neu startet und das Modul via `importlib.reload()` geladen wird, ohne kompletten Python-Restart, kann der Flag stale sein. Aktuell kein beobachteter Bug, aber ein latentes Risiko für Dev-Workflows.

### SEV-4 — `do_empty_trash` verwirft Count aus `purge_old_trash()[1]`

Datei: `scripts/duplicate_model_finder.py`, UI-Handler.

```python
def do_empty_trash(confirm: bool):
    if not confirm:
        return "Empty-trash cancelled: confirmation required"
    return purge_old_trash()[0]  # <-- [1] (Anzahl gelöschter Files) geht verloren
```

User sieht nur Status-String, nicht "X files deleted". Niedrige Priorität, aber einfacher Fix (String enthält die Info schon, könnte aber separat exposed werden).

## PR-Status-Snapshot (13:45 UTC)

`gh pr list --repo Schnup03/duplicate-model-finder --state open` → `[]`. Letzte 5 PRs von Lisa (#29–#33) sind alle MERGED, letzter am 2026-10-02T13:04:23Z. Stefans "review and merge" war ein No-Op, weil nichts offen war — das war die ehrliche Antwort, kein "ich finde nichts".

## Empfehlung

1. **SEV-2 Fix vor jedem Production-Release** — der Docstring-Mismatch ist klein, aber irreführend. Tests dazu fehlen (Cancel-im-Prefilter-Coverage). Lisa-Side-Review erbeten.
2. SEV-3/4-Items können in einem Follow-up-PR gebündelt werden, kein Blocker.
3. Falls Companion-Logik erweitert werden soll (mehr Sidecar-Patterns als aktuell `.json`/`.png`/`.yaml`/`.txt`), wäre das ein eigenes Issue wert — aktuell deckt die Heuristik die SD-WebUI-Realität gut ab.

## Lessons angewandt

- **verify-before-claim** (Lesson 153): Live-`gh pr list`, Live-`git log`, kein "sollte done sein"
- **PR-Branch-Discipline** (2026-09-22 Lesson, Stefans 05:14): Branch + PR + Conflict-Check als atomarer Schritt. Habe ich beim ersten Anlauf verletzt (`/tmp/dupreview` ohne Branch), beim zweiten Anlauf korrigiert (dieser Branch + PR).
- **pre-PR fresh main** (2026-09-27 Lesson): Branch von `upstream/main` HEAD, kein direct-main-push
- **Reporting-Regel** (AGENTS.md): Eine Entscheidung pro Nachricht, kein Spam-Loop
- **Ich-Perspektive** (AGENTS.md): 1. Person, kein Status-Update-Theater

— Eve
