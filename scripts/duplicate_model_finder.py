import concurrent.futures
import hashlib
import logging
import os
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

try:
    import gradio as gr
except ImportError:  # pragma: no cover - only triggered in test environments without Gradio
    gr = None

logger = logging.getLogger(__name__)


MODEL_DIRS: list[str] = [
    os.path.join("models", "Stable-diffusion"),
    os.path.join("models", "Lora"),
    os.path.join("models", "VAE"),
]

ALLOWED_EXTENSIONS: tuple[str, ...] = (".ckpt", ".safetensors", ".pt")
CHUNK_SIZE = 1 << 20  # 1MB
# Upper bound for the hashing thread pool. The actual worker count is derived
# from this default and the number of CPUs available at runtime.
DEFAULT_HASH_WORKERS = 8
# Sibling directory used for soft-delete. The user can restore or permanently
# delete files from there manually if needed.
TRASH_DIR_NAME = ".duplicate_model_finder_trash"
# Default retention window for ``purge_old_trash``: files older than this many
# days are eligible for automatic deletion when ``TRASH_AUTO_EMPTY_DAYS`` is
# set in the environment. The UI button always uses this default as well; the
# env-var only controls *whether* auto-purge runs at scan-start.
TRASH_RETENTION_DAYS = 30
# Environment variable that opts the WebUI into automatic trash purging. When
# set to an integer it enables auto-purge with that many days as retention;
# when unset the UI button is the only path that can remove files from trash.
TRASH_AUTO_EMPTY_DAYS_ENV = "TRASH_AUTO_EMPTY_DAYS"


def is_model_file(file_name: str, extensions: Sequence[str] = ALLOWED_EXTENSIONS) -> bool:
    """Return True if the file name ends with a supported model extension."""

    return file_name.lower().endswith(tuple(ext.lower() for ext in extensions))


def compute_hash(path: str, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA256 hash of a file in chunks to limit memory usage."""

    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_model_files(
    directories: Sequence[str],
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    stop_event: threading.Event | None = None,
) -> Iterable[str]:
    """Yield model file paths from the provided directories.

    Symlinks are followed and only unique real files are yielded so that the
    same physical file linked from multiple locations is reported once. When
    ``stop_event`` is supplied the walk checks the flag between directories
    and before each yielded path so a long-running scan can be cancelled
    cooperatively.
    """

    ext_tuple = tuple(ext.lower() for ext in extensions)
    seen: set[str] = set()
    for directory in directories:
        if stop_event is not None and stop_event.is_set():
            return
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory, followlinks=True):
            if stop_event is not None and stop_event.is_set():
                return
            for name in files:
                if stop_event is not None and stop_event.is_set():
                    return
                if not name.lower().endswith(ext_tuple):
                    continue
                path = os.path.join(root, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                yield path


def _size_duplicate_candidates(
    paths: Sequence[str],
    stop_event: threading.Event | None = None,
) -> list[str]:
    """Return only paths that share their size with at least one other path.

    Files whose size is unique cannot be duplicates of any other file, so we
    skip the expensive SHA256 hashing step for them. When ``stop_event`` is
    supplied the loop returns whatever it has collected so far.
    """

    size_groups: dict[int, list[str]] = {}
    for path in paths:
        if stop_event is not None and stop_event.is_set():
            return []
        try:
            size = os.path.getsize(path)
        except OSError:
            # Skip files we cannot stat; the hash step would fail on them too.
            continue
        size_groups.setdefault(size, []).append(path)
    return [path for group in size_groups.values() if len(group) > 1 for path in group]


def _format_utc_timestamp() -> str:
    """Return the current UTC time formatted as ``YYYYMMDDTHHMMSSffffff``.

    Extracted as a module-level helper so tests can monkeypatch it to a
    fixed value without touching Python's immutable ``datetime`` class.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _resolve_worker_count(max_workers: int | None) -> int:
    """Pick a sane thread pool size based on the explicit override or CPU count."""

    if max_workers is not None and max_workers > 0:
        return max_workers
    cpu_count = os.cpu_count() or 4
    return min(DEFAULT_HASH_WORKERS, max(2, cpu_count))


def collect_hashes(
    file_paths: Iterable[str],
    *,
    use_size_prefilter: bool = True,
    max_workers: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, list[str]]:
    """Compute hashes for the provided files and group identical ones.

    With ``use_size_prefilter=True`` (default) files whose size is unique
    among the input set are skipped, which avoids hashing large model files
    that cannot possibly be duplicates of any other file.

    Hashing is I/O-bound, so by default it runs across a thread pool sized
    from ``max_workers`` (when given) or the CPU count. Pass ``max_workers=1``
    to disable parallelism, e.g. for deterministic tests.

    When ``stop_event`` is supplied the function returns whatever it has
    accumulated so far; the threaded executor is released via its context
    manager on exit.
    """

    paths = [str(path) for path in file_paths]
    if not paths:
        return {}

    if stop_event is not None and stop_event.is_set():
        return {}

    candidates = (
        _size_duplicate_candidates(paths, stop_event=stop_event) if use_size_prefilter else paths
    )
    if not candidates:
        return {}

    if stop_event is not None and stop_event.is_set():
        return {}

    hashes: dict[str, list[str]] = {}
    if len(candidates) == 1 or max_workers == 1:
        for path in candidates:
            if stop_event is not None and stop_event.is_set():
                return hashes
            try:
                file_hash = compute_hash(path)
            except OSError:
                # File disappeared between the size prefilter and the hash
                # step — skip it rather than crash the whole scan.
                continue
            hashes.setdefault(file_hash, []).append(path)
        return hashes

    workers = _resolve_worker_count(max_workers)

    def _hash_one(path: str) -> tuple[str, str] | None:
        try:
            return path, compute_hash(path)
        except OSError:
            return None

    if stop_event is not None and stop_event.is_set():
        return hashes

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_hash_one, candidates):
            if stop_event is not None and stop_event.is_set():
                return hashes
            if result is None:
                # File disappeared mid-scan (deleted, permission changed,
                # symlink target removed, etc.) — skip it cleanly. Crashes
                # the whole scan otherwise (issue #16).
                continue
            path, file_hash = result
            hashes.setdefault(file_hash, []).append(path)
    return hashes


def find_duplicates(
    directories: Sequence[str] = MODEL_DIRS,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    *,
    use_size_prefilter: bool = True,
    max_workers: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, list[str]]:
    """Search model directories for duplicate files.

    Returns a mapping of hash -> list of paths that share that hash.
    ``stop_event`` is forwarded to ``iter_model_files`` and the hashing
    pipeline so a long-running scan can be cancelled cooperatively.
    """

    files = list(iter_model_files(directories, extensions, stop_event=stop_event))
    return collect_hashes(
        files,
        use_size_prefilter=use_size_prefilter,
        max_workers=max_workers,
        stop_event=stop_event,
    )


def format_duplicates_for_display(duplicates: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Prepare human-readable text and selection choices for the UI."""

    lines: list[str] = []
    choices: list[str] = []
    for file_hash, paths in duplicates.items():
        lines.append(f"Hash {file_hash}:")
        for path in paths:
            lines.append(f"  {path}")
            choices.append(path)
    text = "\n".join(lines) if lines else "No duplicates found"
    return text, choices


def purge_old_trash(retention_days: int = TRASH_RETENTION_DAYS) -> tuple[str, int]:
    """Delete files in any ``.duplicate_model_finder_trash/`` older than N days.

    Scans every ``MODEL_DIRS`` tree for ``TRASH_DIR_NAME`` sub-directories and
    removes files whose ``mtime`` is older than the cutoff. Returns a
    human-readable status and the number of files removed.
    """
    if retention_days < 0:
        return "Invalid retention_days: must be >= 0", 0
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    trash_dirs_scanned = 0
    for root_dir in MODEL_DIRS:
        if not os.path.isdir(root_dir):
            continue
        for current, dirs, _files in os.walk(root_dir):
            if TRASH_DIR_NAME in dirs:
                trash_path = os.path.join(current, TRASH_DIR_NAME)
                trash_dirs_scanned += 1
                try:
                    for entry in os.listdir(trash_path):
                        entry_path = os.path.join(trash_path, entry)
                        if not os.path.isfile(entry_path):
                            continue
                        try:
                            if os.path.getmtime(entry_path) < cutoff:
                                os.unlink(entry_path)
                                deleted += 1
                                logger.info("Purged trash file: %s", entry_path)
                        except OSError as exc:
                            logger.error("Cannot purge %s: %s", entry_path, exc)
                except OSError as exc:
                    logger.error("Cannot list trash dir %s: %s", trash_path, exc)
                # Don't descend into the trash dir for further walking.
                dirs.remove(TRASH_DIR_NAME)
    return (
        f"Purged {deleted} file(s) older than {retention_days} day(s) "
        f"from {trash_dirs_scanned} trash dir(s)",
        deleted,
    )


def move_files_to_trash(paths: Sequence[str]) -> tuple[str, list[str]]:
    """Move files into a sibling ``.duplicate_model_finder_trash/`` directory.

    This is the default safe workflow. Files remain on disk inside the trash
    folder (with a UTC-timestamp prefix to avoid collisions) so the user can
    inspect, restore, or permanently delete them later. Missing files and
    files without write permission are skipped and reported in the failure
    list rather than aborting the whole batch.

    Returns ``(status_message, list_of_paths_that_failed)``.
    """

    if not paths:
        return "No files moved", []

    moved: list[str] = []
    failed: list[str] = []
    for path in paths:
        path_str = str(path)
        if not os.path.exists(path_str):
            logger.warning("Skipping %s: file does not exist", path_str)
            failed.append(path_str)
            continue
        if not os.access(path_str, os.W_OK):
            logger.warning("No write permission for %s — cannot move", path_str)
            failed.append(path_str)
            continue
        parent = os.path.dirname(os.path.abspath(path_str)) or "."
        trash_dir = os.path.join(parent, TRASH_DIR_NAME)
        try:
            os.makedirs(trash_dir, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create trash dir %s: %s", trash_dir, exc)
            failed.append(path_str)
            continue
        timestamp = _format_utc_timestamp()
        base = os.path.basename(path_str)
        staged_name = f"{timestamp}_{base}"
        staged_path = os.path.join(trash_dir, staged_name)
        counter = 1
        while os.path.exists(staged_path):
            staged_name = f"{timestamp}_{counter}_{base}"
            staged_path = os.path.join(trash_dir, staged_name)
            counter += 1
        try:
            os.rename(path_str, staged_path)
            moved.append(path_str)
            logger.info("Moved to trash: %s -> %s", path_str, staged_path)
        except OSError as exc:
            logger.error("Failed to move %s to trash: %s", path_str, exc)
            failed.append(path_str)

    if moved and not failed:
        message = f"Moved {len(moved)} file(s) to {TRASH_DIR_NAME}/ (recoverable manually)"
    elif moved and failed:
        message = f"Moved {len(moved)} file(s); failed to move {len(failed)}"
    else:
        message = "No files moved"
    return message, failed


def permanently_delete_files(
    paths: Sequence[str],
    confirm: bool = False,
) -> tuple[str, list[str]]:
    """Permanently delete files. Requires explicit ``confirm=True``.

    Use :func:`move_files_to_trash` for the default safe workflow. This
    function is the irreversible escape hatch and is intentionally hostile
    to silent misuse: passing ``confirm=False`` (the default) returns all
    paths as failed and emits a warning.

    Returns ``(status_message, list_of_paths_that_failed)``.
    """

    if not confirm:
        logger.warning("Refused permanent delete without confirm=True for %d path(s)", len(paths))
        return (
            "Refusing to permanently delete without explicit confirm=True",
            [str(path) for path in paths],
        )

    if not paths:
        return "No files deleted", []

    removed: list[str] = []
    failed: list[str] = []
    for path in paths:
        path_str = str(path)
        if not os.access(path_str, os.W_OK):
            logger.warning("No write permission for %s — cannot delete", path_str)
            failed.append(path_str)
            continue
        try:
            os.remove(path_str)
            removed.append(path_str)
            logger.info("Permanently deleted %s", path_str)
        except OSError as exc:
            logger.error("Failed to delete %s: %s", path_str, exc)
            failed.append(path_str)

    if removed and not failed:
        return f"Permanently deleted {len(removed)} file(s)", failed
    if removed and failed:
        return f"Permanently deleted {len(removed)} file(s); failed to delete {len(failed)}", failed
    return "No files deleted", failed


def delete_files(paths: Sequence[str]) -> tuple[str, list[str]]:
    """Deprecated: kept for backward compatibility, now routes to move_files_to_trash().

    Use :func:`move_files_to_trash` or
    :func:`permanently_delete_files` ``(paths, confirm=True)`` explicitly in
    new code.
    """

    logger.debug("delete_files() called — redirecting to move_files_to_trash()")
    return move_files_to_trash(paths)


def on_ui_tabs():
    if gr is None:  # pragma: no cover - safety net for environments ohne Gradio
        raise ImportError("Gradio ist nicht installiert und wird für die UI benötigt.")

    stop_event = threading.Event()

    with gr.Blocks() as ui:
        gr.Markdown("## Duplicate Model Finder")

        with gr.Row():
            scan_btn = gr.Button(value="Scan for duplicates")
            select_all_btn = gr.Button(value="Select all")
            cancel_btn = gr.Button(value="Cancel scan")
            trash_btn = gr.Button(value="Move selected to trash")
            delete_btn = gr.Button(value="Permanently delete selected")
            empty_trash_btn = gr.Button(value="Empty trash")

        duplicates_box = gr.Textbox(
            label="Duplicate files",
            interactive=False,
            lines=10,
        )
        # Initialize with empty choices so that the list can be updated
        # dynamically after a scan.  Gradio requires the component to be
        # created with a ``choices`` parameter in order to modify it later via
        # ``gr.update`` (fix from PR #3, commit 0efaead).
        delete_choices = gr.CheckboxGroup(label="Select files to handle", choices=[])
        # Holds the current choice list so the ``select_all_btn`` knows which
        # paths belong to the last scan. Updated by ``do_scan`` after every
        # run (restores the helper from PR #3 / commit 1b10314 that was
        # skipped during the PR #13 rebase; closes #18).
        choices_state = gr.State([])
        confirm_check = gr.Checkbox(
            label=("Yes, I really want to permanently delete the selected files " "(irreversible)"),
            value=False,
        )
        empty_trash_confirm = gr.Checkbox(
            label=("Yes, empty the entire trash (irreversible, all model file(s) "
                   "in .duplicate_model_finder_trash/ older than the retention window)"),
            value=False,
        )
        result_box = gr.Textbox(label="Status", interactive=False)

        def do_scan(progress=gr.Progress()):
            """Run the duplicate scan with progressive UI updates (closes #6).

            Implemented as a generator so Gradio can refresh the UI between
            the file-walk and the hashing phases. ``gr.Progress()`` is the
            default-argument form Gradio injects at runtime; it renders as a
            non-blocking progress indicator and keeps the WebUI responsive
            while ``find_duplicates`` walks the model directories.
            """
            stop_event.clear()
            progress(0, desc="Starting scan…")
            yield (
                "Scanning…",
                gr.update(choices=[], value=[]),
                f"Scanning {len(MODEL_DIRS)} root dir(s)…",
            )
            files = list(iter_model_files(MODEL_DIRS, stop_event=stop_event))
            if stop_event.is_set():
                progress(1.0, desc="Cancelled")
                yield (
                    "Scan cancelled",
                    gr.update(choices=[], value=[]),
                    "Cancelled before hashing",
                )
                return
            progress(0.2, desc=f"Found {len(files)} model file(s); hashing…")
            yield (
                f"Found {len(files)} model file(s); computing hashes…",
                gr.update(choices=[], value=[]),
                f"Hashing {len(files)} file(s)…",
            )
            duplicates = collect_hashes(files, stop_event=stop_event)
            text, choices = format_duplicates_for_display(duplicates)
            progress(1.0, desc="Done")
            status = (
                "Scan cancelled"
                if stop_event.is_set()
                else f"Scan complete — {len(duplicates)} hash group(s)"
            )
            # Third return value pushes the freshly computed choices into
            # ``choices_state`` so the select-all button can act on them.
            return text, gr.update(choices=choices, value=[]), choices, status

        def select_all_choices(choices: list[str]) -> list[str]:
            """Return the full choice list to mark every entry as selected."""
            return list(choices)

        def do_trash(selected: list[str]):
            status, failed = move_files_to_trash(selected)
            if failed:
                # Keep the failed entries selected so users can retry.
                return gr.update(value=failed), status
            return gr.update(value=[]), status

        def do_delete(selected: list[str], confirm: bool):
            status, failed = permanently_delete_files(selected, confirm=confirm)
            if failed:
                # Keep the failed entries selected so users can retry.
                return gr.update(value=failed), status
            return gr.update(value=[]), status

        def do_empty_trash(confirm: bool):
            """Permanently delete all trash files older than the retention window."""
            if not confirm:
                return "Empty-trash cancelled: confirmation required"
            return purge_old_trash()[0]

        def cancel_scan():
            """Request the running scan to stop."""
            stop_event.set()
            return "Cancelling…"

        scan_btn.click(
            fn=do_scan,
            outputs=[duplicates_box, delete_choices, choices_state, result_box],
        )
        select_all_btn.click(
            fn=select_all_choices,
            inputs=choices_state,
            outputs=delete_choices,
        )
        cancel_btn.click(fn=cancel_scan, outputs=result_box)
        trash_btn.click(fn=do_trash, inputs=delete_choices, outputs=[delete_choices, result_box])
        delete_btn.click(
            fn=do_delete,
            inputs=[delete_choices, confirm_check],
            outputs=[delete_choices, result_box],
        )
        empty_trash_btn.click(
            fn=do_empty_trash,
            inputs=empty_trash_confirm,
            outputs=result_box,
        )

    return [(ui, "Duplicate Models", "duplicate_model_finder")]
