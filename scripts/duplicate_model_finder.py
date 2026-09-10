import concurrent.futures
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import gradio as gr
except ImportError:  # pragma: no cover - only triggered in test environments without Gradio
    gr = None

logger = logging.getLogger(__name__)


MODEL_DIRS: List[str] = [
    os.path.join("models", "Stable-diffusion"),
    os.path.join("models", "Lora"),
    os.path.join("models", "VAE"),
]

ALLOWED_EXTENSIONS: Tuple[str, ...] = (".ckpt", ".safetensors", ".pt")
CHUNK_SIZE = 1 << 20  # 1MB
# Upper bound for the hashing thread pool. The actual worker count is derived
# from this default and the number of CPUs available at runtime.
DEFAULT_HASH_WORKERS = 8
# Sibling directory used for soft-delete. The user can restore or permanently
# delete files from there manually if needed.
TRASH_DIR_NAME = ".duplicate_model_finder_trash"


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
    stop_event: Optional[threading.Event] = None,
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
    stop_event: Optional[threading.Event] = None,
) -> List[str]:
    """Return only paths that share their size with at least one other path.

    Files whose size is unique cannot be duplicates of any other file, so we
    skip the expensive SHA256 hashing step for them. When ``stop_event`` is
    supplied the loop returns whatever it has collected so far.
    """

    size_groups: Dict[int, List[str]] = {}
    for path in paths:
        if stop_event is not None and stop_event.is_set():
            return []
        try:
            size = os.path.getsize(path)
        except OSError:
            # Skip files we cannot stat; the hash step would fail on them too.
            continue
        size_groups.setdefault(size, []).append(path)
    return [
        path
        for group in size_groups.values()
        if len(group) > 1
        for path in group
    ]


def _format_utc_timestamp() -> str:
    """Return the current UTC time formatted as ``YYYYMMDDTHHMMSSffffff``.

    Extracted as a module-level helper so tests can monkeypatch it to a
    fixed value without touching Python's immutable ``datetime`` class.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _resolve_worker_count(max_workers: Optional[int]) -> int:
    """Pick a sane thread pool size based on the explicit override or CPU count."""

    if max_workers is not None and max_workers > 0:
        return max_workers
    cpu_count = os.cpu_count() or 4
    return min(DEFAULT_HASH_WORKERS, max(2, cpu_count))


def collect_hashes(
    file_paths: Iterable[str],
    *,
    use_size_prefilter: bool = True,
    max_workers: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, List[str]]:
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
        _size_duplicate_candidates(paths, stop_event=stop_event)
        if use_size_prefilter
        else paths
    )
    if not candidates:
        return {}

    if stop_event is not None and stop_event.is_set():
        return {}

    hashes: Dict[str, List[str]] = {}
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

    def _hash_one(path: str) -> Optional[Tuple[str, str]]:
        try:
            return path, compute_hash(path)
        except OSError:
            return None

    if stop_event is not None and stop_event.is_set():
        return hashes

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for path, file_hash in executor.map(_hash_one, candidates):
            if stop_event is not None and stop_event.is_set():
                return hashes
            if file_hash is None:
                continue
            hashes.setdefault(file_hash, []).append(path)
    return hashes


def find_duplicates(
    directories: Sequence[str] = MODEL_DIRS,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    *,
    use_size_prefilter: bool = True,
    max_workers: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, List[str]]:
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


def format_duplicates_for_display(duplicates: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """Prepare human-readable text and selection choices for the UI."""

    lines: List[str] = []
    choices: List[str] = []
    for file_hash, paths in duplicates.items():
        lines.append(f"Hash {file_hash}:")
        for path in paths:
            lines.append(f"  {path}")
            choices.append(path)
    text = "\n".join(lines) if lines else "No duplicates found"
    return text, choices


def move_files_to_trash(paths: Sequence[str]) -> Tuple[str, List[str]]:
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

    moved: List[str] = []
    failed: List[str] = []
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
) -> Tuple[str, List[str]]:
    """Permanently delete files. Requires explicit ``confirm=True``.

    Use :func:`move_files_to_trash` for the default safe workflow. This
    function is the irreversible escape hatch and is intentionally hostile
    to silent misuse: passing ``confirm=False`` (the default) returns all
    paths as failed and emits a warning.

    Returns ``(status_message, list_of_paths_that_failed)``.
    """

    if not confirm:
        logger.warning(
            "Refused permanent delete without confirm=True for %d path(s)", len(paths)
        )
        return (
            "Refusing to permanently delete without explicit confirm=True",
            [str(path) for path in paths],
        )

    if not paths:
        return "No files deleted", []

    removed: List[str] = []
    failed: List[str] = []
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


def delete_files(paths: Sequence[str]) -> Tuple[str, List[str]]:
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
            cancel_btn = gr.Button(value="Cancel scan")
            trash_btn = gr.Button(value="Move selected to trash")
            delete_btn = gr.Button(value="Permanently delete selected")

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
        confirm_check = gr.Checkbox(
            label=(
                "Yes, I really want to permanently delete the selected files "
                "(irreversible)"
            ),
            value=False,
        )
        result_box = gr.Textbox(label="Status", interactive=False)

        def do_scan():
            stop_event.clear()
            duplicates = find_duplicates(stop_event=stop_event)
            text, choices = format_duplicates_for_display(duplicates)
            status = "Scan cancelled" if stop_event.is_set() else "Scan complete"
            return text, gr.update(choices=choices, value=[]), status

        def do_trash(selected: List[str]):
            status, failed = move_files_to_trash(selected)
            if failed:
                # Keep the failed entries selected so users can retry.
                return gr.update(value=failed), status
            return gr.update(value=[]), status

        def do_delete(selected: List[str], confirm: bool):
            status, failed = permanently_delete_files(selected, confirm=confirm)
            if failed:
                # Keep the failed entries selected so users can retry.
                return gr.update(value=failed), status
            return gr.update(value=[]), status

        def cancel_scan():
            """Request the running scan to stop."""
            stop_event.set()
            return "Cancelling…"

        scan_btn.click(fn=do_scan, outputs=[duplicates_box, delete_choices, result_box])
        cancel_btn.click(fn=cancel_scan, outputs=result_box)
        trash_btn.click(fn=do_trash, inputs=delete_choices, outputs=[delete_choices, result_box])
        delete_btn.click(
            fn=do_delete,
            inputs=[delete_choices, confirm_check],
            outputs=[delete_choices, result_box],
        )

    return [(ui, "Duplicate Models", "duplicate_model_finder")]
