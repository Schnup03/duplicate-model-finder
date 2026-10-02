import concurrent.futures
import hashlib
import logging
import os
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from functools import wraps
from typing import Any

# Gradio is optional: the module has to stay importable for the scan/delete
# helpers and for the test suite, even where no WebUI environment exists.
gr: Any
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
# Days since a file was moved to trash, not since its contents were modified.
TRASH_RETENTION_DAYS = 30
# A positive integer opts in to automatic purge at UI scan start.
TRASH_AUTO_EMPTY_DAYS_ENV = "TRASH_AUTO_EMPTY_DAYS"
# Optional alternative profile; throughput depends on storage and CPU.
NVME_CHUNK_SIZE = 16 * 1024 * 1024
_operation_lock = threading.RLock()
_ui_operation_lock = threading.Lock()


def _serialized_operation(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _operation_lock:
            return function(*args, **kwargs)

    return locked


def _real_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def get_model_directories() -> list[str]:
    """Resolve the running WebUI's roots; keep MODEL_DIRS as standalone fallback."""
    try:
        from modules import paths, shared
    except ImportError:
        directories = list(MODEL_DIRS)
    else:
        model_root = getattr(paths, "models_path", None)
        directories = (
            [os.path.join(model_root, category) for category in ("Stable-diffusion", "Lora", "VAE")]
            if model_root
            else list(MODEL_DIRS)
        )
        options = getattr(shared, "cmd_opts", None)
        for name in ("ckpt_dir", "lora_dir", "vae_dir"):
            configured = getattr(options, name, None)
            if configured:
                directories.append(configured)
    unique: dict[str, str] = {}
    for directory in directories:
        absolute = os.path.abspath(directory)
        unique.setdefault(_real_path(absolute), absolute)
    return list(unique.values())


def _inside_roots(path: str, roots: Sequence[str]) -> bool:
    """Compare resolved paths, including junctions and Windows case folding."""
    real = _real_path(path)
    for root in roots:
        resolved_root = _real_path(root)
        try:
            if real != resolved_root and os.path.commonpath([real, resolved_root]) == resolved_root:
                return True
        except ValueError:  # Different Windows drives.
            continue
    return False


def _in_trash(path: str) -> bool:
    # Check both the visible path and the target of directory/file links.
    return any(
        TRASH_DIR_NAME.casefold() in normalized.replace("\\", "/").casefold().split("/")
        for normalized in (os.path.abspath(path), _real_path(path))
    )


def _safe_model_path(path: str, roots: Sequence[str]) -> bool:
    return (
        is_model_file(path)
        and os.path.isfile(path)
        and not os.path.islink(path)
        and not _in_trash(path)
        and _inside_roots(path, roots)
    )


def _file_identity(path: str):
    """Also collapse hardlinks; some filesystems do not expose an inode."""
    try:
        info = os.stat(path)
        if info.st_ino:
            return info.st_dev, info.st_ino
    except OSError:
        pass
    return _real_path(path)


def duplicate_groups(hashes: dict[str, list[str]]) -> dict[str, list[str]]:
    """Keep only groups containing at least two different physical files."""
    result: dict[str, list[str]] = {}
    for digest, paths in hashes.items():
        unique: dict[Any, str] = {}
        for path in paths:
            unique.setdefault(_file_identity(path), path)
        if len(unique) > 1:
            result[digest] = list(unique.values())
    return result


def is_model_file(file_name: str, extensions: Sequence[str] = ALLOWED_EXTENSIONS) -> bool:
    """Return True if the file name ends with a supported model extension."""

    return file_name.lower().endswith(tuple(ext.lower() for ext in extensions))


# Companion files that belong to a model bundle: metadata, previews and
# configs that travel with the weight file. Everything else in the folder
# (README.md, unrelated checkpoints, …) is deliberately not matched.
COMPANION_DOUBLE_EXTS: frozenset[str] = frozenset({".civitai.info", ".info"})
COMPANION_SINGLE_EXTS: frozenset[str] = frozenset(
    {".json", ".html", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".webp"}
)


def _path_suffixes(path: str) -> tuple[str, ...]:
    """All suffixes of a path, lowercased, in order.

    ``model.safetensors`` -> ``('.safetensors',)``,
    ``model.safetensors.json`` -> ``('.safetensors', '.json')``.
    """

    name = os.path.basename(path)
    return tuple("." + part.lower() for part in name.split(".")[1:] if part)


def _is_companion(name: str) -> bool:
    """True when the file name carries an allowed companion extension."""

    suffixes = _path_suffixes(name)
    if not suffixes:
        return False
    if len(suffixes) >= 2 and (suffixes[-2] + suffixes[-1]) in COMPANION_DOUBLE_EXTS:
        return True
    return suffixes[-1] in COMPANION_SINGLE_EXTS


def _bundle_stems(name: str) -> list[str]:
    """Every meaningful stem prefix of a file name.

    ``model.safetensors`` -> ``['model']``,
    ``model.safetensors.json`` -> ``['model.safetensors', 'model']``.
    """

    parts = name.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts)) if parts[:index]]


def detect_bundle_members(model_path: str) -> list[str]:
    """Return the companion files that belong to ``model_path``.

    Companions are siblings in the same directory whose name starts with one
    of the model's stems and whose extension is a known companion extension.
    ``model.safetensors`` does not appear in the result. A missing or
    non-file path yields an empty list.
    """

    if not os.path.isfile(model_path):
        return []
    parent = os.path.dirname(os.path.abspath(model_path))
    if not os.path.isdir(parent):
        return []
    stems = _bundle_stems(os.path.basename(model_path))
    if not stems:
        return []
    found: list[str] = []
    try:
        entries = sorted(os.listdir(parent))
    except OSError:
        return []
    base = os.path.basename(model_path)
    for entry in entries:
        if entry == base or not entry.startswith(tuple(f"{stem}." for stem in stems)):
            continue
        candidate = os.path.join(parent, entry)
        if _is_companion(entry) and os.path.isfile(candidate) and not os.path.islink(candidate):
            found.append(candidate)
    return found


def total_bundle_size(model_path: str) -> int:
    """Total bytes of a model plus its detected companions (0 if missing)."""

    if not os.path.isfile(model_path):
        return 0
    total = os.path.getsize(model_path)
    for companion in detect_bundle_members(model_path):
        try:
            total += os.path.getsize(companion)
        except OSError:
            continue
    return total


def compute_hash(path: str, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA256 hash of a file in chunks to limit memory usage."""

    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_hash_blake2b(path: str, chunk_size: int = NVME_CHUNK_SIZE) -> str:
    """Compute a 128-character BLAKE2b digest using larger read chunks."""

    hasher = hashlib.blake2b()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _scandir_walk(
    directory: str, stop_event: threading.Event | None = None
) -> Iterable[tuple[str, list[str], list[str]]]:
    """Iterative walk; callers can prune the mutable dirs list before descent."""
    pending: list[str] = [directory]
    while pending:
        current = pending.pop()
        if stop_event is not None and stop_event.is_set():
            return
        try:
            scandir_it = os.scandir(current)
        except OSError:
            continue
        dirs: list[str] = []
        files: list[str] = []
        with scandir_it:
            for entry in scandir_it:
                try:
                    if entry.is_dir(follow_symlinks=True):
                        dirs.append(entry.name)
                    elif entry.is_file(follow_symlinks=True):
                        files.append(entry.name)
                except OSError:
                    continue
        yield current, dirs, files
        for d in dirs:
            pending.append(os.path.join(current, d))


def _walk_directories(directories, stop_event=None, *, prefer_nvme=False, include_trash=False):
    """Prune cycles across roots and exclude trash, including aliases to it."""
    visited = set()
    for directory in directories:
        if stop_event is not None and stop_event.is_set():
            return
        if not os.path.isdir(directory):
            continue
        walker = (
            _scandir_walk(directory, stop_event)
            if prefer_nvme
            else os.walk(directory, topdown=True, followlinks=True)
        )
        for root, dirs, files in walker:
            if stop_event is not None and stop_event.is_set():
                return
            identity = _file_identity(root)
            if identity in visited or (not include_trash and _in_trash(root)):
                dirs.clear()
                continue
            visited.add(identity)
            dirs[:] = [
                name
                for name in dirs
                if _file_identity(os.path.join(root, name)) not in visited
                and (include_trash or not _in_trash(os.path.join(root, name)))
            ]
            yield root, dirs, files


def iter_model_files(
    directories: Sequence[str] | None = None,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    stop_event: threading.Event | None = None,
) -> Iterable[str]:
    """Yield unique model files, following links without traversing cycles or trash."""
    yield from _iter_model_files(directories, extensions, stop_event, prefer_nvme=False)


def iter_model_files_scandir(
    directories: Sequence[str] | None = None,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    stop_event: threading.Event | None = None,
) -> Iterable[str]:
    """The iterative scandir variant with the same safety rules as the default."""
    yield from _iter_model_files(directories, extensions, stop_event, prefer_nvme=True)


def _iter_model_files(directories, extensions, stop_event, *, prefer_nvme):
    roots = get_model_directories() if directories is None else directories
    seen = set()
    for root, _dirs, files in _walk_directories(roots, stop_event, prefer_nvme=prefer_nvme):
        for name in files:
            if stop_event is not None and stop_event.is_set():
                return
            if not is_model_file(name, extensions):
                continue
            path = os.path.join(root, name)
            if _in_trash(path) or not os.path.isfile(path):
                continue
            identity = _file_identity(path)
            if identity not in seen:
                seen.add(identity)
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
    prefer_nvme: bool = False,
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
                file_hash = compute_hash_blake2b(path) if prefer_nvme else compute_hash(path)
            except OSError:
                # File disappeared between the size prefilter and the hash
                # step — skip it rather than crash the whole scan.
                continue
            hashes.setdefault(file_hash, []).append(path)
        return hashes

    workers = _resolve_worker_count(max_workers)

    def _hash_one(path: str) -> tuple[str, str] | None:
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            return path, (compute_hash_blake2b(path) if prefer_nvme else compute_hash(path))
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


@_serialized_operation
def find_duplicates(
    directories: Sequence[str] | None = None,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
    *,
    use_size_prefilter: bool = True,
    max_workers: int | None = None,
    stop_event: threading.Event | None = None,
    prefer_nvme: bool = False,
) -> dict[str, list[str]]:
    """Search model directories for duplicate files.

    Returns a mapping of hash -> list of paths that share that hash.
    ``stop_event`` is forwarded to ``iter_model_files`` and the hashing
    pipeline so a long-running scan can be cancelled cooperatively.

    ``prefer_nvme=True`` swaps in the NVMe-tuned profile: ``os.scandir``
    walker + BLAKE2b hashing with 16 MB chunks (closes #19). Off by default
    so Cluster-1 callers (SHA256/1 MB/os.walk) stay byte-compatible.
    """

    walker = iter_model_files_scandir if prefer_nvme else iter_model_files
    files = list(walker(directories, extensions, stop_event=stop_event))
    hashes = collect_hashes(
        files,
        use_size_prefilter=use_size_prefilter,
        max_workers=max_workers,
        stop_event=stop_event,
        prefer_nvme=prefer_nvme,
    )
    return duplicate_groups(hashes)


def format_duplicates_for_display(duplicates: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Prepare human-readable text and selection choices for the UI.

    Companion files (metadata, previews) are listed read-only under their
    model so the user can see what would be left orphaned. They are never
    part of the returned choices and therefore never deletable.
    """

    lines: list[str] = []
    choices: list[str] = []
    for file_hash, paths in duplicate_groups(duplicates).items():
        lines.append(f"Hash {file_hash}:")
        for path in paths:
            lines.append(f"  {path}")
            companions = detect_bundle_members(path)
            if companions:
                lines.append(f"    companions ({len(companions)}):")
                lines.extend(f"      {name}" for name in companions)
            choices.append(path)
    text = "\n".join(lines) if lines else "No duplicates found"
    return text, choices


def _trashed_at(name: str) -> float | None:
    try:
        timestamp, _original_name = name.split("_", 1)
        return (
            datetime.strptime(timestamp, "%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc).timestamp()
        )
    except (ValueError, OverflowError, OSError):
        return None


@_serialized_operation
def purge_old_trash(
    retention_days: int = TRASH_RETENTION_DAYS, *, allowed_roots: Sequence[str] | None = None
) -> tuple[str, int]:
    """Purge by disposal time. Zero explicitly empties regular files, even legacy names.

    For positive retention, unknown names are preserved rather than guessing their age.
    Directory/file links and locations outside the configured roots are never purged.
    """
    if not isinstance(retention_days, int) or not 0 <= retention_days <= 36500:
        return "Invalid retention_days: must be an integer between 0 and 36500", 0
    roots = get_model_directories() if allowed_roots is None else list(allowed_roots)
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    trash_dirs_scanned = 0
    resolved_roots = {_real_path(root) for root in roots}
    for current, dirs, _files in _walk_directories(roots, include_trash=True):
        if _in_trash(current) or (
            _real_path(current) not in resolved_roots and not _inside_roots(current, roots)
        ):
            dirs.clear()
            continue
        trash_path = os.path.join(_real_path(current), TRASH_DIR_NAME)
        if TRASH_DIR_NAME in dirs and _plain_trash_directory(trash_path, roots):
            trash_dirs_scanned += 1
            try:
                for entry in os.listdir(trash_path):
                    entry_path = os.path.join(trash_path, entry)
                    if os.path.islink(entry_path) or not os.path.isfile(entry_path):
                        continue
                    disposed = _trashed_at(entry)
                    if retention_days != 0 and (disposed is None or disposed >= cutoff):
                        continue
                    try:
                        # Recheck the directory immediately before deletion.
                        if not _plain_trash_directory(trash_path, roots):
                            break
                        if not _inside_roots(entry_path, roots) or os.path.islink(entry_path):
                            continue
                        os.unlink(entry_path)
                        deleted += 1
                        logger.info("Purged trash file: %s", entry_path)
                    except OSError as exc:
                        logger.error("Cannot purge %s: %s", entry_path, exc)
            except OSError as exc:
                logger.error("Cannot list trash dir %s: %s", trash_path, exc)
        dirs[:] = [name for name in dirs if not _in_trash(os.path.join(current, name))]
    return (
        f"Purged {deleted} file(s) older than {retention_days} day(s) "
        f"from {trash_dirs_scanned} trash dir(s)",
        deleted,
    )


def auto_purge_trash(roots: Sequence[str]) -> str:
    configured = os.environ.get(TRASH_AUTO_EMPTY_DAYS_ENV)
    if configured is None:
        return ""
    try:
        days = int(configured)
        if not 1 <= days <= 36500:
            raise ValueError
    except ValueError:
        return f"Automatic trash cleanup skipped: {TRASH_AUTO_EMPTY_DAYS_ENV} must be 1–36500"
    return purge_old_trash(days, allowed_roots=roots)[0]


def _plain_trash_directory(path: str, roots: Sequence[str]) -> bool:
    return (
        not os.path.islink(path)
        and _real_path(path) == os.path.normcase(os.path.abspath(path))
        and _inside_roots(path, roots)
        and (not os.path.lexists(path) or os.path.isdir(path))
    )


def _would_remove_last_copy(paths: Sequence[str], groups: dict[str, list[str]] | None) -> bool:
    """True when removing ``paths`` would wipe out a whole duplicate group.

    A group is only endangered when it currently has files on disk and *all*
    of them are part of the request. Groups whose files are already gone do
    not block anything, and a missing/empty group map means the caller opted
    out of the check.
    """

    if not groups:
        return False
    requested = {str(path) for path in paths}
    for group_paths in groups.values():
        existing = {path for path in group_paths if os.path.isfile(path)}
        if existing and existing.issubset(requested):
            return True
    return False


@_serialized_operation
def move_files_to_trash(
    paths: Sequence[str],
    *,
    allowed_roots: Sequence[str] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> tuple[str, list[str]]:
    """Move files into a sibling ``.duplicate_model_finder_trash/`` directory.

    This is the default safe workflow. Files remain on disk inside the trash
    folder (with a UTC-timestamp prefix to avoid collisions) so the user can
    inspect, restore, or permanently delete them later. Missing files and
    files without write permission are skipped and reported in the failure
    list rather than aborting the whole batch.

    ``groups`` is the mapping returned by :func:`find_duplicates`. Pass it to
    keep the last copy of every duplicate group alive; without it the function
    moves exactly what it is given. The UI always supplies it.

    Returns ``(status_message, list_of_paths_that_failed)``.
    """

    if not paths:
        return "No files moved", []

    if _would_remove_last_copy(paths, groups):
        logger.warning("Refused to move every copy of a duplicate group: %s", list(paths))
        return "Keeping at least one copy of each model", [str(path) for path in paths]

    roots = get_model_directories() if allowed_roots is None else list(allowed_roots)
    moved: list[str] = []
    failed: list[str] = []
    for path in paths:
        path_str = str(path)
        if not _safe_model_path(path_str, roots):
            logger.warning(
                "Refusing non-model, missing, linked, external or trashed path: %s", path_str
            )
            failed.append(path_str)
            continue
        if not os.path.exists(path_str):
            logger.warning("Skipping %s: file does not exist", path_str)
            failed.append(path_str)
            continue
        if not os.access(path_str, os.W_OK):
            logger.warning("No write permission for %s — cannot move", path_str)
            failed.append(path_str)
            continue
        source = _real_path(path_str)
        parent = os.path.dirname(source)
        trash_dir = os.path.join(parent, TRASH_DIR_NAME)
        if not _plain_trash_directory(trash_dir, roots):
            logger.warning("Refusing redirected trash directory: %s", trash_dir)
            failed.append(path_str)
            continue
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
        while os.path.lexists(staged_path):
            staged_name = f"{timestamp}_{counter}_{base}"
            staged_path = os.path.join(trash_dir, staged_name)
            counter += 1
        try:
            if not _safe_model_path(source, roots) or not _plain_trash_directory(trash_dir, roots):
                failed.append(path_str)
                continue
            os.rename(source, staged_path)
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


@_serialized_operation
def permanently_delete_files(
    paths: Sequence[str],
    confirm: bool = False,
    *,
    allowed_roots: Sequence[str] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> tuple[str, list[str]]:
    """Permanently delete files. Requires explicit ``confirm=True``.

    Use :func:`move_files_to_trash` for the default safe workflow. This
    function is the irreversible escape hatch and is intentionally hostile
    to silent misuse: passing ``confirm=False`` (the default) returns all
    paths as failed and emits a warning.

    ``groups`` behaves exactly as in :func:`move_files_to_trash`: pass the
    mapping from :func:`find_duplicates` to keep the last copy of every
    duplicate group alive.

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

    if _would_remove_last_copy(paths, groups):
        logger.warning("Refused to delete every copy of a duplicate group: %s", list(paths))
        return "Keeping at least one copy of each model", [str(path) for path in paths]

    roots = get_model_directories() if allowed_roots is None else list(allowed_roots)
    removed: list[str] = []
    failed: list[str] = []
    for path in paths:
        path_str = str(path)
        if not _safe_model_path(path_str, roots):
            logger.warning(
                "Refusing non-model, missing, linked, external or trashed path: %s", path_str
            )
            failed.append(path_str)
            continue
        if not os.access(path_str, os.W_OK):
            logger.warning("No write permission for %s — cannot delete", path_str)
            failed.append(path_str)
            continue
        try:
            source = _real_path(path_str)
            if not _safe_model_path(source, roots):
                failed.append(path_str)
                continue
            os.remove(source)
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


def delete_files(
    paths: Sequence[str], *, allowed_roots: Sequence[str] | None = None
) -> tuple[str, list[str]]:
    """Deprecated: kept for backward compatibility, now routes to move_files_to_trash().

    Use :func:`move_files_to_trash` or
    :func:`permanently_delete_files` ``(paths, confirm=True)`` explicitly in
    new code.
    """

    logger.debug("delete_files() called — redirecting to move_files_to_trash()")
    return move_files_to_trash(paths, allowed_roots=allowed_roots)


# Guard flag: prevents double-mount if both A1111's auto-discovery AND
# script_callbacks.on_ui_tabs() registration paths are active in the same
# WebUI instance. Reset only on full WebUI restart (Python module reload).
_ui_tab_mounted = False


def on_ui_tabs():
    global _ui_tab_mounted
    if _ui_tab_mounted:
        # Guard against double-mount: can happen when Forge/reForge activates
        # both the module-function path (A1111 auto-discovery) AND the
        # script_callbacks.on_ui_tabs() hook. We want the tab mounted exactly
        # once regardless of which path fires first.
        print(
            "[duplicate_model_finder] on_ui_tabs() already mounted, skipping duplicate",
            flush=True,
        )
        return []
    print("[duplicate_model_finder] on_ui_tabs() CALLED", flush=True)
    if gr is None:  # pragma: no cover - safety net for environments ohne Gradio
        raise ImportError("Gradio ist nicht installiert und wird für die UI benötigt.")

    stop_event = threading.Event()

    # ``title=`` und ``analytics_enabled=False`` sind Forge/reForge-Best-Practice.
    # Ohne ``title=`` rendern manche Forge-Versionen den Tab nicht korrekt im Mount-Layer,
    # und ``analytics_enabled=False`` verhindert gradio-Telemetry-Calls die im WebUI-Kontext
    # nichts zu suchen haben. vanilla A1111 akzeptiert die Parameter ebenfalls problemlos.
    with gr.Blocks(title="Duplicate Models", analytics_enabled=False) as ui:
        gr.Markdown("## Duplicate Model Finder")

        with gr.Row():
            scan_btn = gr.Button(value="Scan for duplicates")
            select_all_btn = gr.Button(value="Select extra copies")
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
        # Server-side groups and roots validate checkbox submissions.
        choices_state = gr.State({"groups": {}, "roots": []})
        gr.Markdown(
            "Linked models outside the configured model folders are shown but protected from deletion."
        )
        confirm_check = gr.Checkbox(
            label=("Yes, I really want to permanently delete the selected files (irreversible)"),
            value=False,
        )
        empty_trash_confirm = gr.Checkbox(
            label=(
                "Yes, empty the entire trash (irreversible, all model file(s) "
                "in .duplicate_model_finder_trash/ older than the retention window)"
            ),
            value=False,
        )
        nvme_check = gr.Checkbox(
            label=(
                "Aggressive scan (NVMe-optimized, BLAKE2b + 16 MB chunks, "
                "opt-in feature flag from issue #19)"
            ),
            value=False,
        )
        result_box = gr.Textbox(label="Status", interactive=False)

        def view(groups, roots, status, selected=()):
            groups = duplicate_groups(groups)
            text, _paths = format_duplicates_for_display(groups)
            choices = [
                path for paths in groups.values() for path in paths if _safe_model_path(path, roots)
            ]
            return (
                text,
                gr.update(choices=choices, value=[path for path in selected if path in choices]),
                {"groups": groups, "roots": list(roots)},
                status,
            )

        def do_scan(prefer_nvme: bool = False, progress=gr.Progress()):
            """Every progress event, including the final result, has four outputs."""
            if not _ui_operation_lock.acquire(blocking=False):
                yield view({}, get_model_directories(), "Another scan or file operation is running")
                return
            try:
                stop_event.clear()
                roots = get_model_directories()
                cleanup = auto_purge_trash(roots)
                progress(0, desc="Starting scan…")
                yield view({}, roots, f"Scanning {len(roots)} model folder(s)… {cleanup}".strip())
                walker = iter_model_files_scandir if prefer_nvme else iter_model_files
                files = list(walker(roots, stop_event=stop_event))
                if stop_event.is_set():
                    progress(1.0, desc="Cancelled")
                    yield view({}, roots, "Scan cancelled before hashing")
                    return
                progress(0.2, desc=f"Found {len(files)} model file(s); hashing…")
                yield view({}, roots, f"Hashing {len(files)} model file(s)…")
                hashes = collect_hashes(files, stop_event=stop_event, prefer_nvme=prefer_nvme)
                if stop_event.is_set():
                    progress(1.0, desc="Cancelled")
                    yield view({}, roots, "Scan cancelled during hashing")
                    return
                groups = duplicate_groups(hashes)
                progress(1.0, desc="Done")
                profile = " (NVMe profile)" if prefer_nvme else ""
                yield view(
                    groups, roots, f"Scan complete{profile}: {len(groups)} duplicate group(s)"
                )
            finally:
                _ui_operation_lock.release()

        def select_extra_copies(snapshot):
            """Bulk-select duplicates while retaining one physical copy per group."""
            snapshot = snapshot or {"groups": {}, "roots": []}
            selected = []
            for paths in snapshot["groups"].values():
                protected = [
                    path for path in paths if not _safe_model_path(path, snapshot["roots"])
                ]
                survivor = protected[0] if protected else paths[0]
                selected.extend(
                    path
                    for path in paths
                    if path != survivor and _safe_model_path(path, snapshot["roots"])
                )
            return selected

        def handle_selection(selected, snapshot, *, permanent=False, confirm=False):
            snapshot = snapshot or {"groups": {}, "roots": []}
            selected = selected or []
            if not _ui_operation_lock.acquire(blocking=False):
                return view(
                    snapshot["groups"],
                    snapshot["roots"],
                    "A scan or file operation is running",
                    selected,
                )
            try:
                roots = snapshot["roots"]
                groups = snapshot["groups"]
                eligible = {
                    path
                    for paths in groups.values()
                    for path in paths
                    if _safe_model_path(path, roots)
                }
                if not selected:
                    return view(groups, roots, "Select an extra copy first")
                if any(path not in eligible for path in selected):
                    return view(groups, roots, "Selection changed; scan again before deleting")
                selected_set = set(selected)
                for paths in groups.values():
                    existing = {path for path in paths if os.path.isfile(path)}
                    if existing and existing.issubset(selected_set):
                        return view(groups, roots, "Keep at least one copy of each model", selected)
                if permanent and not confirm:
                    return view(groups, roots, "Permanent deletion requires confirmation", selected)
                if permanent:
                    status, failed = permanently_delete_files(
                        selected, confirm=True, allowed_roots=roots, groups=groups
                    )
                else:
                    status, failed = move_files_to_trash(
                        selected, allowed_roots=roots, groups=groups
                    )
                remaining = duplicate_groups(
                    {
                        digest: [
                            path for path in paths if os.path.isfile(path) and not _in_trash(path)
                        ]
                        for digest, paths in groups.items()
                    }
                )
                return view(remaining, roots, status, failed)
            finally:
                _ui_operation_lock.release()

        def do_trash(selected, snapshot):
            return handle_selection(selected, snapshot)

        def do_delete(selected, snapshot, confirm):
            return handle_selection(selected, snapshot, permanent=True, confirm=confirm)

        def do_empty_trash(confirm: bool):
            """Permanently delete all trash files older than the retention window."""
            if not confirm:
                return "Empty-trash cancelled: confirmation required"
            if not _ui_operation_lock.acquire(blocking=False):
                return "A scan or file operation is running"
            try:
                return purge_old_trash()[0]
            finally:
                _ui_operation_lock.release()

        def cancel_scan():
            """Request the running scan to stop."""
            stop_event.set()
            return "Cancelling…"

        scan_btn.click(
            fn=do_scan,
            inputs=nvme_check,
            outputs=[duplicates_box, delete_choices, choices_state, result_box],
        )
        select_all_btn.click(
            fn=select_extra_copies,
            inputs=choices_state,
            outputs=delete_choices,
        )
        cancel_btn.click(fn=cancel_scan, outputs=result_box)
        trash_btn.click(
            fn=do_trash,
            inputs=[delete_choices, choices_state],
            outputs=[duplicates_box, delete_choices, choices_state, result_box],
        )
        delete_btn.click(
            fn=do_delete,
            inputs=[delete_choices, choices_state, confirm_check],
            outputs=[duplicates_box, delete_choices, choices_state, result_box],
        )
        empty_trash_btn.click(
            fn=do_empty_trash,
            inputs=empty_trash_confirm,
            outputs=result_box,
        )

    _ui_tab_mounted = True
    result = [(ui, "Duplicate Models", "duplicate_model_finder")]
    print(f"[duplicate_model_finder] on_ui_tabs() RETURNING {len(result)} tab(s)", flush=True)
    return result


# Forge/reForge-compatible tab registration via script_callbacks.
#
# Why: reForge 1.10.x has its own extension layer that does NOT always
# auto-detect module-level on_ui_tabs() the way vanilla A1111 does. The
# script_callbacks.on_ui_tabs(callback) hook is the documented A1111+/Forge
# API and works on every modern WebUI. Registering it here is a no-op for
# vanilla A1111 (which still finds the module function via auto-discovery)
# and ensures the tab actually mounts on Forge/reForge when the auto-discovery
# path is the one silently failing.
#
# The dedupe guard inside on_ui_tabs() prevents double-mount when both paths
# fire in the same WebUI instance.
try:
    from modules import script_callbacks

    script_callbacks.on_ui_tabs(on_ui_tabs)
    print(
        "[duplicate_model_finder] Registered via script_callbacks.on_ui_tabs() (Forge/A1111 path)",
        flush=True,
    )
except ImportError:
    # Standalone Python (tests) or vanilla A1111 with auto-discovery only -
    # the module-level on_ui_tabs() above is the registration path.
    print(
        "[duplicate_model_finder] script_callbacks unavailable - "
        "module-level on_ui_tabs() is the registration path",
        flush=True,
    )
