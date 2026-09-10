import concurrent.futures
import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import gradio as gr
except ImportError:  # pragma: no cover - only triggered in test environments without Gradio
    gr = None


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
) -> Iterable[str]:
    """Yield model file paths from the provided directories."""

    ext_tuple = tuple(ext.lower() for ext in extensions)
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(ext_tuple):
                    yield os.path.join(root, name)


def _size_duplicate_candidates(paths: Sequence[str]) -> List[str]:
    """Return only paths that share their size with at least one other path.

    Files whose size is unique cannot be duplicates of any other file, so we
    skip the expensive SHA256 hashing step for them.
    """

    size_groups: Dict[int, List[str]] = {}
    for path in paths:
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
) -> Dict[str, List[str]]:
    """Compute hashes for the provided files and group identical ones.

    With ``use_size_prefilter=True`` (default) files whose size is unique
    among the input set are skipped, which avoids hashing large model files
    that cannot possibly be duplicates of any other file.

    Hashing is I/O-bound, so by default it runs across a thread pool sized
    from ``max_workers`` (when given) or the CPU count. Pass ``max_workers=1``
    to disable parallelism, e.g. for deterministic tests.
    """

    paths = [str(path) for path in file_paths]
    if not paths:
        return {}

    candidates = _size_duplicate_candidates(paths) if use_size_prefilter else paths
    if not candidates:
        return {}

    hashes: Dict[str, List[str]] = {}
    if len(candidates) == 1 or max_workers == 1:
        for path in candidates:
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for path, file_hash in executor.map(_hash_one, candidates):
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
) -> Dict[str, List[str]]:
    """Search model directories for duplicate files.

    Returns a mapping of hash -> list of paths that share that hash.
    """

    files = list(iter_model_files(directories, extensions))
    return collect_hashes(
        files,
        use_size_prefilter=use_size_prefilter,
        max_workers=max_workers,
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


def delete_files(paths: Sequence[str]) -> Tuple[str, List[str]]:
    """Attempt to delete the provided paths and report the result."""

    removed: List[str] = []
    failed: List[str] = []
    for path in paths:
        path_str = str(path)
        try:
            os.remove(path_str)
            removed.append(path_str)
        except OSError:
            failed.append(path_str)
    if removed and not failed:
        message = f"Deleted {len(removed)} file(s)"
    elif removed and failed:
        message = f"Deleted {len(removed)} file(s); failed to delete {len(failed)}"
    else:
        message = "No files deleted"
    return message, failed


def on_ui_tabs():
    if gr is None:  # pragma: no cover - safety net for environments ohne Gradio
        raise ImportError("Gradio ist nicht installiert und wird für die UI benötigt.")

    with gr.Blocks() as ui:
        gr.Markdown("## Duplicate Model Finder")

        with gr.Row():
            scan_btn = gr.Button(value="Scan for duplicates")
            delete_btn = gr.Button(value="Delete selected")

        duplicates_box = gr.Textbox(
            label="Duplicate files",
            interactive=False,
            lines=10,
        )
        delete_choices = gr.CheckboxGroup(label="Select files to delete")
        result_box = gr.Textbox(label="Status", interactive=False)

        def do_scan():
            duplicates = find_duplicates()
            text, choices = format_duplicates_for_display(duplicates)
            return text, gr.update(choices=choices, value=[]), "Scan complete"

        def do_delete(selected: List[str]):
            status, failed = delete_files(selected)
            if failed:
                # Keep the failed entries selected so users can retry.
                return gr.update(value=failed), status
            return gr.update(value=[]), status

        scan_btn.click(fn=do_scan, outputs=[duplicates_box, delete_choices, result_box])
        delete_btn.click(fn=do_delete, inputs=delete_choices, outputs=[delete_choices, result_box])

    return [(ui, "Duplicate Models", "duplicate_model_finder")]
