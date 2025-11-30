import hashlib
import os
from typing import Dict, Iterable, List, Sequence, Tuple

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


def iter_model_files(directories: Sequence[str]) -> Iterable[str]:
    """Yield model file paths from the provided directories."""

    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for name in files:
                if is_model_file(name):
                    yield os.path.join(root, name)


def collect_hashes(file_paths: Iterable[str]) -> Dict[str, List[str]]:
    """Compute hashes for the provided files and group identical ones."""

    hashes: Dict[str, List[str]] = {}
    for path in file_paths:
        path_str = str(path)
        file_hash = compute_hash(path_str)
        hashes.setdefault(file_hash, []).append(path_str)
    return hashes


def find_duplicates(
    directories: Sequence[str] = MODEL_DIRS,
    extensions: Sequence[str] = ALLOWED_EXTENSIONS,
) -> Dict[str, List[str]]:
    """Search model directories for duplicate files.

    Returns a mapping of hash -> list of paths that share that hash.
    """

    files = iter_model_files(directories)
    filtered_files = (path for path in files if is_model_file(os.path.basename(path), extensions))
    hashes = collect_hashes(filtered_files)
    return {file_hash: paths for file_hash, paths in hashes.items() if len(paths) > 1}


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
