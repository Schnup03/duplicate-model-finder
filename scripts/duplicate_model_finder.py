import os
import hashlib
from typing import Dict, Iterable, List

try:
    import gradio as gr  # type: ignore
except Exception:  # pragma: no cover - gradio is optional for CLI usage
    gr = None  # type: ignore

try:
    from modules import script_callbacks  # type: ignore
except Exception:  # pragma: no cover - script_callbacks only in WebUI
    script_callbacks = None  # type: ignore


# common model directories that ship with the WebUI
MODEL_DIRS = [
    os.path.join("models", "Stable-diffusion"),
    os.path.join("models", "Lora"),
    os.path.join("models", "VAE"),
]

MODEL_EXTS = (".ckpt", ".safetensors", ".pt")

CHUNK_SIZE = 1 << 20  # 1MB


def compute_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_model_files() -> Iterable[str]:
    """Yield paths to model files under the configured directories.

    Symlinks are followed and only unique real files are yielded to avoid
    duplicate reporting when the same file is linked from multiple locations.
    """
    seen: set[str] = set()
    for dir in MODEL_DIRS:
        if not os.path.isdir(dir):
            continue
        for root, _, files in os.walk(dir, followlinks=True):
            for name in files:
                if not name.endswith(MODEL_EXTS):
                    continue
                path = os.path.join(root, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                yield path


def find_duplicates() -> Dict[str, List[str]]:
    """Search model directories for duplicate files.

    The search first groups files by size to avoid unnecessary hashing and then
    computes SHA256 for files that share the same size.

    Returns a mapping of hash -> list of paths that share that hash.
    """
    size_map: Dict[int, List[str]] = {}
    for path in iter_model_files():
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        size_map.setdefault(size, []).append(path)

    hashes: Dict[str, List[str]] = {}
    for paths in size_map.values():
        if len(paths) < 2:
            continue
        for path in paths:
            try:
                file_hash = compute_hash(path)
            except OSError:
                continue
            hashes.setdefault(file_hash, []).append(path)

    return {h: p for h, p in hashes.items() if len(p) > 1}


if __name__ == "__main__":
    duplicates = find_duplicates()
    for h, paths in duplicates.items():
        print(f"Hash {h}:")
        for p in paths:
            print(f"  {p}")


def on_ui_tabs():
    if gr is None:
        raise RuntimeError("gradio is required for the WebUI extension")

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
            lines = []
            choices = []
            for file_hash, paths in sorted(duplicates.items()):
                lines.append(f"Hash {file_hash}:")
                for p in sorted(paths):
                    lines.append(f"  {p}")
                    choices.append(p)
            text = "\n".join(lines) if lines else "No duplicates found"
            return text, gr.update(choices=choices, value=[]), "Scan complete"

        def do_delete(selected: List[str]):
            removed = []
            for path in selected:
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError:
                    pass
            if removed:
                return f"Deleted {len(removed)} file(s)"
            return "No files deleted"

        scan_btn.click(fn=do_scan, outputs=[duplicates_box, delete_choices, result_box])
        delete_btn.click(fn=do_delete, inputs=delete_choices, outputs=result_box)

    return [(ui, "Duplicate Models", "duplicate_model_finder")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(on_ui_tabs)
