import os
import hashlib
from typing import Dict, List

import gradio as gr


MODEL_DIRS = [
    os.path.join("models", "Stable-diffusion"),
    os.path.join("models", "Lora"),
    os.path.join("models", "VAE"),
]

CHUNK_SIZE = 1 << 20  # 1MB


def compute_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates() -> Dict[str, List[str]]:
    """Search model directories for duplicate files.

    Returns a mapping of hash -> list of paths that share that hash.
    """
    hashes: Dict[str, List[str]] = {}
    for dir in MODEL_DIRS:
        if not os.path.isdir(dir):
            continue
        for root, _, files in os.walk(dir):
            for name in files:
                if name.endswith((".ckpt", ".safetensors", ".pt")):
                    path = os.path.join(root, name)
                    file_hash = compute_hash(path)
                    hashes.setdefault(file_hash, []).append(path)
    return {h: p for h, p in hashes.items() if len(p) > 1}


def on_ui_tabs():
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
            for file_hash, paths in duplicates.items():
                lines.append(f"Hash {file_hash}:")
                for p in paths:
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
