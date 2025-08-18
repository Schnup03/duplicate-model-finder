import os
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Tuple

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

# Larger chunks take better advantage of fast NVMe SSDs
CHUNK_SIZE = 1 << 22  # 4MB


def compute_hash(path: str) -> str:
    """Compute BLAKE2b hash of a file using a reusable buffer."""
    h = hashlib.blake2b()
    buf = bytearray(CHUNK_SIZE)
    mv = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


def iter_model_files(
    stop_event: threading.Event | None = None,
) -> Iterator[Tuple[str, int]]:
    """Yield ``(path, size)`` for model files under the configured directories.

    ``os.scandir`` is used for efficient directory traversal and to obtain file
    sizes in the same system call.  Symlinks are followed and only unique real
    files are yielded to avoid duplicate reporting when the same file is linked
    from multiple locations.  The ``stop_event`` may be set to request early
    termination.
    """

    seen: set[str] = set()

    def _scan(path: str) -> Iterator[Tuple[str, int]]:
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if stop_event and stop_event.is_set():
                        return
                    if entry.is_dir(follow_symlinks=True):
                        yield from _scan(entry.path)
                        if stop_event and stop_event.is_set():
                            return
                        continue
                    if not entry.name.endswith(MODEL_EXTS):
                        continue
                    real = os.path.realpath(entry.path)
                    if real in seen:
                        continue
                    try:
                        size = entry.stat(follow_symlinks=True).st_size
                    except OSError:
                        continue
                    seen.add(real)
                    yield entry.path, size
                    if stop_event and stop_event.is_set():
                        return
        except OSError:
            return

    for dir in MODEL_DIRS:
        if stop_event and stop_event.is_set():
            break
        if os.path.isdir(dir):
            yield from _scan(dir)
            if stop_event and stop_event.is_set():
                break


def find_duplicates(
    stop_event: threading.Event | None = None, hash_workers: int | None = None
) -> Dict[str, List[str]]:
    """Search model directories for duplicate files.

    The search first groups files by size to avoid unnecessary hashing and then
    computes BLAKE2b for files that share the same size. Hashing can be
    parallelised by setting ``hash_workers``.  If ``stop_event`` is set the
    search stops early.

    Returns a mapping of hash -> list of paths that share that hash.
    """
    size_map: Dict[int, List[str]] = {}
    for path, size in iter_model_files(stop_event):
        if stop_event and stop_event.is_set():
            break
        size_map.setdefault(size, []).append(path)

    hashes: Dict[str, List[str]] = {}
    # default to 2x CPU cores to better utilise fast storage and CPUs
    workers = hash_workers or ((os.cpu_count() or 1) * 2)
    for paths in size_map.values():
        if stop_event and stop_event.is_set():
            break
        if len(paths) < 2:
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_path = {ex.submit(compute_hash, p): p for p in paths}
            for future in as_completed(future_to_path):
                if stop_event and stop_event.is_set():
                    ex.shutdown(cancel_futures=True)
                    break
                path = future_to_path[future]
                try:
                    file_hash = future.result()
                except OSError:
                    continue
                hashes.setdefault(file_hash, []).append(path)
        if stop_event and stop_event.is_set():
            break

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
            cancel_btn = gr.Button(value="Cancel scan")
            select_all_btn = gr.Button(value="Select all")
            delete_btn = gr.Button(value="Delete selected")

        max_workers = (os.cpu_count() or 1) * 2
        thread_slider = gr.Slider(
            minimum=1,
            maximum=max_workers,
            value=max_workers,
            step=1,
            label="Hash threads",
        )

        duplicates_box = gr.Textbox(
            label="Duplicate files",
            interactive=False,
            lines=10,
        )
        overview_table = gr.DataFrame(headers=["Keep", "Delete"], interactive=False)
        # Initialize with empty choices so that the list can be updated
        # dynamically after a scan.  Gradio requires the component to be
        # created with a ``choices`` parameter in order to modify it later via
        # ``gr.update``.  Without this, clicking the scan button would raise an
        # error and the UI would appear unresponsive.
        delete_choices = gr.CheckboxGroup(label="Select files to delete", choices=[])
        result_box = gr.Textbox(label="Status", interactive=False)
        choices_state = gr.State([])
        stop_event = threading.Event()

        def do_scan(workers: int):
            stop_event.clear()
            duplicates = find_duplicates(stop_event=stop_event, hash_workers=int(workers))
            lines = []
            choices = []
            table = []
            for file_hash, paths in sorted(duplicates.items()):
                lines.append(f"Hash {file_hash}:")
                sorted_paths = sorted(paths)
                for p in sorted_paths:
                    lines.append(f"  {p}")
                keep = sorted_paths[0] if sorted_paths else ""
                for p in sorted_paths[1:]:
                    table.append([keep, p])
                    choices.append(p)
            text = "\n".join(lines) if lines else "No duplicates found"
            status = "Scan cancelled" if stop_event.is_set() else "Scan complete"
            return (
                text,
                gr.update(choices=choices, value=[]),
                table,
                choices,
                status,
            )

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

        def select_all(choices: List[str]):
            """Return an update selecting all available choices."""
            return gr.update(value=choices)

        def cancel_scan():
            """Request the running scan to stop."""
            stop_event.set()
            return "Cancelling..."

        scan_btn.click(
            fn=do_scan,
            inputs=thread_slider,
            outputs=[duplicates_box, delete_choices, overview_table, choices_state, result_box],
        )
        cancel_btn.click(fn=cancel_scan, outputs=result_box)
        select_all_btn.click(
            fn=select_all,
            inputs=choices_state,
            outputs=delete_choices,
        )
        delete_btn.click(fn=do_delete, inputs=delete_choices, outputs=result_box)

    return [(ui, "Duplicate Models", "duplicate_model_finder")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(on_ui_tabs)
