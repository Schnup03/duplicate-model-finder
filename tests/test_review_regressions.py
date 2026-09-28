import os
import sys
import threading
import types
from datetime import datetime, timedelta, timezone

import pytest

from scripts import duplicate_model_finder as dmf


@pytest.mark.parametrize("prefer_nvme", [False, True])
def test_same_size_different_models_are_not_duplicates(tmp_path, prefer_nvme):
    (tmp_path / "first.ckpt").write_bytes(b"AAAA")
    (tmp_path / "second.ckpt").write_bytes(b"BBBB")
    assert dmf.find_duplicates([str(tmp_path)], prefer_nvme=prefer_nvme) == {}


@pytest.mark.parametrize("prefer_nvme", [False, True])
def test_trashed_models_do_not_reappear_in_scan(tmp_path, prefer_nvme):
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert dmf.move_files_to_trash([str(second)])[1] == []
    assert dmf.find_duplicates([str(tmp_path)], prefer_nvme=prefer_nvme) == {}


def test_retention_starts_when_model_enters_trash(tmp_path):
    model = tmp_path / "old.ckpt"
    model.write_bytes(b"old model")
    old_mtime = datetime.now(timezone.utc).timestamp() - 60 * 86400
    os.utime(model, (old_mtime, old_mtime))
    assert dmf.move_files_to_trash([str(model)])[1] == []
    trashed = next((tmp_path / dmf.TRASH_DIR_NAME).iterdir())
    assert dmf.purge_old_trash(30)[1] == 0
    assert trashed.exists()


def test_auto_cleanup_uses_validated_positive_retention(tmp_path, monkeypatch):
    trash = tmp_path / dmf.TRASH_DIR_NAME
    trash.mkdir()
    disposed_at = datetime.now(timezone.utc) - timedelta(days=3)
    old = trash / f"{disposed_at:%Y%m%dT%H%M%S%f}_model.ckpt"
    old.write_bytes(b"x")
    monkeypatch.setenv(dmf.TRASH_AUTO_EMPTY_DAYS_ENV, "0")
    assert "skipped" in dmf.auto_purge_trash([str(tmp_path)])
    assert old.exists()
    monkeypatch.setenv(dmf.TRASH_AUTO_EMPTY_DAYS_ENV, "2")
    assert "Purged 1" in dmf.auto_purge_trash([str(tmp_path)])
    assert not old.exists()


def test_webui_model_directories_include_configured_roots(tmp_path, monkeypatch):
    paths = types.SimpleNamespace(models_path=str(tmp_path / "webui_models"))
    shared = types.SimpleNamespace(
        cmd_opts=types.SimpleNamespace(
            ckpt_dir=str(tmp_path / "checkpoints"),
            lora_dir=str(tmp_path / "loras"),
            vae_dir=None,
        )
    )
    monkeypatch.setitem(sys.modules, "modules", types.SimpleNamespace(paths=paths, shared=shared))
    roots = set(dmf.get_model_directories())
    assert str(tmp_path / "webui_models" / "Stable-diffusion") in roots
    assert str(tmp_path / "webui_models" / "Lora") in roots
    assert str(tmp_path / "webui_models" / "VAE") in roots
    assert str(tmp_path / "checkpoints") in roots
    assert str(tmp_path / "loras") in roots


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_windows_junctions_cannot_move_or_delete_external_models(tmp_path):
    import _winapi

    hot = tmp_path / "hot"
    archive = tmp_path / "archive"
    hot.mkdir()
    archive.mkdir()
    original = archive / "original.ckpt"
    original.write_bytes(b"archive model")
    alias = hot / "linked_archive"
    _winapi.CreateJunction(str(archive), str(alias))
    try:
        linked = str(alias / original.name)
        assert dmf.move_files_to_trash([linked], allowed_roots=[str(hot)])[1] == [linked]
        assert dmf.permanently_delete_files([linked], confirm=True, allowed_roots=[str(hot)])[
            1
        ] == [linked]
        assert original.read_bytes() == b"archive model"
    finally:
        os.rmdir(alias)  # Remove the junction only, never its target.


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
@pytest.mark.parametrize("prefer_nvme", [False, True])
def test_windows_junction_cycle_terminates_in_both_scanners(tmp_path, prefer_nvme):
    import _winapi

    root = tmp_path / "models"
    root.mkdir()
    (root / "one.ckpt").write_bytes(b"one")
    alias = root / "back"
    _winapi.CreateJunction(str(root), str(alias))
    timer_fired = threading.Event()
    stop = threading.Event()
    timer = threading.Timer(2, lambda: (timer_fired.set(), stop.set()))
    timer.start()
    try:
        walker = dmf.iter_model_files_scandir if prefer_nvme else dmf.iter_model_files
        assert list(walker([str(root)], stop_event=stop)) == [str(root / "one.ckpt")]
        assert not timer_fired.is_set(), "scanner followed a junction cycle until timeout"
    finally:
        stop.set()
        timer.cancel()
        os.rmdir(alias)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_trash_junction_cannot_purge_external_files(tmp_path):
    import _winapi

    root = tmp_path / "models"
    archive = tmp_path / "archive"
    root.mkdir()
    archive.mkdir()
    disposed_at = datetime.now(timezone.utc) - timedelta(days=90)
    external = archive / f"{disposed_at:%Y%m%dT%H%M%S%f}_model.ckpt"
    external.write_bytes(b"external")
    alias = root / dmf.TRASH_DIR_NAME
    _winapi.CreateJunction(str(archive), str(alias))
    try:
        assert dmf.purge_old_trash(30, allowed_roots=[str(root)])[1] == 0
        assert external.read_bytes() == b"external"
    finally:
        os.rmdir(alias)


class FakeComponent:
    components = []
    bindings = []

    def __init__(self, *args, **kwargs):
        self.value = kwargs.get("value")
        self.label = kwargs.get("label")
        self.components.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def click(self, **kwargs):
        self.bindings.append((self, kwargs))


@pytest.fixture
def ui_bindings(monkeypatch):
    FakeComponent.components = []
    FakeComponent.bindings = []
    fake_gradio = types.SimpleNamespace(
        **{
            name: FakeComponent
            for name in (
                "Blocks",
                "Row",
                "Button",
                "Markdown",
                "Textbox",
                "CheckboxGroup",
                "State",
                "Checkbox",
            )
        }
    )
    fake_gradio.Progress = lambda: lambda *args, **kwargs: None
    fake_gradio.update = lambda **kwargs: kwargs
    monkeypatch.setattr(dmf, "gr", fake_gradio)
    monkeypatch.setattr(dmf, "_ui_tab_mounted", False)
    monkeypatch.delenv(dmf.TRASH_AUTO_EMPTY_DAYS_ENV, raising=False)
    tabs = dmf.on_ui_tabs()
    assert len(tabs) == 1
    return {component.value: binding for component, binding in FakeComponent.bindings}


@pytest.mark.parametrize("prefer_nvme", [False, True])
def test_ui_buttons_scan_select_and_trash_a_duplicate(tmp_path, ui_bindings, prefer_nvme):
    labels = set(ui_bindings)
    assert labels == {
        "Scan for duplicates",
        "Select extra copies",
        "Cancel scan",
        "Move selected to trash",
        "Permanently delete selected",
        "Empty trash",
    }
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    scan_binding = ui_bindings["Scan for duplicates"]
    assert len(scan_binding["outputs"]) == 4
    emissions = list(scan_binding["fn"](prefer_nvme))
    assert len(emissions) == 3
    assert all(len(event) == 4 for event in emissions)
    text, update, snapshot, status = emissions[-1]
    assert "Scan complete" in status
    assert str(first) in text and str(second) in text
    assert set(update["choices"]) == {str(first), str(second)}
    select = ui_bindings["Select extra copies"]
    assert select["inputs"] is scan_binding["outputs"][2]
    selected = select["fn"](snapshot)
    assert len(selected) == 1
    trash = ui_bindings["Move selected to trash"]
    assert len(trash["outputs"]) == 4
    after = trash["fn"](selected, snapshot)
    assert "Moved 1 file" in after[3]
    assert after[1]["choices"] == []
    assert len([path for path in (first, second) if path.exists()]) == 1
    assert len(list((tmp_path / dmf.TRASH_DIR_NAME).iterdir())) == 1


def test_ui_cancel_confirmation_and_empty_trash_controls(tmp_path, ui_bindings, monkeypatch):
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    scan = ui_bindings["Scan for duplicates"]["fn"]
    running = scan(False)
    assert len(next(running)) == 4
    cancel = ui_bindings["Cancel scan"]
    assert cancel["fn"]() == "Cancelling…"
    cancelled = list(running)[-1]
    assert len(cancelled) == 4
    assert "cancelled" in cancelled[3]
    assert cancelled[1]["choices"] == []

    snapshot = list(scan(False))[-1][2]
    delete = ui_bindings["Permanently delete selected"]["fn"]
    assert "confirmation" in delete([str(second)], snapshot, False)[3]
    assert second.exists()
    assert "Keep at least one" in delete([str(first), str(second)], snapshot, True)[3]
    assert first.exists() and second.exists()
    assert "Permanently deleted 1" in delete([str(second)], snapshot, True)[3]
    assert first.exists() and not second.exists()

    trash = tmp_path / dmf.TRASH_DIR_NAME
    trash.mkdir()
    disposed_at = datetime.now(timezone.utc) - timedelta(days=90)
    (trash / f"{disposed_at:%Y%m%dT%H%M%S%f}_old.ckpt").write_bytes(b"old")
    empty = ui_bindings["Empty trash"]
    assert "confirmation required" in empty["fn"](False)
    assert "Purged 1" in empty["fn"](True)
