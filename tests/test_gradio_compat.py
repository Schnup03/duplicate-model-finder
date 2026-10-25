"""Gradio wiring that holds for every Gradio version the WebUI may ship.

``test_real_gradio_ui.py`` covers the Gradio 3.x callback API that CI pins.
The checks here are version-agnostic (Gradio 3.41.2 for A1111, 4.x for
Forge-Neo) and run against a real Gradio instance: they verify that the tab
registers every action and that the destructive actions reset their
confirmation checkbox.
"""

import pytest

from scripts import duplicate_model_finder as dmf

gr = pytest.importorskip("gradio")

EXPECTED_CALLBACKS = {
    "do_scan",
    "select_extra_copies",
    "cancel_scan",
    "do_trash",
    "do_delete",
    "do_empty_trash",
}


def _callback_functions(ui):
    """Gradio 3 exposes ``ui.fns`` as a list, Gradio 4 as a dict."""
    fns = ui.fns
    if isinstance(fns, dict):
        return [getattr(entry, "fn", entry) for entry in fns.values()]
    return [block.fn for block in fns if getattr(block, "fn", None) is not None]


def test_real_gradio_ui_registers_every_action(monkeypatch):
    monkeypatch.setattr(dmf, "_ui_tab_mounted", False)

    tabs = dmf.on_ui_tabs()

    assert len(tabs) == 1
    names = {fn.__name__ for fn in _callback_functions(tabs[0][0])}
    assert names >= EXPECTED_CALLBACKS


def test_real_gradio_delete_resets_the_confirmation_checkbox(tmp_path, monkeypatch):
    """The confirmation must be consumed, not left ticked for the next batch."""
    monkeypatch.setattr(dmf, "_ui_tab_mounted", False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    first = model_dir / "first.ckpt"
    second = model_dir / "second.ckpt"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    monkeypatch.setattr(dmf, "MODEL_DIRS", [str(model_dir)])

    tabs = dmf.on_ui_tabs()
    functions = {fn.__name__: fn for fn in _callback_functions(tabs[0][0])}
    snapshot = list(functions["do_scan"](False, progress=lambda *args, **kwargs: None))[-1][2]

    result = functions["do_delete"]([str(second)], snapshot, True)

    assert not second.exists()
    # The marker differs between Gradio versions ("update" vs "generic_update"),
    # the payload does not.
    assert result[4]["value"] is False
