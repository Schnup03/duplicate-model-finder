"""Run in CI with the Gradio version used by AUTOMATIC1111.

The callback wiring asserted here is Gradio 3.x specific (``tab.fns`` holds
Block objects with ``validate_outputs``). A1111 / Forge pin Gradio 3.x, so
that is the contract worth testing; on Gradio 4+ these tests skip instead of
reporting a false failure.
"""

import pytest

from scripts import duplicate_model_finder as dmf

gr = pytest.importorskip("gradio")

if int(gr.__version__.split(".", 1)[0]) >= 4:
    pytest.skip(
        f"Gradio {gr.__version__} does not expose the Gradio 3.x callback API "
        "this module tests; CI pins gradio==3.41.2.",
        allow_module_level=True,
    )


def test_registered_gradio_scan_accepts_all_progress_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(dmf, "_ui_tab_mounted", False)
    (tmp_path / "first.ckpt").write_bytes(b"same")
    (tmp_path / "second.ckpt").write_bytes(b"same")
    tab = dmf.on_ui_tabs()[0][0]
    index = next(i for i, block in enumerate(tab.fns) if block.fn.__name__ == "do_scan")
    assert len(tab.dependencies[index]["inputs"]) == 1
    assert len(tab.dependencies[index]["outputs"]) == 4
    events = list(tab.fns[index].fn(False, progress=lambda *args, **kwargs: None))
    assert len(events) == 3
    for event in events:
        tab.validate_outputs(index, event)
    assert "Scan complete" in events[-1][3]
    assert len(events[-1][1]["choices"]) == 2


def test_registered_gradio_controls_cover_all_actions(monkeypatch):
    monkeypatch.setattr(dmf, "_ui_tab_mounted", False)
    tab = dmf.on_ui_tabs()[0][0]
    actual = {
        block.fn.__name__: len(tab.dependencies[index]["outputs"])
        for index, block in enumerate(tab.fns)
    }
    assert actual == {
        "do_scan": 4,
        "select_extra_copies": 1,
        "cancel_scan": 1,
        "do_trash": 4,
        "do_delete": 4,
        "do_empty_trash": 1,
    }
