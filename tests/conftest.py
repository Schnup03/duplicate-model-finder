import pytest


@pytest.fixture(autouse=True)
def model_roots_are_isolated(tmp_path, monkeypatch):
    """Destructive tests may operate only inside their own temporary root."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
