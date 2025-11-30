from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest

from scripts.duplicate_model_finder import (
    ALLOWED_EXTENSIONS,
    collect_hashes,
    compute_hash,
    delete_files,
    find_duplicates,
    format_duplicates_for_display,
    is_model_file,
    iter_model_files,
)


def test_is_model_file_respects_extensions():
    assert is_model_file("model.ckpt")
    assert is_model_file("model.safetensors")
    assert not is_model_file("notes.txt")
    assert not is_model_file("archive.zip", extensions=(".pt",))


def test_compute_hash_produces_consistent_result(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"abc" * 1024)

    first_hash = compute_hash(target)
    second_hash = compute_hash(target)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_find_duplicates_detects_matching_files(tmp_path: Path):
    model_dir = tmp_path / "models" / "Stable-diffusion"
    model_dir.mkdir(parents=True)

    a = model_dir / "a.ckpt"
    b = model_dir / "b.ckpt"
    c = model_dir / "c.pt"
    a.write_bytes(b"duplicate")
    b.write_bytes(b"duplicate")
    c.write_bytes(b"unique")

    duplicates = find_duplicates(directories=[model_dir], extensions=ALLOWED_EXTENSIONS)
    assert len(duplicates) == 1
    paths = next(iter(duplicates.values()))
    assert sorted(paths) == sorted([str(a), str(b)])


def test_collect_hashes_groups_paths_by_hash(tmp_path: Path):
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    third = tmp_path / "third.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    third.write_bytes(b"world")

    hashes = collect_hashes([first, second, third])
    assert len(hashes) == 2
    matching_group = next(paths for paths in hashes.values() if len(paths) == 2)
    assert sorted(matching_group) == sorted([str(first), str(second)])


def test_format_duplicates_for_display_outputs_lines():
    duplicates = {"hash": ["/path/a", "/path/b"]}
    text, choices = format_duplicates_for_display(duplicates)

    assert "hash" in text
    assert "/path/a" in text
    assert choices == ["/path/a", "/path/b"]


def test_delete_files_reports_failures(tmp_path: Path):
    deletable = tmp_path / "delete.pt"
    deletable.write_bytes(b"data")

    message, failed = delete_files([deletable, tmp_path / "missing.pt"])

    assert "failed to delete" in message
    assert failed == [str(tmp_path / "missing.pt")]
    assert not deletable.exists()


def test_iter_model_files_yields_supported_models(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    valid = model_dir / "model.safetensors"
    invalid = model_dir / "notes.txt"
    valid.write_bytes(b"data")
    invalid.write_text("ignore")

    files = list(iter_model_files([model_dir]))

    assert files == [str(valid)]
    assert invalid.exists()

