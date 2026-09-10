from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest

from scripts.duplicate_model_finder import (
    ALLOWED_EXTENSIONS,
    TRASH_DIR_NAME,
    collect_hashes,
    compute_hash,
    delete_files,
    find_duplicates,
    format_duplicates_for_display,
    is_model_file,
    iter_model_files,
    move_files_to_trash,
    permanently_delete_files,
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

    hashes = collect_hashes([first, second, third], max_workers=1)
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

    assert "Moved 1 file" in message
    assert failed == [str(tmp_path / "missing.pt")]
    assert not deletable.exists()
    assert (tmp_path / TRASH_DIR_NAME).is_dir()


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


# ---------------------------------------------------------------------------
# Cluster 1 (Performance) — size prefilter + parallel hashing
# ---------------------------------------------------------------------------


def test_collect_hashes_skips_unique_sizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Files with unique sizes are skipped before the SHA256 I/O happens."""

    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    lonely = tmp_path / "lonely.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    lonely.write_bytes(b"x")  # unique size — must be skipped

    hash_calls: list[str] = []

    def fake_compute_hash(path: str, chunk_size: int = 1 << 20) -> str:
        hash_calls.append(path)
        return compute_hash(path, chunk_size=chunk_size)

    monkeypatch.setattr("scripts.duplicate_model_finder.compute_hash", fake_compute_hash)

    hashes = collect_hashes([first, second, lonely], max_workers=1)

    assert len(hashes) == 1
    group = next(iter(hashes.values()))
    assert sorted(group) == sorted([str(first), str(second)])
    assert hash_calls == [str(first), str(second)]


def test_collect_hashes_size_prefilter_disabled_hashes_everything(tmp_path: Path):
    """With the prefilter disabled every file gets hashed."""

    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    lonely = tmp_path / "lonely.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    lonely.write_bytes(b"x")

    hashes = collect_hashes(
        [first, second, lonely],
        use_size_prefilter=False,
        max_workers=1,
    )

    assert len(hashes) == 2
    groups = sorted((len(paths), sorted(paths)) for paths in hashes.values())
    assert groups == [(1, [str(lonely)]), (2, sorted([str(first), str(second)]))]


def test_collect_hashes_parallel_matches_sequential(tmp_path: Path):
    """The threaded path returns the same grouping as the serial one."""

    files = []
    for i in range(8):
        path = tmp_path / f"file_{i}.ckpt"
        # Half the files share content "alpha", half share "beta".
        path.write_bytes(b"alpha" if i % 2 == 0 else b"beta")
        files.append(path)

    sequential = collect_hashes(files, max_workers=1)
    parallel = collect_hashes(files, max_workers=4)

    sequential_normalised = sorted(tuple(sorted(group)) for group in sequential.values())
    parallel_normalised = sorted(tuple(sorted(group)) for group in parallel.values())
    assert sequential_normalised == parallel_normalised


def test_collect_hashes_ignores_files_that_cannot_be_stated(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    missing = tmp_path / "ghost.ckpt"
    real.write_bytes(b"x")
    # `missing` is never written — must not crash the scan.
    # With the size prefilter, missing is dropped during stat; real has a
    # unique size so it is also dropped. Force the prefilter off to verify
    # the hashing step itself survives a missing file.

    with_prefilter = collect_hashes([real, missing], max_workers=1)
    without_prefilter = collect_hashes(
        [real, missing], use_size_prefilter=False, max_workers=1
    )

    assert with_prefilter == {}
    assert without_prefilter == {compute_hash(real): [str(real)]}


def test_iter_model_files_respects_extensions_parameter(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    ckpt = model_dir / "model.ckpt"
    txt = model_dir / "model.txt"
    bin_ = model_dir / "model.bin"
    ckpt.write_bytes(b"x")
    txt.write_text("y")
    bin_.write_bytes(b"z")

    files = list(iter_model_files([model_dir], extensions=(".txt", ".bin")))

    assert sorted(files) == sorted([str(txt), str(bin_)])


# ---------------------------------------------------------------------------
# Cluster 2 (Safety) — soft-delete + explicit permanent-delete confirmation
# ---------------------------------------------------------------------------


def test_move_files_to_trash_creates_sibling_trash_dir(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"data")

    message, failed = move_files_to_trash([target])

    assert failed == []
    assert "Moved 1 file" in message
    assert "recoverable manually" in message
    assert not target.exists()
    trash_dir = tmp_path / TRASH_DIR_NAME
    assert trash_dir.is_dir()
    assert len(list(trash_dir.iterdir())) == 1


def test_move_files_to_trash_handles_missing_files(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    missing = tmp_path / "ghost.ckpt"
    real.write_bytes(b"x")
    # `missing` is never written.

    message, failed = move_files_to_trash([real, missing])

    assert failed == [str(missing)]
    assert "Moved 1 file" in message
    assert not real.exists()


def test_move_files_to_trash_skips_unwriteable_files(tmp_path, monkeypatch):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    monkeypatch.setattr(
        "scripts.duplicate_model_finder.os.access", lambda *a, **kw: False
    )

    message, failed = move_files_to_trash([target])

    assert failed == [str(target)]
    assert target.exists()


def test_move_files_to_trash_handles_empty_input(tmp_path):
    message, failed = move_files_to_trash([])
    assert failed == []
    assert message == "No files moved"


def test_permanently_delete_files_requires_confirm(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = permanently_delete_files([target])  # default confirm=False

    assert "Refusing" in message
    assert "confirm=True" in message
    assert failed == [str(target)]
    assert target.exists()


def test_permanently_delete_files_with_confirm(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = permanently_delete_files([target], confirm=True)

    assert "Permanently deleted 1" in message
    assert failed == []
    assert not target.exists()


def test_permanently_delete_files_partial_failure(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    real.write_bytes(b"x")
    missing = tmp_path / "ghost.ckpt"

    message, failed = permanently_delete_files([real, missing], confirm=True)

    assert "Permanently deleted 1" in message
    assert str(missing) in failed
    assert not real.exists()


def test_delete_files_routes_to_move_files_to_trash(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = delete_files([target])

    assert failed == []
    assert not target.exists()
    assert (tmp_path / TRASH_DIR_NAME).is_dir()
    assert "Moved 1 file" in message
