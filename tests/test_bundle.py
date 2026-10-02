"""Bundle-/Companion-Erkennung und letzter-Kopien-Schutz der Python-API."""

import os

import pytest

from scripts import duplicate_model_finder as dmf


@pytest.fixture()
def model_dir(tmp_path):
    d = tmp_path / "Stable-diffusion"
    d.mkdir()
    return d


def _touch(path, content=b"x"):
    path.write_bytes(content)
    return path


class TestDetectBundleMembers:
    def test_nonexistent_returns_empty(self, tmp_path):
        assert dmf.detect_bundle_members(str(tmp_path / "missing.safetensors")) == []

    def test_directory_is_not_a_model(self, model_dir):
        d = model_dir / "folder.safetensors"
        d.mkdir()
        _touch(d / "folder.safetensors.json")
        assert dmf.detect_bundle_members(str(d)) == []

    def test_no_companions(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 10)
        assert dmf.detect_bundle_members(str(m)) == []

    def test_basic_companions(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        _touch(model_dir / "my_model.civitai.info", b"{...}")
        _touch(model_dir / "my_model.json", b"{}")
        _touch(model_dir / "my_model.png", b"\x89PNG")
        found = [os.path.basename(p) for p in dmf.detect_bundle_members(str(m))]
        assert "my_model.civitai.info" in found
        assert "my_model.json" in found
        assert "my_model.png" in found
        assert "my_model.safetensors" not in found
        assert len(found) == 3

    def test_double_ending_safetensors_json(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        _touch(model_dir / "my_model.safetensors.json", b"{}")
        found = [os.path.basename(p) for p in dmf.detect_bundle_members(str(m))]
        assert found == ["my_model.safetensors.json"]

    def test_random_files_ignored(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        _touch(model_dir / "my_model.txt", b"notes")
        _touch(model_dir / "my_model.ckpt", b"other model")
        assert dmf.detect_bundle_members(str(m)) == []

    def test_other_models_not_included(self, model_dir):
        m = _touch(model_dir / "model_a.safetensors", b"a" * 100)
        _touch(model_dir / "model_b.png", b"\x89PNG")
        assert dmf.detect_bundle_members(str(m)) == []

    def test_subdirectory_not_included(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        (model_dir / "my_model.json").mkdir()
        assert dmf.detect_bundle_members(str(m)) == []

    def test_case_insensitive_extension(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        _touch(model_dir / "my_model.CIVITAI.INFO", b"{...}")
        found = [os.path.basename(p) for p in dmf.detect_bundle_members(str(m))]
        assert found == ["my_model.CIVITAI.INFO"]

    def test_symlinked_companion_is_skipped(self, model_dir, tmp_path):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        outside = _touch(tmp_path / "secret.json", b"{}")
        os.symlink(outside, model_dir / "my_model.json")
        assert dmf.detect_bundle_members(str(m)) == []

    def test_shared_stem_chain_is_reported_under_both_models(self, model_dir):
        # `foo` is a stem of both models, so `foo.bar.json` genuinely cannot be
        # attributed to one of them from the file names alone. Documented
        # behaviour, not a bug: the listing is read-only and never deletable.
        short = _touch(model_dir / "foo.ckpt", b"a")
        long = _touch(model_dir / "foo.bar.safetensors", b"b")
        _touch(model_dir / "foo.bar.json", b"{}")
        _touch(model_dir / "foo.json", b"{}")
        from_short = {os.path.basename(p) for p in dmf.detect_bundle_members(str(short))}
        from_long = {os.path.basename(p) for p in dmf.detect_bundle_members(str(long))}
        assert from_short == {"foo.json", "foo.bar.json"}
        assert from_long == {"foo.json", "foo.bar.json"}

    def test_directory_listing_is_read_once_per_folder(self, model_dir, monkeypatch):
        # Guards the quadratic-scan regression: the display path used to call
        # os.listdir once per model instead of once per directory.
        for index in range(20):
            _touch(model_dir / f"m{index}.ckpt", b"same")
        groups = {"h1": [str(model_dir / f"m{index}.ckpt") for index in range(20)]}
        calls = []
        real_listdir = os.listdir

        def counting_listdir(path):
            calls.append(path)
            return real_listdir(path)

        monkeypatch.setattr(dmf.os, "listdir", counting_listdir)
        dmf.format_duplicates_for_display(groups)
        assert calls.count(str(model_dir)) <= 1

    def test_output_is_sorted(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        for name in ("my_model.yaml", "my_model.civitai.info", "my_model.json"):
            _touch(model_dir / name, b"x")
        found = dmf.detect_bundle_members(str(m))
        assert found == sorted(found)


class TestTotalBundleSize:
    def test_missing_returns_zero(self, tmp_path):
        assert dmf.total_bundle_size(str(tmp_path / "missing.safetensors")) == 0

    def test_only_model(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        assert dmf.total_bundle_size(str(m)) == 100

    def test_model_plus_companions(self, model_dir):
        m = _touch(model_dir / "my_model.safetensors", b"a" * 100)
        _touch(model_dir / "my_model.json", b"b" * 40)
        _touch(model_dir / "my_model.civitai.info", b"c" * 7)
        assert dmf.total_bundle_size(str(m)) == 147


class TestCompanionsAreNeverDeletable:
    def test_companions_stay_out_of_the_choices(self, model_dir):
        first = _touch(model_dir / "first.ckpt", b"same")
        second = _touch(model_dir / "second.ckpt", b"same")
        _touch(model_dir / "second.json", b"{}")
        _touch(model_dir / "first.json", b"{}")
        groups = dmf.find_duplicates([str(model_dir)], max_workers=1)
        text, choices = dmf.format_duplicates_for_display(groups)
        assert set(choices) == {str(first), str(second)}
        assert "second.json" not in choices
        assert "first.json" not in choices
        assert "companions (1):" in text

    def test_companion_survives_trashing_its_model(self, model_dir):
        model = _touch(model_dir / "only.ckpt", b"data")
        companion = _touch(model_dir / "only.json", b"{}")
        status, failed = dmf.move_files_to_trash([str(model)], allowed_roots=[str(model_dir)])
        assert failed == []
        assert "Moved 1" in status
        assert not os.path.exists(model)
        assert companion.exists()


class TestLastCopyProtection:
    def test_api_refuses_to_remove_every_copy(self, tmp_path):
        first = _touch(tmp_path / "a.ckpt", b"same")
        second = _touch(tmp_path / "b.ckpt", b"same")
        groups = dmf.find_duplicates([str(tmp_path)], max_workers=1)
        status, failed = dmf.move_files_to_trash(
            [str(first), str(second)], allowed_roots=[str(tmp_path)], groups=groups
        )
        assert "at least one copy" in status
        assert set(failed) == {str(first), str(second)}
        assert first.exists() and second.exists()

    def test_api_permanent_delete_also_protects(self, tmp_path):
        first = _touch(tmp_path / "a.ckpt", b"same")
        second = _touch(tmp_path / "b.ckpt", b"same")
        groups = dmf.find_duplicates([str(tmp_path)], max_workers=1)
        status, _ = dmf.permanently_delete_files(
            [str(first), str(second)], confirm=True, allowed_roots=[str(tmp_path)], groups=groups
        )
        assert "at least one copy" in status
        assert first.exists() and second.exists()

    def test_extra_copy_is_still_removable(self, tmp_path):
        first = _touch(tmp_path / "a.ckpt", b"same")
        second = _touch(tmp_path / "b.ckpt", b"same")
        groups = dmf.find_duplicates([str(tmp_path)], max_workers=1)
        status, failed = dmf.move_files_to_trash(
            [str(second)], allowed_roots=[str(tmp_path)], groups=groups
        )
        assert failed == []
        assert "Moved 1" in status
        assert first.exists() and not second.exists()

    def test_unrelated_files_are_not_blocked(self, tmp_path):
        first = _touch(tmp_path / "a.ckpt", b"same")
        second = _touch(tmp_path / "b.ckpt", b"same")
        other = _touch(tmp_path / "solo.ckpt", b"unique")
        groups = dmf.find_duplicates([str(tmp_path)], max_workers=1)
        status, failed = dmf.move_files_to_trash(
            [str(first), str(second), str(other)],
            allowed_roots=[str(tmp_path)],
            groups=groups,
        )
        assert "at least one copy" in status
        assert first.exists() and second.exists() and other.exists()

    def test_without_groups_behaviour_is_unchanged(self, tmp_path):
        # Opt-out stays possible for callers that manage their own bookkeeping.
        only = _touch(tmp_path / "solo.ckpt", b"unique")
        status, failed = dmf.move_files_to_trash([str(only)], allowed_roots=[str(tmp_path)])
        assert failed == []
        assert "Moved 1" in status
        assert not only.exists()

    def test_group_whose_files_are_all_gone_does_not_block(self, tmp_path):
        present = _touch(tmp_path / "a.ckpt", b"same")
        groups = {"deadbeef": [str(tmp_path / "gone1.ckpt"), str(tmp_path / "gone2.ckpt")]}
        status, failed = dmf.move_files_to_trash(
            [str(present)], allowed_roots=[str(tmp_path)], groups=groups
        )
        assert failed == []
        assert "Moved 1" in status
        assert not present.exists()

    def test_last_surviving_copy_is_protected_even_if_siblings_vanished(self, tmp_path):
        # a.ckpt is the only file left of a group whose other members are gone.
        # Removing it would destroy the model's last remaining copy.
        present = _touch(tmp_path / "a.ckpt", b"same")
        groups = {"deadbeef": [str(present), str(tmp_path / "gone.ckpt")]}
        status, failed = dmf.move_files_to_trash(
            [str(present)], allowed_roots=[str(tmp_path)], groups=groups
        )
        assert "at least one copy" in status
        assert present.exists()
