"""Checkpoint discovery: compatibility filtering, manifests, skipped folders, bad files."""
import json
import zipfile

from marl_asteroids.checkpoints import current_obs_dim, discover, read_obs_dim


def fake_checkpoint(path, dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"observation_space": {":type:": "x", "_shape": [dim]}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps(data))
    return path


def test_reads_obs_dim_without_loading_torch(tmp_path):
    p = fake_checkpoint(tmp_path / "a.zip", 123)
    assert read_obs_dim(str(p)) == 123


def test_bad_files_are_unreadable_not_fatal(tmp_path):
    (tmp_path / "junk.zip").write_bytes(b"not a zip")
    assert read_obs_dim(str(tmp_path / "junk.zip")) is None
    assert read_obs_dim(str(tmp_path / "missing.zip")) is None
    assert discover(tmp_path) == []


def test_filters_incompatible_and_skips_archive_folders(tmp_path):
    dim = current_obs_dim()
    fake_checkpoint(tmp_path / "run" / "final_model.zip", dim)
    fake_checkpoint(tmp_path / "run" / "old.zip", 191)
    fake_checkpoint(tmp_path / "run" / "snapshots" / "snap_1k.zip", dim)
    fake_checkpoint(tmp_path / "_stale" / "x.zip", dim)
    fake_checkpoint(tmp_path / "legacy" / "y.zip", dim)
    names = [c.name for c in discover(tmp_path)]
    assert names == ["run/final_model"]
    with_snaps = [c.name for c in discover(tmp_path, include_snapshots=True)]
    assert "run/snapshots/snap_1k" in with_snaps
    everything = discover(tmp_path, compatible_only=False)
    assert {c.name for c in everything} == {"run/final_model", "run/old"}
    assert [c.compatible for c in everything if c.name == "run/old"] == [False]


def test_manifest_names_and_elo_ordering(tmp_path):
    dim = current_obs_dim()
    for name in ("strong", "weak", "unrated_b", "unrated_a"):
        fake_checkpoint(tmp_path / "set" / f"{name}.zip", dim)
    (tmp_path / "set" / "manifest.json").write_text(json.dumps({"files": {
        "strong.zip": {"name": "Ace", "elo": 1700, "description": "best"},
        "weak.zip": {"name": "Rookie", "elo": 1000},
    }}))
    found = discover(tmp_path)
    assert [c.name for c in found] == ["Ace", "Rookie", "set/unrated_a", "set/unrated_b"]
    assert found[0].label == "Ace (1700)" and found[0].description == "best"
    assert found[2].label == "set/unrated_a"


def test_broken_manifest_is_ignored(tmp_path):
    fake_checkpoint(tmp_path / "set" / "m.zip", current_obs_dim())
    (tmp_path / "set" / "manifest.json").write_text("{not json")
    assert [c.name for c in discover(tmp_path)] == ["set/m"]


def test_missing_root_is_empty(tmp_path):
    assert discover(tmp_path / "nope") == []


def test_shipped_phase_a_models_are_compatible():
    names = {c.name for c in discover()}
    assert {"Self-play ace", "Self-play rookie", "Bot hunter"} <= names
