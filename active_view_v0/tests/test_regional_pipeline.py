import json

from active_view_v0.regional_pipeline import _collection_is_complete


def test_complete_collection_is_recognized(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entries = []
    for index in range(3):
        frame_dir = run_dir / "frames" / f"{index:06d}"
        frame_dir.mkdir(parents=True)
        (frame_dir / "frame.json").write_text("{}")
        entries.append({"path": str(frame_dir.relative_to(run_dir))})
    (run_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item) for item in entries) + "\n"
    )
    (run_dir / "collection_status.json").write_text(
        json.dumps({"status": "complete", "keyframe_count": 3})
    )
    cfg = {"grid": {"keyframe_interval_s": 2.0}}
    assert _collection_is_complete(run_dir, cfg, 4.0)


def test_incomplete_collection_is_rejected(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.jsonl").write_text('{"path":"frames/000000"}\n')
    cfg = {"grid": {"keyframe_interval_s": 2.0}}
    assert not _collection_is_complete(run_dir, cfg, 4.0)
