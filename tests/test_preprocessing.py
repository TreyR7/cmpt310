from smart_livestock_gate.data.preprocessing import image_paths


def test_image_paths_returns_supported_files_in_order(tmp_path):
    (tmp_path / "b.png").touch()
    (tmp_path / "a.jpg").touch()
    (tmp_path / "notes.txt").touch()

    assert [path.name for path in image_paths(tmp_path)] == ["a.jpg", "b.png"]


def test_image_paths_rejects_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    try:
        image_paths(missing)
    except ValueError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("image_paths should reject a missing directory")

