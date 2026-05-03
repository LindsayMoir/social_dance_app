import sys


sys.path.insert(0, "src")

import secret_paths


def test_purge_auth_artifacts_deletes_db_row_and_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    temp_auth = tmp_path / "temp_facebook_auth.json"
    temp_auth.write_text("{}", encoding="utf-8")

    local_auth = tmp_path / "facebook_auth.json"
    local_auth.write_text("{}", encoding="utf-8")

    executed = {"deleted": False, "committed": False}

    class _FakeConn:
        def execute(self, *_args, **_kwargs):
            executed["deleted"] = True
            return None

        def commit(self):
            executed["committed"] = True

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    monkeypatch.setattr(secret_paths, "_get_db_connection", lambda: _FakeEngine())

    result = secret_paths.purge_auth_artifacts("facebook", file_path=str(temp_auth))

    assert result["db_deleted"] is True
    assert result["file_deleted"] is True
    assert result["fallback_deleted"] is True
    assert executed["deleted"] is True
    assert executed["committed"] is True
    assert not temp_auth.exists()
    assert not local_auth.exists()
