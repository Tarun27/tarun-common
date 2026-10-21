"""tarun_common.backup: URL parsing, snapshots, the middleware and router with restore
faked, and real dump/restore round trips against a local Postgres (skipped without one)."""

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tarun_common import backup
from tarun_common.backup import BackupError, ReadOnlyMiddleware, Validation

PRIMARY = "postgresql+asyncpg://app:p%40ss@db.local:6543/appdb?sslmode=require"
VALIDATION = "postgresql+asyncpg://app:p%40ss@db.local:6543/appdb_validation"


# ── URLs ──────────────────────────────────────────────────────────────────────


def test_pg_target_reads_sqlalchemy_urls():
    t = backup.pg_target(PRIMARY)
    assert t.dbname == "appdb"
    assert t.args == (
        "--dbname",
        "appdb",
        "--host",
        "db.local",
        "--port",
        "6543",
        "--username",
        "app",
    )
    assert t.env["PGPASSWORD"] == "p@ss"
    assert t.env["PGSSLMODE"] == "require"
    assert "p@ss" not in " ".join(t.args)


def test_pg_target_minimal_and_invalid():
    t = backup.pg_target("postgresql://localhost/notes")
    assert t.args == ("--dbname", "notes", "--host", "localhost")
    with pytest.raises(BackupError):
        backup.pg_target("mysql://x/y")
    with pytest.raises(BackupError):
        backup.pg_target("postgresql://localhost/")


# ── Snapshots ─────────────────────────────────────────────────────────────────


def test_snapshots_newest_first_ignoring_partials(tmp_path):
    for stamp in ("2026-10-18_023000", "2026-10-20_013000", ".partial-2026-10-21_023000"):
        dump = tmp_path / "db" / stamp / "shared" / "appdb.dump"
        dump.parent.mkdir(parents=True)
        dump.write_bytes(b"PGDMP...")
    (tmp_path / "db" / "2026-10-19_023000" / "shared").mkdir(parents=True)  # no appdb
    found = backup.snapshots(tmp_path, "appdb")
    assert [s.stamp for s in found] == ["2026-10-20_013000", "2026-10-18_023000"]
    assert found[0].bytes == 8
    assert backup.snapshots(tmp_path / "missing", "appdb") == []


# ── Validation + middleware + router (restore faked) ──────────────────────────


@pytest.fixture
def fake_pg(monkeypatch):
    calls = []

    def fake_restore(url, source):
        data = (
            Path(source).read_bytes()
            if isinstance(source, (str, os.PathLike))
            else (source if isinstance(source, bytes) else source.read())
        )
        if data == b"broken":
            raise BackupError("psql failed: syntax error")
        calls.append((url, data))

    monkeypatch.setattr(backup, "restore", fake_restore)
    monkeypatch.setattr(backup, "dump_chunks", lambda url: iter([b"-- dump of ", b"appdb\n"]))
    return calls


@pytest.fixture
def app_setup(tmp_path, fake_pg):
    switches = []

    async def on_switch(mode):
        switches.append(mode)

    validation = Validation(PRIMARY, VALIDATION, on_switch=on_switch)
    snap = tmp_path / "backups" / "db" / "2026-10-20_013000" / "shared" / "appdb.dump"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"PGDMP snapshot")
    app = FastAPI()
    app.include_router(
        backup.router(
            validation,
            filename_prefix="appdb",
            snapshots_root=tmp_path / "backups",
            restore_roots=[tmp_path / "backups"],
        ),
        prefix="/api/admin/backup",
    )

    @app.post("/api/notes")
    async def create_note():
        return {"created": True}

    app.add_middleware(ReadOnlyMiddleware, validation=validation, allow=("/api/admin/backup/",))
    return TestClient(app), validation, switches, fake_pg, tmp_path


def test_export_streams_a_named_dump(app_setup):
    client, *_ = app_setup
    r = client.get("/api/admin/backup/export")
    assert r.status_code == 200
    assert r.content == b"-- dump of appdb\n"
    assert 'filename="appdb-' in r.headers["content-disposition"]


def test_snapshot_load_switches_to_read_only_and_back(app_setup):
    client, _, switches, calls, _ = app_setup
    assert client.get("/api/admin/backup/snapshots").json()[0]["stamp"] == "2026-10-20_013000"
    r = client.post("/api/admin/backup/restore-snapshot", json={"stamp": "2026-10-20_013000"})
    assert r.status_code == 200, r.text
    assert r.json()["read_only"] is True
    assert r.json()["active_dump_label"] == "snapshot 2026-10-20_013000"
    assert calls == [(VALIDATION, b"PGDMP snapshot")]  # never the primary
    assert switches == ["primary", "validation"]

    assert client.post("/api/notes").status_code == 403  # read-only
    assert client.post("/api/admin/backup/switch-primary").json()["mode"] == "primary"
    assert client.post("/api/notes").status_code == 200
    assert (
        client.post(
            "/api/admin/backup/restore-snapshot", json={"stamp": "1999-01-01_000000"}
        ).status_code
        == 404
    )


def test_upload_and_failed_load_stays_on_primary(app_setup):
    client, validation, *_ = app_setup
    r = client.post(
        "/api/admin/backup/restore-upload",
        files={"file": ("mine.sql", b"CREATE TABLE t();", "application/sql")},
    )
    assert r.status_code == 200 and r.json()["active_dump_label"] == "mine.sql"
    r = client.post(
        "/api/admin/backup/restore-upload",
        files={"file": ("bad.sql", b"broken", "application/sql")},
    )
    assert r.status_code == 502 and "syntax error" in r.json()["detail"]
    assert validation.mode == "primary"


def test_restore_path_is_confined_to_restore_roots(app_setup, tmp_path):
    client, *_ = app_setup
    outside = tmp_path / "elsewhere.sql"
    outside.write_text("x")
    for path, code in (
        ("relative.sql", 400),
        (str(outside), 400),
        (str(tmp_path / "backups/../elsewhere.sql"), 400),  # no escaping via ..
        (str(tmp_path / "backups/db/2026-10-20_013000/shared/appdb.dump"), 200),
    ):
        assert (
            client.post("/api/admin/backup/restore-path", json={"path": path}).status_code == code
        ), path


def test_not_configured_is_a_clear_400(fake_pg):
    app = FastAPI()
    app.include_router(backup.router(Validation(PRIMARY), filename_prefix="x"))
    client = TestClient(app)
    assert client.get("/status").json()["backup_configured"] is False
    r = client.post("/restore-upload", files={"file": ("a.sql", b"x")})
    assert r.status_code == 400


# ── Real Postgres ─────────────────────────────────────────────────────────────


def _psql(db: str, sql: str, user: str | None = None) -> str:
    cmd = ["psql", "-h", "localhost", "-d", db, "-Atqc", sql]
    if user:
        cmd[1:1] = ["-U", user]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _pg_up() -> bool:
    return (
        bool(shutil.which("pg_isready"))
        and subprocess.run(["pg_isready", "-q", "-h", "localhost"]).returncode == 0
    )


@pytest.mark.skipif(not _pg_up(), reason="no local Postgres")
async def test_round_trip_plain_and_custom_as_a_non_superuser(tmp_path):
    tag = uuid.uuid4().hex[:8]
    role, primary, target = f"tc_role_{tag}", f"tc_primary_{tag}", f"tc_validation_{tag}"
    _psql("postgres", f"CREATE ROLE {role} LOGIN PASSWORD 'pw'")
    try:
        _psql("postgres", f"CREATE DATABASE {primary}")  # owned by the superuser
        _psql("postgres", f"CREATE DATABASE {target} OWNER {role}")
        _psql(
            primary,
            "CREATE TABLE notes (id serial PRIMARY KEY, body text);"
            "INSERT INTO notes (body) SELECT 'n' || g FROM generate_series(1, 30) g",
        )
        primary_url = f"postgresql+psycopg://localhost/{primary}"
        target_url = f"postgresql+asyncpg://{role}:pw@localhost/{target}"

        # Plain SQL from /export, loaded by a role that owns nothing in the source.
        plain = b"".join(backup.dump_chunks(primary_url))
        assert b"CREATE TABLE public.notes" in plain
        validation = Validation(primary_url, target_url)
        await validation.load(plain, label="plain", origin="upload")
        assert _psql(target, "SELECT count(*) FROM notes") == "30"
        assert _psql(target, "SELECT tableowner FROM pg_tables WHERE tablename='notes'") == role

        # Custom format, as the nightly snapshots are written, over the previous load.
        _psql(primary, "INSERT INTO notes (body) VALUES ('one more')")
        custom = tmp_path / f"{primary}.dump"
        subprocess.run(
            ["pg_dump", "-h", "localhost", "-Fc", "-f", str(custom), primary], check=True
        )
        await validation.load(custom, label="custom", origin="snapshot")
        assert _psql(target, "SELECT count(*) FROM notes") == "31"
        assert validation.read_only

        # A failing dump stops at the first error and leaves the app on primary.
        with pytest.raises(BackupError, match="psql failed"):
            await validation.load(b"CREATE TABLE broken (;", label="bad", origin="upload")
        assert validation.mode == "primary"

        with pytest.raises(BackupError, match="pg_dump failed"):
            b"".join(backup.dump_chunks(f"postgresql://localhost/tc_missing_{tag}"))
    finally:
        for db in (primary, target):
            _psql("postgres", f"DROP DATABASE IF EXISTS {db} WITH (FORCE)")
        _psql("postgres", f"DROP ROLE IF EXISTS {role}")
