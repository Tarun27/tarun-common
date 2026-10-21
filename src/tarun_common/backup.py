"""Database backups from an app's admin page: download a dump, load one into a validation
database and browse it read-only, switch back.

Orange, SDE Prep and IRA each carried their own copy of this. What stays in each app is
what differs between them: who counts as an admin (the router's `dependencies`), how the
app's sessions follow the mode (`on_switch`), and which paths stay writable while the
app is read-only (`ReadOnlyMiddleware(allow=…)`).

The rules, shared by all of them:

- **A dump is only ever loaded into the validation database.** Restoring over live data
  is disaster recovery, done outside the app, never a button.
- **Validation mode is read-only.** The middleware refuses POST/PUT/PATCH/DELETE with a
  403 except on the paths the app allows (the backup endpoints themselves, auth).
- **Nightly snapshots load directly.** Point `snapshots_root` at a folder laid out as
  `<root>/db/<stamp>/<server>/<database>.dump` (custom-format pg_dump) and the admin page
  can list and load them without a download/upload round trip. Plain SQL dumps (what
  `/export` produces) and custom-format ones are both accepted, by path or upload.

pg_dump, pg_restore and psql run as subprocesses; passwords travel in PGPASSWORD, never
on a command line. FastAPI is imported only by `router()`. Nothing here logs a URL.
"""

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Annotated, Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

Mode = Literal["primary", "validation"]
OnSwitch = Callable[[Mode], Awaitable[None] | None]

_CUSTOM_MAGIC = b"PGDMP"
_RESET_SQL = "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"
_STAMP_FORMAT = "%Y-%m-%d_%H%M%S"
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class BackupError(RuntimeError):
    """A pg tool failed, or a dump can't be used. The message is safe to show an admin."""


# ── Connection details ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PgTarget:
    """libpq arguments and environment for one database URL."""

    dbname: str
    args: tuple[str, ...]
    env: dict[str, str]


def pg_target(url: str) -> PgTarget:
    """`postgresql[+driver]://user:pass@host:port/db?sslmode=…` → libpq args + env.

    Accepts SQLAlchemy URLs as the apps store them (`+asyncpg`, `+psycopg`, …).
    """
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.split("+", 1)[0] not in ("postgresql", "postgres"):
        raise BackupError("not a PostgreSQL URL")
    parsed = urlparse(f"postgresql://{rest}")
    dbname = unquote(parsed.path.lstrip("/"))
    if not dbname:
        raise BackupError("the database URL names no database")
    args = ["--dbname", dbname]
    if parsed.hostname:
        args += ["--host", parsed.hostname]
    if parsed.port:
        args += ["--port", str(parsed.port)]
    if parsed.username:
        args += ["--username", unquote(parsed.username)]
    env = dict(os.environ)
    if parsed.password:
        env["PGPASSWORD"] = unquote(parsed.password)
    sslmode = parse_qs(parsed.query).get("sslmode")
    if sslmode:
        env["PGSSLMODE"] = sslmode[0]
    return PgTarget(dbname=dbname, args=tuple(args), env=env)


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BackupError(f"{name} is not installed on this server")
    return path


def _run(cmd: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BackupError(f"{Path(cmd[0]).name} failed: {detail[-600:]}")


# ── Dump ──────────────────────────────────────────────────────────────────────


def dump_chunks(url: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Stream a plain-SQL pg_dump (no owners, no grants) of `url`, chunk by chunk.

    Raises BackupError after the last chunk if pg_dump failed, so a truncated download
    ends in an error rather than looking complete.
    """
    target = pg_target(url)
    cmd = [_tool("pg_dump"), "--format=plain", "--no-owner", "--no-privileges", *target.args]
    process = subprocess.Popen(cmd, env=target.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    try:
        yield from iter(lambda: process.stdout.read(chunk_size), b"")
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode(errors="replace").strip()
        process.stderr.close()
        if process.wait() != 0:
            raise BackupError(f"pg_dump failed: {stderr[-600:]}")


def dump_filename(prefix: str, now: datetime | None = None) -> str:
    return f"{prefix}-{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}.sql"


# ── Restore ───────────────────────────────────────────────────────────────────


def is_custom_format(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(len(_CUSTOM_MAGIC)) == _CUSTOM_MAGIC


def restore(target_url: str, source: str | os.PathLike[str] | bytes | IO[bytes]) -> None:
    """Replace the `public` schema of `target_url` with the dump in `source`.

    `source` is a path, the dump's bytes, or a binary file object (an upload). Plain SQL
    goes through psql, custom format through pg_restore; both stop at the first error and
    skip owners and grants, so a dump from any role loads as the target's role.
    """
    if isinstance(source, (str, os.PathLike)):
        _restore_file(target_url, Path(source))
        return
    with tempfile.NamedTemporaryFile(suffix=".dump") as tmp:
        if isinstance(source, bytes):
            tmp.write(source)
        else:
            shutil.copyfileobj(source, tmp)
        tmp.flush()
        _restore_file(target_url, Path(tmp.name))


def _restore_file(target_url: str, path: Path) -> None:
    if not path.is_file():
        raise BackupError(f"no dump at {path}")
    if path.stat().st_size == 0:
        raise BackupError("the dump is empty")
    target = pg_target(target_url)
    psql = _tool("psql")
    _run(
        [psql, "--no-psqlrc", "--set=ON_ERROR_STOP=1", *target.args, "--command", _RESET_SQL],
        target.env,
    )
    if is_custom_format(path):
        _run(
            [
                _tool("pg_restore"),
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                *target.args,
                str(path),
            ],
            target.env,
        )
    else:
        _run(
            [
                psql,
                "--no-psqlrc",
                "--quiet",
                "--set=ON_ERROR_STOP=1",
                *target.args,
                "--file",
                str(path),
            ],
            target.env,
        )


# ── Nightly snapshots ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Snapshot:
    stamp: str  # folder name, e.g. "2026-10-20_013000"
    path: Path
    bytes: int
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "stamp": self.stamp,
            "bytes": self.bytes,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


def snapshots(root: str | os.PathLike[str], dbname: str) -> list[Snapshot]:
    """Nightly dumps of `dbname` under `<root>/db/<stamp>/<server>/<dbname>.dump`, newest
    first. In-progress folders (not named by a timestamp) are ignored."""
    db_dir = Path(root).expanduser() / "db"
    if not db_dir.is_dir():
        return []
    found = []
    for folder in db_dir.iterdir():
        try:
            created = datetime.strptime(folder.name, _STAMP_FORMAT)
        except ValueError:
            continue
        for dump in sorted(folder.glob(f"*/{dbname}.dump")):
            found.append(Snapshot(folder.name, dump, dump.stat().st_size, created))
            break
    return sorted(found, key=lambda s: s.created_at, reverse=True)


# ── Validation mode ───────────────────────────────────────────────────────────


class Validation:
    """Which database the app is on, and loading a dump into the validation one.

    `on_switch(mode)` is the app's hook to point its sessions at the other database
    (sync or async). It is called with "primary" before a dump is loaded — so nothing
    is reading the validation database while its schema is replaced — and with
    "validation" once the load succeeded.
    """

    def __init__(
        self,
        primary_url: str,
        validation_url: str | None = None,
        *,
        on_switch: OnSwitch | None = None,
    ) -> None:
        self.primary_url = primary_url
        self.validation_url = validation_url or None
        self._on_switch = on_switch
        self._mode: Mode = "primary"
        self.label: str | None = None
        self.origin: str | None = None

    @property
    def configured(self) -> bool:
        return self.validation_url is not None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def read_only(self) -> bool:
        return self._mode == "validation"

    async def _switch(self, mode: Mode) -> None:
        self._mode = mode
        if self._on_switch is not None:
            result = self._on_switch(mode)
            if inspect.isawaitable(result):
                await result

    async def load(
        self, source: str | os.PathLike[str] | bytes | IO[bytes], *, label: str, origin: str
    ) -> None:
        """Load a dump into the validation database and switch to it (read-only)."""
        if self.validation_url is None:
            raise BackupError("no validation database is configured")
        await self.use_primary()
        await asyncio.to_thread(restore, self.validation_url, source)
        self.label, self.origin = label, origin
        await self._switch("validation")

    async def use_primary(self) -> None:
        self.label = self.origin = None
        await self._switch("primary")

    def status(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "read_only": self.read_only,
            "backup_configured": self.configured,
            "active_dump_label": self.label,
            "active_dump_source": self.origin,
        }


class ReadOnlyMiddleware:
    """Pure ASGI (streaming responses pass through): 403 on writes in validation mode,
    except under the `allow` path prefixes."""

    _BODY = json.dumps({"detail": "Application is in read-only validation mode"}).encode()

    def __init__(self, app: Any, validation: Validation, allow: Iterable[str] = ()) -> None:
        self.app = app
        self.validation = validation
        self.allow = tuple(allow)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            scope["type"] == "http"
            and self.validation.read_only
            and scope.get("method") in _WRITE_METHODS
            and not scope.get("path", "").startswith(self.allow)
        ):
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(self._BODY)).encode()),
            ]
            await send({"type": "http.response.start", "status": 403, "headers": headers})
            await send({"type": "http.response.body", "body": self._BODY})
            return
        await self.app(scope, receive, send)


# ── FastAPI ───────────────────────────────────────────────────────────────────


def router(
    validation: Validation,
    *,
    filename_prefix: str,
    snapshots_root: str | os.PathLike[str] | None = None,
    restore_roots: Iterable[str | os.PathLike[str]] | None = None,
    dependencies: Iterable[Any] = (),
) -> Any:
    """The admin backup endpoints as a FastAPI router; mount it under the app's admin
    prefix and guard it with the app's admin dependency:

        GET  /status            mode, read-only, what is loaded
        GET  /export            download a plain-SQL dump of the primary database
        GET  /snapshots         nightly snapshots of this database (needs snapshots_root)
        POST /restore-snapshot  {"stamp": …}  load one into validation
        POST /restore-upload    multipart `file`  load an uploaded dump into validation
        POST /restore-path      {"path": …}  load a dump on the server's disk
        POST /switch-primary    back to the live database

    `restore_roots` limits /restore-path to files under those folders (default: any
    absolute path, as the apps had it). Needs `fastapi` and `python-multipart`.
    """
    from fastapi import APIRouter, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    roots = [Path(r).expanduser().resolve() for r in restore_roots] if restore_roots else None
    api = APIRouter(tags=["backup"], dependencies=list(dependencies))
    dbname = pg_target(validation.primary_url).dbname

    class StampIn(BaseModel):
        stamp: str

    class PathIn(BaseModel):
        path: str

    async def load(source: Any, label: str, origin: str) -> dict[str, Any]:
        if not validation.configured:
            raise HTTPException(400, "No validation database is configured (BACKUP_TEST_URL)")
        try:
            await validation.load(source, label=label, origin=origin)
        except BackupError as exc:
            raise HTTPException(502, str(exc)) from exc
        return validation.status()

    @api.get("/status")
    async def status() -> dict[str, Any]:
        return {**validation.status(), "snapshots_configured": snapshots_root is not None}

    @api.get("/export")
    async def export() -> Any:
        filename = dump_filename(filename_prefix)
        return StreamingResponse(
            dump_chunks(validation.primary_url),  # sync generator: runs in the threadpool
            media_type="application/sql",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @api.get("/snapshots")
    async def list_snapshots() -> list[dict[str, Any]]:
        if snapshots_root is None:
            return []
        return [s.as_dict() for s in snapshots(snapshots_root, dbname)]

    @api.post("/restore-snapshot")
    async def restore_snapshot(body: StampIn) -> dict[str, Any]:
        if snapshots_root is None:
            raise HTTPException(400, "No snapshots folder is configured")
        match = next((s for s in snapshots(snapshots_root, dbname) if s.stamp == body.stamp), None)
        if match is None:
            raise HTTPException(404, f"No snapshot {body.stamp} of {dbname}")
        return await load(match.path, f"snapshot {match.stamp}", "snapshot")

    @api.post("/restore-upload")
    async def restore_upload(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        try:
            return await load(file.file, file.filename or "upload", "upload")
        finally:
            await file.close()

    @api.post("/restore-path")
    async def restore_path(body: PathIn) -> dict[str, Any]:
        path = Path(body.path)
        if not path.is_absolute():
            raise HTTPException(400, "path must be absolute")
        resolved = path.resolve()
        if roots is not None and not any(resolved.is_relative_to(r) for r in roots):
            raise HTTPException(400, "path is outside the allowed backup folders")
        if not resolved.is_file():
            raise HTTPException(400, f"File not found: {path}")
        return await load(resolved, resolved.name, "path")

    @api.post("/switch-primary")
    async def switch_primary() -> dict[str, Any]:
        await validation.use_primary()
        return validation.status()

    return api
