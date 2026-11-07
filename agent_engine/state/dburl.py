"""Database locations shared by the API and the agent engine.

A location is either a SQLite file path / ``sqlite:///`` URL or a PostgreSQL URL. PostgreSQL locations may carry a ``schema``
query parameter (``postgresql://user:pw@host:5432/researchos?schema=engine``): the connection then uses
``search_path=<schema>`` so the API and the engine can share one server and one database yet keep their tables apart (both define
tables called ``runs`` and ``projects``). Everything else in the URL is passed to libpq unchanged. The ``schema`` parameter is
removed before connecting because libpq rejects unknown parameters."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_PG_PREFIXES = ("postgresql://", "postgres://", "postgresql+psycopg://")
_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def is_postgres(target) -> bool:
    return isinstance(target, str) and target.lower().startswith(_PG_PREFIXES)


def split_schema(url: str) -> tuple[str, str | None]:
    """(url without the ``schema`` parameter, schema or None)."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    schema = None
    rest = []
    for k, v in query:
        if k == "schema":
            schema = v or None
        else:
            rest.append((k, v))
    if schema is not None and not _SCHEMA.match(schema):
        raise ValueError(f"invalid schema name {schema!r}: use lower-case letters, digits and underscores")
    return urlunsplit(parts._replace(query=urlencode(rest))), schema


def with_schema(url: str, schema: str | None) -> str:
    """The same URL pointing at `schema` (replacing any schema already present)."""
    base, _ = split_schema(url)
    if not schema:
        return base
    if not _SCHEMA.match(schema):
        raise ValueError(f"invalid schema name {schema!r}: use lower-case letters, digits and underscores")
    parts = urlsplit(base)
    query = parse_qsl(parts.query, keep_blank_values=True) + [("schema", schema)]
    return urlunsplit(parts._replace(query=urlencode(query)))


def psycopg_dsn(url: str) -> tuple[str, str | None]:
    """(libpq connection URI, schema) for psycopg."""
    base, schema = split_schema(url)
    parts = urlsplit(base)
    if parts.scheme in ("postgresql+psycopg", "postgres"):
        base = urlunsplit(parts._replace(scheme="postgresql"))
    return base, schema


def sqlalchemy_url(url: str) -> tuple[str, str | None]:
    """(SQLAlchemy URL using the psycopg 3 driver, schema)."""
    base, schema = split_schema(url)
    parts = urlsplit(base)
    return urlunsplit(parts._replace(scheme="postgresql+psycopg")), schema


def connect_options(schema: str | None, application: str) -> dict:
    """libpq keyword arguments shared by every connection the application opens."""
    out: dict = {"application_name": application, "connect_timeout": 10}
    if schema:
        out["options"] = f"-c search_path={schema}"
    return out


def redact(url: str) -> str:
    """The URL with the password replaced, for logs and error messages."""
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        return urlunsplit(parts._replace(netloc=netloc))
    return url
