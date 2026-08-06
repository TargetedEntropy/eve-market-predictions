"""Batched PostgreSQL upserts.

Row-by-row ``session.merge()`` issues a SELECT plus an INSERT/UPDATE per record
(365 days x N items per collection run). These helpers use
``INSERT ... ON CONFLICT DO UPDATE`` in chunks instead.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.dialects.postgresql import insert

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000


def _primary_key_columns(model) -> List[str]:
    return [column.name for column in model.__table__.primary_key.columns]


def _dedupe_by_key(rows: Sequence[Dict[str, Any]], key_columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Keep the last row for each key.

    PostgreSQL rejects an ``ON CONFLICT DO UPDATE`` statement that would touch
    the same row twice, so duplicates inside one batch have to go.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        seen[tuple(row.get(col) for col in key_columns)] = row
    return list(seen.values())


def _chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


async def bulk_upsert(
    session,
    model,
    rows: Sequence[Dict[str, Any]],
    update_columns: Optional[Sequence[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Insert rows, updating existing ones on primary-key conflict.

    Args:
        session: Active ``AsyncSession``. The caller commits.
        model: Declarative model class.
        rows: Dicts keyed by column name.
        update_columns: Columns to overwrite on conflict. Defaults to every
            non-primary-key column present in the payload.
        chunk_size: Rows per statement.

    Returns:
        Number of rows sent (after de-duplication).
    """
    if not rows:
        return 0

    key_columns = _primary_key_columns(model)
    deduped = _dedupe_by_key(rows, key_columns)

    if update_columns is None:
        payload_columns = {column for row in deduped for column in row}
        update_columns = [
            column.name
            for column in model.__table__.columns
            if column.name not in key_columns and column.name in payload_columns
        ]

    for chunk in _chunks(deduped, chunk_size):
        statement = insert(model).values(list(chunk))
        if update_columns:
            statement = statement.on_conflict_do_update(
                index_elements=key_columns,
                set_={column: getattr(statement.excluded, column) for column in update_columns},
            )
        else:
            statement = statement.on_conflict_do_nothing(index_elements=key_columns)
        await session.execute(statement)

    return len(deduped)


async def bulk_insert_ignore(
    session,
    model,
    rows: Sequence[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Insert rows, skipping primary-key conflicts (safe to re-run)."""
    return await bulk_upsert(session, model, rows, update_columns=[], chunk_size=chunk_size)
