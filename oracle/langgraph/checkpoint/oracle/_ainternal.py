"""Shared async utility functions for the Oracle checkpoint & storage classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Union

from oracledb import AsyncConnection

AsyncConnectionFactory = Callable[[], Awaitable[AsyncConnection]]
Conn = Union[AsyncConnection, AsyncConnectionFactory]


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[AsyncConnection]:
    if isinstance(conn, AsyncConnection):
        yield conn
    elif callable(conn):
        async with await conn() as connection:
            yield connection
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
