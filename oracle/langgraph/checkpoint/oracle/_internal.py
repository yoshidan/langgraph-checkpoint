"""Shared utility functions for the Postgres checkpoint & storage classes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Union

from oracledb import Connection

ConnectionFactory = Callable[[], Connection]
Conn = Union[Connection, ConnectionFactory]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Connection]:
    if isinstance(conn, Connection):
        yield conn
    elif callable(conn):
        with conn() as c:
            yield c
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
