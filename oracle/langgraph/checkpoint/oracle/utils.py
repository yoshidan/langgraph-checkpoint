from typing import Optional

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def decode_pending_sends(
    src: Optional[str], decoder: JsonPlusSerializer
) -> list[tuple[str, bytes]]:
    """Decode the pending sends from a JSON string."""
    if not src:
        return []
    return [
        (v[0], bytes.fromhex(v[1]) if len(v) == 2 else b"")
        for v in decoder.loads(src.encode())
    ]


def decode_channel_values(
    src: Optional[str], decoder: JsonPlusSerializer
) -> list[tuple[str, str, bytes]]:
    """Decode the pending sends from a JSON string."""
    if not src:
        return []
    ret = []
    for channel, data_type, *data in decoder.loads(src.encode()):
        ret.append(
            (
                channel,
                data_type,
                b"" if data_type == "empty" else bytes.fromhex(data[0]),
            )
        )
    return ret


def decode_pending_writes(
    src: Optional[str], decoder: JsonPlusSerializer
) -> list[tuple[str, str, str, bytes]]:
    """Decode the pending sends from a JSON string."""
    if not src:
        return []
    return [
        (v[0], v[1], v[2], bytes.fromhex(v[3]) if len(v) == 4 else b"")
        for v in decoder.loads(src.encode())
    ]


def decode_channel_versions(
    src: Optional[str], decoder: JsonPlusSerializer
) -> list[tuple[str, str]]:
    """Decode the pending sends from a JSON string."""
    if not src:
        return []
    return [(v[0], v[1]) for v in decoder.loads(src.encode())]
