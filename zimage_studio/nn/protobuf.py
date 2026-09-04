"""A minimal protobuf wire-format reader.

ONNX files are protobuf messages.  The official ``onnx`` package has no 32-bit
Windows wheel, and it would pull in ``protobuf`` as well, so the few dozen lines
needed to walk the wire format are implemented here instead.

Only decoding is supported, and only the four wire types ONNX actually uses.
Byte payloads are returned as :class:`memoryview` slices of the caller's buffer,
so reading a 200 MB model does not copy 200 MB of weights.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

WIRE_VARINT = 0
WIRE_64BIT = 1
WIRE_LENGTH = 2
WIRE_32BIT = 5


class ProtobufError(ValueError):
    """Raised when a message cannot be decoded."""


def read_varint(data: memoryview, offset: int) -> Tuple[int, int]:
    """Return ``(value, new_offset)`` for the varint at ``offset``."""
    result = 0
    shift = 0
    size = len(data)
    while True:
        if offset >= size:
            raise ProtobufError("truncated varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7
        if shift > 70:
            raise ProtobufError("varint too long")


def iter_fields(data: memoryview) -> Iterator[Tuple[int, int, object]]:
    """Yield ``(field_number, wire_type, value)`` for every field in a message.

    ``value`` is an int for varint/fixed fields and a memoryview for
    length-delimited ones.
    """
    offset = 0
    size = len(data)
    while offset < size:
        key, offset = read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == WIRE_VARINT:
            value, offset = read_varint(data, offset)
        elif wire_type == WIRE_LENGTH:
            length, offset = read_varint(data, offset)
            end = offset + length
            if end > size:
                raise ProtobufError("truncated length-delimited field")
            value = data[offset:end]
            offset = end
        elif wire_type == WIRE_64BIT:
            value = int.from_bytes(data[offset : offset + 8], "little")
            offset += 8
        elif wire_type == WIRE_32BIT:
            value = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
        else:
            raise ProtobufError("unsupported wire type %d" % wire_type)
        yield field_number, wire_type, value


def parse_message(data: memoryview) -> Dict[int, List[object]]:
    """Group every field of a message by field number."""
    fields: Dict[int, List[object]] = {}
    for number, _wire, value in iter_fields(data):
        fields.setdefault(number, []).append(value)
    return fields


def unpack_varints(data: memoryview) -> List[int]:
    """Decode a packed repeated varint field."""
    values = []
    offset = 0
    while offset < len(data):
        value, offset = read_varint(data, offset)
        values.append(value)
    return values


def as_signed(value: int, bits: int = 64) -> int:
    """Reinterpret an unsigned varint as a two's complement signed integer."""
    limit = 1 << (bits - 1)
    return value - (1 << bits) if value >= limit else value


def as_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def first(fields: Dict[int, List[object]], number: int, default=None):
    values = fields.get(number)
    return values[0] if values else default
