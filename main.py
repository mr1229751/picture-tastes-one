"""Entry point for the make.fun bounty "the-picture-that-tastes-of-one".

The grader runs this with a stock Python 3.13 interpreter, no network and no
third-party packages, and expects exactly one file: /output/entry.png — a
1024x1024 picture of something delicious, at most 1 MiB.

The picture is rendered ahead of time (see render.py) and ships in the repo as
assets/dish.png. This script copies those bytes out, then re-verifies the
result from scratch using only the standard library: PNG signature, chunk
CRCs, IHDR dimensions, the full IDAT stream decompressed to the pixel count
implied by the header, presence of IEND, and the 1 MiB size gate. If any check
fails it deletes the bad output and exits non-zero with a specific message,
so a broken file never gets graded.
"""

from __future__ import annotations

import shutil
import struct
import sys
import zlib
from pathlib import Path

OUTPUT = Path("/output/entry.png")
SOURCE = Path(__file__).resolve().parent / "assets" / "dish.png"

MAX_BYTES = 1_048_576
SIDE = 1024
PNG_SIG = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}       # colour type -> samples/pixel


class GateError(RuntimeError):
    """The produced file would not pass the grader's gate."""


def verify_png(path: Path) -> tuple[int, int, int]:
    """Parse and fully validate a PNG. Returns (width, height, size_in_bytes)."""
    size = path.stat().st_size
    if size == 0:
        raise GateError(f"{path} is empty")
    if size > MAX_BYTES:
        raise GateError(f"{path} is {size} bytes, over the {MAX_BYTES} byte limit")

    blob = path.read_bytes()
    if not blob.startswith(PNG_SIG):
        raise GateError(f"{path} does not start with the PNG signature")

    pos = len(PNG_SIG)
    width = height = None
    idat = bytearray()
    saw_end = False

    while pos < len(blob):
        if pos + 8 > len(blob):
            raise GateError("truncated chunk header")
        length, ctype = struct.unpack(">I4s", blob[pos:pos + 8])
        body_end = pos + 8 + length
        if body_end + 4 > len(blob):
            raise GateError(f"chunk {ctype!r} runs past the end of the file")
        body = blob[pos + 8:body_end]
        (stored_crc,) = struct.unpack(">I", blob[body_end:body_end + 4])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != stored_crc:
            raise GateError(f"CRC mismatch in chunk {ctype!r}")

        if ctype == b"IHDR":
            if length != 13:
                raise GateError("IHDR is not 13 bytes")
            width, height, depth, colour, comp, filt, inter = struct.unpack(
                ">IIBBBBB", body)
            if width != SIDE or height != SIDE:
                raise GateError(f"image is {width}x{height}, expected {SIDE}x{SIDE}")
            if depth != 8 or colour not in CHANNELS:
                raise GateError(f"unsupported bit depth/colour type {depth}/{colour}")
            if comp != 0 or filt != 0 or inter != 0:
                raise GateError("compression, filter or interlace method not supported")
            nch = CHANNELS[colour]
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            saw_end = True
            break
        pos = body_end + 4

    if width is None:
        raise GateError("no IHDR chunk found")
    if not idat:
        raise GateError("no IDAT data found")
    if not saw_end:
        raise GateError("file is truncated: no IEND chunk")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise GateError(f"image data does not inflate: {exc}") from exc

    stride = 1 + width * nch
    expected = stride * height
    if len(raw) != expected:
        raise GateError(f"inflated {len(raw)} bytes, expected {expected}")
    return width, height, size


def main() -> int:
    if not SOURCE.is_file():
        print(f"error: missing asset {SOURCE}", file=sys.stderr)
        return 1

    try:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create {OUTPUT.parent}: {exc}", file=sys.stderr)
        return 1

    shutil.copyfile(SOURCE, OUTPUT)

    try:
        width, height, size = verify_png(OUTPUT)
    except (GateError, OSError) as exc:
        OUTPUT.unlink(missing_ok=True)
        print(f"error: entry gate failed — {exc}", file=sys.stderr)
        return 1

    print(f"wrote {OUTPUT} — {width}x{height}, {size} bytes, "
          f"{MAX_BYTES - size} bytes of headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
