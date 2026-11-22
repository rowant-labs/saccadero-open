"""TRUTH-FREE: a deterministic PNG codec that writes and accepts only IHDR, IDAT and IEND chunks (E6 plan section 8).

Why our own codec. Plan section 8 says each PNG that leaves the evaluator "may hold only IHDR, IDAT and IEND
chunks", and section 9 says images are checked by re-derivation and a chunk check. Pillow writes only those chunks
for a plain array today, but what it writes can change with its version (filter heuristics, optional chunks). This
module pins the bytes: one IHDR, one IDAT (zlib level 6), one IEND; per-row filter chosen from None, Sub and Up by
the minimum sum of absolute signed bytes (ties to the lower filter number). The same array gives the same bytes on
one host (zlib 1.2.12 measured on this Mac).

Supported: 8-bit grey (``uint8`` HxW), 16-bit grey (``uint16`` HxW, big-endian samples, used for depth), 8-bit RGB
(``uint8`` HxWx3). Non-interlaced only. The decoder reads every filter type (0-4), so it also reads PNGs written by
other tools as long as they use those three layouts; ancillary chunks are refused by ``check_clean`` and by
``decode_png(strict=True)``.

Truth status: reads and writes bytes only. It never opens a file by itself except through ``read_png`` /
``write_png_file`` on a path the caller names.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

TRUTH_FREE = True
SIGNATURE = b"\x89PNG\r\n\x1a\n"
ALLOWED_CHUNKS = ("IHDR", "IDAT", "IEND")
ZLIB_LEVEL = 6
_LAYOUTS = {  # (colour type, bit depth) -> (channels, numpy dtype)
    (0, 8): (1, np.uint8),
    (0, 16): (1, np.uint16),
    (2, 8): (3, np.uint8),
}


class PngError(ValueError):
    """Malformed, unsupported or unclean PNG bytes."""


# --------------------------------------------------------------------------------------------- encode
def _layout(array: np.ndarray) -> tuple[int, int, int]:
    """(colour type, bit depth, channels) for a supported array, else PngError."""
    a = np.asarray(array)
    if a.ndim == 2 and a.dtype == np.uint8:
        return 0, 8, 1
    if a.ndim == 2 and a.dtype == np.uint16:
        return 0, 16, 1
    if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
        return 2, 8, 3
    raise PngError(f"unsupported array: shape {a.shape}, dtype {a.dtype} (uint8 HxW, uint16 HxW, uint8 HxWx3)")


def _row_bytes(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scanline bytes, shape (H, W * bytes per pixel), uint8 (16-bit samples big-endian)."""
    a = np.ascontiguousarray(array)
    h = a.shape[0]
    if bit_depth == 16:
        a = a.astype(">u2")
    return np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(h, -1)


def _filtered(rows: np.ndarray, bpp: int) -> bytes:
    """Per-row filter None/Sub/Up by minimum sum of |signed byte|; returns the filtered stream (filter byte + row)."""
    r = rows.astype(np.int16)
    left = np.zeros_like(r)
    left[:, bpp:] = r[:, :-bpp]
    up = np.zeros_like(r)
    up[1:] = r[:-1]
    cands = np.stack([r, (r - left) % 256, (r - up) % 256])  # (3, H, n)
    signed = np.where(cands > 127, 256 - cands, cands)
    choice = np.argmin(signed.sum(axis=2), axis=0)  # first minimum: the lower filter number wins ties
    picked = cands[choice, np.arange(r.shape[0])].astype(np.uint8)
    out = np.concatenate([choice.astype(np.uint8)[:, None], picked], axis=1)
    return out.tobytes()


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(array) -> bytes:
    """PNG bytes holding only IHDR, IDAT, IEND (deterministic for a given array on one host)."""
    a = np.asarray(array)
    colour, depth, channels = _layout(a)
    h, w = a.shape[:2]
    if h < 1 or w < 1:
        raise PngError("empty image")
    bpp = max(1, channels * depth // 8)
    raw = _filtered(_row_bytes(a, depth), bpp)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, colour, 0, 0, 0)
    return SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, ZLIB_LEVEL)) + _chunk(b"IEND", b"")


# --------------------------------------------------------------------------------------------- chunks
def chunks(data: bytes) -> list[tuple[str, bytes]]:
    """Every chunk as (type, payload); checks the signature, lengths, CRCs and that nothing follows IEND."""
    if not isinstance(data, (bytes, bytearray)) or data[:8] != SIGNATURE:
        raise PngError("not a PNG (bad signature)")
    out, pos, n = [], 8, len(data)
    while True:
        if pos + 12 > n:
            raise PngError(f"truncated chunk header at byte {pos}")
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = bytes(data[pos + 4:pos + 8])
        end = pos + 12 + length
        if end > n:
            raise PngError(f"truncated chunk {kind!r} at byte {pos}")
        payload = bytes(data[pos + 8:pos + 8 + length])
        (crc,) = struct.unpack(">I", data[pos + 8 + length:end])
        if crc != (zlib.crc32(kind + payload) & 0xFFFFFFFF):
            raise PngError(f"CRC mismatch in chunk {kind!r}")
        try:
            name = kind.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PngError(f"non-ASCII chunk type {kind!r}") from exc
        out.append((name, payload))
        pos = end
        if name == "IEND":
            break
    if pos != n:
        raise PngError(f"{n - pos} bytes after IEND")
    return out


def chunk_types(data: bytes) -> list[str]:
    return [name for name, _ in chunks(data)]


def check_clean(data: bytes) -> dict:
    """Refuse anything but IHDR, one or more consecutive IDAT, IEND; returns the IHDR fields."""
    found = chunks(data)
    names = [name for name, _ in found]
    bad = sorted({name for name in names if name not in ALLOWED_CHUNKS})
    if bad:
        raise PngError(f"chunks outside {ALLOWED_CHUNKS}: {bad}")
    if names[0] != "IHDR" or names[-1] != "IEND" or names.count("IHDR") != 1 or names.count("IEND") != 1:
        raise PngError(f"chunk order {names}")
    idat = [i for i, name in enumerate(names) if name == "IDAT"]
    if not idat or idat != list(range(idat[0], idat[0] + len(idat))):
        raise PngError(f"IDAT chunks missing or not consecutive: {names}")
    ihdr = found[0][1]
    if len(ihdr) != 13:
        raise PngError("IHDR length != 13")
    w, h, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (colour, depth) not in _LAYOUTS or comp != 0 or filt != 0 or interlace != 0:
        raise PngError(f"unsupported IHDR: colour {colour}, depth {depth}, compression {comp}, filter {filt}, "
                       f"interlace {interlace}")
    if found[-1][1]:
        raise PngError("IEND carries data")
    return {"width": w, "height": h, "bit_depth": depth, "colour_type": colour}


# --------------------------------------------------------------------------------------------- decode
def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size != h * (stride + 1):
        raise PngError(f"decompressed size {buf.size} != {h} x ({stride} + 1)")
    rows = buf.reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        ftype, line = int(rows[y, 0]), rows[y, 1:].astype(np.int32)
        if ftype == 0:
            rec = line
        elif ftype == 1:  # Sub: running sum per byte position modulo the pixel width
            rec = np.cumsum(line.reshape(-1, bpp), axis=0).reshape(-1) % 256
        elif ftype == 2:  # Up
            rec = (line + prior) % 256
        elif ftype in (3, 4):  # Average / Paeth: sequential (only met in PNGs written by other tools)
            rec = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                a = int(rec[x - bpp]) if x >= bpp else 0
                b = int(prior[x])
                if ftype == 3:
                    pred = (a + b) // 2
                else:
                    c = int(prior[x - bpp]) if x >= bpp else 0
                    pred = _paeth(a, b, c)
                rec[x] = (int(line[x]) + pred) % 256
        else:
            raise PngError(f"unknown filter type {ftype} in row {y}")
        out[y] = rec.astype(np.uint8)
        prior = rec.astype(np.int32)
    return out


def decode_png(data: bytes, *, strict: bool = True) -> np.ndarray:
    """Array from PNG bytes (uint8 HxW, uint16 HxW or uint8 HxWx3). ``strict`` also applies ``check_clean``."""
    if strict:
        check_clean(data)
    found = chunks(data)
    ihdr = dict(found)["IHDR"]
    w, h, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (colour, depth) not in _LAYOUTS or interlace != 0:
        raise PngError(f"unsupported PNG layout: colour {colour}, depth {depth}, interlace {interlace}")
    channels, dtype = _LAYOUTS[(colour, depth)]
    stride = w * channels * depth // 8
    raw = zlib.decompress(b"".join(payload for name, payload in found if name == "IDAT"))
    rows = _unfilter(raw, h, stride, max(1, channels * depth // 8))
    if depth == 16:
        arr = np.frombuffer(rows.tobytes(), dtype=">u2").astype(np.uint16).reshape(h, w)
    else:
        arr = rows.reshape(h, w, channels) if channels > 1 else rows.reshape(h, w)
    return np.ascontiguousarray(arr.astype(dtype))


# --------------------------------------------------------------------------------------------- files
def read_png(path, *, strict: bool = True) -> np.ndarray:
    return decode_png(Path(path).read_bytes(), strict=strict)


def write_png_file(path, array) -> bytes:
    """Write ``encode_png(array)`` to ``path`` (refuses to overwrite); returns the bytes written."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists (never overwritten)")
    data = encode_png(array)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data
