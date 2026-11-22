"""M2 bridge between the venvs (plan venv_bridge): AF_UNIX stream socket, JSON header plus raw array bytes.

M2 (perceived poses, grounded instruction): perception in the loop. Standard library and numpy only; it runs in both
venvs (venv-libero's numpy 1.26 and the Monty venv's numpy 2.4: raw C-order buffers, never pickle).

Wire format of one message: a 4-byte big-endian length L, then L bytes of UTF-8 JSON (the header), then the raw
buffers the header lists in ``arrays`` ([{name, dtype, shape, nbytes}], in order), read with exact byte counts and
rebuilt with ``np.frombuffer`` (checked against the header). An array in the payload is replaced in the JSON by
``{"__array__": name}``.

The client half (``Client``) is what the runner's workers use; the server half (``serve``) is the perception
server's loop (one connection at a time, requests in arrival order). Nothing is written to any file.

Hello (plan venv_bridge.requests; verifier minor M9): with ``require_hello`` the server answers nothing but
``hello`` and ``shutdown`` on a connection until a ``hello`` on it has succeeded; the handler refuses a hello whose
protocol, venv versions (``EXPECTED_VERSIONS``), shared code sha256 (``SHARED_CODE``), library manifest sha256 or
frozen config sha256 differ from the server's own, and the client refuses a reply that differs likewise. The checks
themselves live in the runner (``m2_run``) and the server (``m2_perceive``); this module only carries the constants
and the per-connection gate, so its client half stays under the policy rules.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

PROTOCOL = "rowant-m2-bridge-v2"  # v2: the hello carries versions and sha256 and is checked on both sides
EXPECTED_VERSIONS = {"venv-libero": {"python": "3.11.11", "numpy": "1.26.4"},
                     "monty": {"python": "3.11.11", "numpy": "2.4.6"}}
SHARED_CODE = ("rowant_lab/libero_env/m2_bridge.py", "rowant_lab/libero_env/m2_kin.py",
               "rowant_lab/libero_env/m2_paths.py", "rowant_lab/libero_env/m2_library.py")
MAX_HEADER = 16 * 2 ** 20
MAX_ARRAY = 64 * 2 ** 20
DTYPES = ("uint8", "uint16", "int32", "int64", "float32", "float64", "bool")


def _split(obj, arrays: list, path: str = "a"):
    if isinstance(obj, np.ndarray):
        if obj.dtype.name not in DTYPES:
            raise TypeError(f"dtype {obj.dtype} not allowed on the wire")
        name = f"{path}{len(arrays)}"
        arrays.append((name, np.ascontiguousarray(obj)))
        return {"__array__": name}
    if isinstance(obj, dict):
        return {str(k): _split(v, arrays, path) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split(v, arrays, path) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _join(obj, arrays: dict):
    if isinstance(obj, dict):
        if set(obj) == {"__array__"}:
            return arrays[obj["__array__"]]
        return {k: _join(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_join(v, arrays) for v in obj]
    return obj


def encode(msg: dict) -> list:
    arrays: list = []
    body = _split(msg, arrays)
    head = {"msg": body, "arrays": [{"name": n, "dtype": a.dtype.name, "shape": list(a.shape), "nbytes": int(a.nbytes)}
                                    for n, a in arrays]}
    hb = json.dumps(head, allow_nan=False).encode("utf-8")
    return [struct.pack(">I", len(hb)), hb] + [a.tobytes(order="C") for _, a in arrays]


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return bytes(buf)


def send(sock, msg: dict) -> int:
    parts = encode(msg)
    n = 0
    for p in parts:
        sock.sendall(p)
        n += len(p)
    return n


def recv(sock) -> tuple[dict, int]:
    (L,) = struct.unpack(">I", _recv_exact(sock, 4))
    if L > MAX_HEADER:
        raise ValueError(f"header too large ({L})")
    head = json.loads(_recv_exact(sock, L).decode("utf-8"))
    arrays = {}
    total = 4 + L
    for spec in head["arrays"]:
        if spec["dtype"] not in DTYPES or spec["nbytes"] > MAX_ARRAY:
            raise ValueError(f"array {spec['name']} refused ({spec['dtype']}, {spec['nbytes']} bytes)")
        raw = _recv_exact(sock, int(spec["nbytes"]))
        a = np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
        if a.nbytes != int(spec["nbytes"]):
            raise ValueError(f"array {spec['name']}: size mismatch")
        arrays[spec["name"]] = a
        total += int(spec["nbytes"])
    return _join(head["msg"], arrays), total


class Client:
    """One worker's connection to the perception server (``path`` relative to the worker's working folder)."""

    def __init__(self, path: str, timeout_s: float = 600.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(path)
        self.bytes_out = 0
        self.bytes_in = 0

    def call(self, msg: dict) -> dict:
        self.bytes_out += send(self.sock, msg)
        rep, n = recv(self.sock)
        self.bytes_in += n
        if rep.get("error"):
            raise RemoteError(rep["error"])
        return rep

    def close(self):
        try:
            send(self.sock, {"op": "bye"})
        except OSError:
            pass
        self.sock.close()


class RemoteError(RuntimeError):
    """An exception inside the perception server while it handled a request (the episode's perception_error)."""


def serve(path: str, handler, *, max_clients: int = 2, log=None, require_hello: bool = False) -> None:
    """Accept connections on ``path`` (a relative name inside the server's working folder) and answer each request
    with ``handler(msg) -> dict`` in arrival order; an exception becomes ``{"error": ...}``. ``op: shutdown`` stops.
    With ``require_hello`` a connection gets only errors until its ``hello`` has succeeded."""
    import os
    import selectors
    import traceback

    greeted: set = set()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(max_clients)
    sel = selectors.DefaultSelector()
    sel.register(srv, selectors.EVENT_READ, None)
    running = True
    clients = 0
    try:
        while running:
            for key, _ in sel.select(timeout=1.0):
                if key.data is None:
                    conn, _addr = srv.accept()
                    if clients >= max_clients:
                        conn.close()
                        continue
                    clients += 1
                    sel.register(conn, selectors.EVENT_READ, "client")
                    continue
                conn = key.fileobj
                try:
                    msg, _n = recv(conn)
                except (ConnectionError, OSError, ValueError):
                    sel.unregister(conn)
                    greeted.discard(id(conn))
                    conn.close()
                    clients -= 1
                    continue
                op = msg.get("op")
                if op == "bye":
                    sel.unregister(conn)
                    greeted.discard(id(conn))
                    conn.close()
                    clients -= 1
                    continue
                if require_hello and op not in ("hello", "shutdown") and id(conn) not in greeted:
                    try:
                        send(conn, {"error": f"HelloRequired: no accepted hello on this connection (op {op!r})"})
                    except OSError:
                        pass
                    continue
                if op == "shutdown":
                    try:
                        send(conn, {"ok": True})
                    except OSError:
                        pass
                    running = False
                    break
                try:
                    rep = handler(msg)
                    if op == "hello" and not rep.get("error"):
                        greeted.add(id(conn))
                except Exception as exc:  # returned, never hidden: the episode's perception_error
                    rep = {"error": f"{type(exc).__name__}: {exc}",
                           "traceback": "".join(traceback.format_exception(exc))[-4000:]}
                    if log:
                        log(rep)
                try:
                    send(conn, rep)
                except OSError:
                    sel.unregister(conn)
                    greeted.discard(id(conn))
                    conn.close()
                    clients -= 1
    finally:
        sel.close()
        srv.close()
        try:
            os.unlink(path)
        except OSError:
            pass
