"""M2 record stamping (verifier minor M3): every record carries the UTC time it was actually written.

M2 (perceived poses, grounded instruction): perception in the loop. Standard library only (both venvs; runner side).
The verifier found two records whose time fields did not match their write times: ``plan-amendments.json`` was first
written 78 s after the freeze although its entries read earlier times, and the gate precheck said 23:30:00Z but was
written at 23:25:57Z. From the fix round on, every M2 writer goes through ``write_json`` (or ``stamp``), which sets
``written_utc`` from the clock at the moment of the write, and keeps any caller-supplied decision time under its own
key.

``append_amendment`` appends one amendment to ``evidence-m2/plan-amendments.json`` AT THE MOMENT it is made, with the
real UTC, and ``append_erratum`` appends an erratum about an earlier record (the earlier record itself is never
rewritten).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stamp(doc: dict) -> dict:
    """``doc`` with ``written_utc`` set to now (a decision time the caller gave stays under its own key)."""
    out = dict(doc)
    out["written_utc"] = now_utc()
    return out


def write_json(path, doc: dict, *, indent: int = 1, sort_keys: bool = True) -> str:
    """Write ``doc`` stamped with its actual write time, atomically (tmp + rename); returns the file's sha256."""
    path = Path(path)
    data = json.dumps(stamp(doc), indent=indent, sort_keys=sort_keys).encode("utf-8")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def append_amendment(path, *, amend_id: str, what: str, why: str, by: str, extra: dict | None = None) -> dict:
    """Append one amendment now: ``time_utc`` and ``written_utc`` are both the clock at this call."""
    path = Path(path)
    doc = _load(path)
    ids = {a["id"] for a in doc.get("amendments", [])}
    if amend_id in ids:
        raise ValueError(f"amendment {amend_id} already exists")
    t = now_utc()
    entry = {"id": amend_id, "time_utc": t, "written_utc": t, "by": by, "what": what, "why": why,
             "moves_no_bar": "moves no bar, denominator, split, gate-state choice, step cap, success predicate or "
                             "category rule; made on DEV states 0-9 only, before any GATE read"}
    if extra:
        entry.update(extra)
    doc.setdefault("amendments", []).append(entry)
    data = json.dumps(doc, indent=1).encode("utf-8")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return entry


def append_erratum(path, *, err_id: str, about: str, what: str, by: str) -> dict:
    path = Path(path)
    doc = _load(path)
    if any(e["id"] == err_id for e in doc.get("errata", [])):
        raise ValueError(f"erratum {err_id} already exists")
    t = now_utc()
    entry = {"id": err_id, "written_utc": t, "by": by, "about": about, "what": what}
    doc.setdefault("errata", []).append(entry)
    data = json.dumps(doc, indent=1).encode("utf-8")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return entry


POST_FREEZE_FORMAT = "rowant libero_env M2 post-freeze ledger v1"


def append_post_freeze(path, *, entry_id: str, what: str, why: str, by: str, extra: dict | None = None) -> dict:
    """m2r3:fix-r2 (A89, verifier PV-M3): one entry of the POST-FREEZE ledger (a file no freeze binds, e.g.
    ``evidence-m2/post-freeze-log.json``), appended at the moment it is made with the real UTC: what a frozen step
    records after its freeze (a declaration made after a reading, a crash continuation, a data push, a stop by PID),
    since appending to the bound amendments file would break the freeze."""
    path = Path(path)
    doc = _load(path) if path.exists() else {"format": POST_FREEZE_FORMAT, "entries": []}
    if any(e["id"] == entry_id for e in doc["entries"]):
        raise ValueError(f"post-freeze entry {entry_id} already exists")
    t = now_utc()
    entry = {"id": entry_id, "time_utc": t, "written_utc": t, "by": by, "what": what, "why": why, **(extra or {})}
    doc["entries"].append(entry)
    data = json.dumps(doc, indent=1).encode("utf-8")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return entry


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="append an M2 plan amendment or erratum with the real UTC")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("amend")
    a.add_argument("--file", required=True)
    a.add_argument("--id", required=True)
    a.add_argument("--what", required=True)
    a.add_argument("--why", required=True)
    a.add_argument("--by", required=True)
    a.add_argument("--extra", default=None, help="JSON object merged into the entry")
    e = sub.add_parser("erratum")
    e.add_argument("--file", required=True)
    e.add_argument("--id", required=True)
    e.add_argument("--about", required=True)
    e.add_argument("--what", required=True)
    e.add_argument("--by", required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "amend":
        out = append_amendment(ns.file, amend_id=ns.id, what=ns.what, why=ns.why, by=ns.by,
                               extra=json.loads(ns.extra) if ns.extra else None)
    else:
        out = append_erratum(ns.file, err_id=ns.id, about=ns.about, what=ns.what, by=ns.by)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
