#!/usr/bin/env python3
"""Compare two built object-library trees file by file, or one tree with a stored digest of another.

    python scripts/compare_libraries.py A B [--include GLOB ...] [--index GLOB ...] [--verbose] [--json OUT]
    python scripts/compare_libraries.py --digest A > reference.json     # a path-free digest of tree A
    python scripts/compare_libraries.py reference.json B                # compare tree B with that digest

A is the reference, B the tree being judged (for example a fresh build). Files are matched by their path relative
to the two roots. Each file gets one result:

``identical``     the same bytes.
``same content``  the bytes differ only in what a rebuild cannot repeat and no run depends on (listed below).
``follows``       its own content is the same, but it records the checksum of another file that differs.
``differs``       a real difference; the report says where (which arrays and by how much, which JSON fields).
``missing``       in A, not in B.            ``extra``  in B, not in A.

Why byte equality is not expected everywhere. A library names its textures, meshes and memory files by ABSOLUTE
path, and its build records hold time stamps, timings, machine load readings and the checksums of the builder's own
source files. Those are inside JSON files and inside the JSON metadata of ``memory.npz`` and
``colour-memory.npz``, so two builds of the same library in two places never give the same bytes for them. This
tool therefore compares CONTENT:

* ``.npz``: member by member. A numeric array must have the same bytes (dtype, shape, every value bit for bit). A
  member that holds JSON metadata is compared as JSON (next point). For an array that differs the report gives
  the shape and dtype on both sides, how many elements differ and the largest absolute difference (numpy needed
  for those numbers; without numpy the report says only that the bytes differ).
* ``.json``: parsed and compared field by field after
  - PATH STRINGS are normalised: an absolute path is cut down to the part below the simulator's assets folder,
    below the benchmark download, below the data folder's ``runs`` or below a library folder's own name
    (``normalise_path``);
  - VOLATILE FIELDS are left out: time stamps (``*_utc``), timings (``seconds``, ``elapsed_s``, ...), machine
    readings (``machine_start``, ``machine_end``, ``isolation``) and the checksums of builder code or of build
    records kept outside the tree (``builder_sha256``, ``onboard_sha256``, ...): ``VOLATILE`` is the full list;
    a build record named ``seconds.json`` holds nothing but timings and is treated as such;
  - a differing SHA-256 that is, on each side, the checksum of the same-named file of that tree is not counted
    against this file: it is settled by that other file's own result (``same content``, or ``follows`` if that
    file differs). Any other differing checksum counts as a real difference.
  Every number, string, list length and key that is left must be equal.
* ``.npy`` and every other file: the same bytes, or for text files the same text after path normalisation.

The summary counts how many differences of each ignored kind were met, so nothing is dropped silently. The exit
status is 0 when no file is ``differs``, ``follows``, ``missing`` or ``extra``, else 1.

Standard library only; numpy, when importable, adds the size of numeric differences.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import re
import struct
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
except Exception:  # the comparison is exact without it; only the size of a numeric difference is lost
    np = None

DIGEST_FORMAT = "object library digest v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# an absolute path is cut after the LAST of these markers found in it (the part before it is machine-specific)
PATH_ANCHORS = (("/libero/libero/assets", "<assets>"), ("/simulator-source", "<benchmark>/simulator-source"),
                ("/service-source", "<benchmark>/service-source"), ("/runs", "<data>/runs"),
                # the library folders by their own names: a tree may have been built outside a "runs" folder
                ("/libero-m1", "<data>/runs/libero-m1"), ("/libero-m2", "<data>/runs/libero-m2"),
                ("/goal-lib", "<data>/runs/pro2/goal-lib"), ("/long-lib", "<data>/runs/pro2/long-lib"))
PATH_TOKEN = re.compile(r"/[^\s\"'<>|,;]*")
VOLATILE = {
    "time stamps": ("time_utc", "written_utc", "isolation_utc", "created_utc", "probe_time_utc", "src_time_utc",
                    "time_utc_start", "time_utc_end"),
    "timings": ("seconds", "seconds_per_type", "elapsed_s", "export_s_type", "wall_s"),
    "machine readings": ("machine_start", "machine_end", "isolation"),
    "checksums of builder code or build records": ("builder_sha256", "exporter_sha256", "onboard_sha256",
                                                   "rules_builder_sha256", "src_exporter_sha256", "generator",
                                                   "probe_sha256"),
}
VOLATILE_OF = {k: group for group, keys in VOLATILE.items() for k in keys}
TIMING_FILES = ("seconds.json",)  # build records that hold nothing but seconds per object type
TEXT_SUFFIXES = (".ply", ".yaml", ".yml", ".txt", ".md", ".tsv", ".csv")
SKIP_DIRS = ("numba-cache", "__pycache__")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ normalising
def normalise_path(token: str) -> str:
    """An absolute path with its machine-specific front cut off (see ``PATH_ANCHORS``); other text unchanged."""
    best = None
    for marker, name in PATH_ANCHORS:
        i = token.rfind(marker)
        while i >= 0 and token[i + len(marker):i + len(marker) + 1] not in ("", "/"):
            i = token.rfind(marker, 0, i)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, marker, name)
    if best is None:
        return token
    i, marker, name = best
    return name + token[i + len(marker):]


def normalise_text(s: str) -> str:
    return PATH_TOKEN.sub(lambda m: normalise_path(m.group(0)), s)


def canonical(doc):
    """A JSON document as the digest stores it: paths normalised, volatile fields dropped, checksums blanked."""
    if isinstance(doc, dict):
        return {k: canonical(v) for k, v in doc.items() if k not in VOLATILE_OF}
    if isinstance(doc, list):
        return [canonical(v) for v in doc]
    if isinstance(doc, str):
        return "<sha256>" if HEX64.match(doc) else normalise_text(doc)
    return doc


def canonical_hash(doc) -> str:
    return sha256_bytes(json.dumps(canonical(doc), sort_keys=True).encode())


# ------------------------------------------------------------------------------------------------ JSON
def json_diff(a, b, path: str, out: list) -> None:
    """Append ``(kind, where, detail)`` for every difference between two JSON values. Kinds: 'path' (equal after
    path normalisation), 'volatile:<group>', 'checksum' (two different SHA-256 values: settled by the caller),
    'number', 'value', 'structure'."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            where = f"{path}.{k}"
            if k in VOLATILE_OF:
                if a.get(k) != b.get(k):
                    out.append(("volatile:" + VOLATILE_OF[k], where, None))
            elif k not in a or k not in b:
                out.append(("structure", where, "only in " + ("A" if k in a else "B")))
            else:
                json_diff(a[k], b[k], where, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(("structure", path, f"list length {len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff(x, y, f"{path}[{i}]", out)
        return
    if a == b and type(a) is type(b):
        return
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        if a != b:
            out.append(("number", path, abs(float(a) - float(b))))
        return
    if isinstance(a, str) and isinstance(b, str):
        if HEX64.match(a) and HEX64.match(b):
            out.append(("checksum", path, (a, b)))
        elif normalise_text(a) == normalise_text(b):
            out.append(("path", path, None))
        else:
            out.append(("value", path, f"{a[:60]!r} vs {b[:60]!r}"))
        return
    out.append(("value", path, f"{str(a)[:60]!r} vs {str(b)[:60]!r}"))


# ------------------------------------------------------------------------------------------------ npy / npz
def read_npy(raw: bytes):
    """(descr, shape, data bytes) of a .npy file's bytes, without numpy."""
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    if raw[6] == 1:
        n, off = struct.unpack("<H", raw[8:10])[0], 10
    else:
        n, off = struct.unpack("<I", raw[8:12])[0], 12
    head = ast.literal_eval(raw[off:off + n].decode("latin1"))
    return head["descr"], tuple(head["shape"]), raw[off + n:]


def member_json(raw: bytes):
    """The JSON document a .npy member holds as text (a string scalar or a byte vector), else None."""
    try:
        descr, shape, data = read_npy(raw)
        if isinstance(descr, str) and descr[1:2] == "U" and shape == ():
            text = data.decode("utf-32-le" if descr[0] != ">" else "utf-32-be").rstrip("\x00")
        elif descr in ("|u1", "|S1") and len(shape) == 1:
            text = data.decode("utf-8")
        else:
            return None
        doc = json.loads(text)
        return doc if isinstance(doc, (dict, list)) else None
    except Exception:
        return None


def array_detail(ra: bytes, rb: bytes) -> str:
    """How two numeric .npy members differ."""
    try:
        da, sa, _ = read_npy(ra)
        db, sb, _ = read_npy(rb)
    except Exception:
        return "bytes differ"
    if (da, sa) != (db, sb):
        return f"dtype/shape {da}{list(sa)} vs {db}{list(sb)}"
    if np is None:
        return f"{da}{list(sa)}: bytes differ (numpy not available for the size of the difference)"
    a, b = np.load(io.BytesIO(ra), allow_pickle=False), np.load(io.BytesIO(rb), allow_pickle=False)
    if a.dtype.kind not in "biufc":
        return f"{da}{list(sa)}: {int((a != b).sum())} of {a.size} elements differ"
    neq = a != b
    if a.dtype.kind in "fc":
        neq &= ~(np.isnan(a) & np.isnan(b))
    n = int(neq.sum())
    if n == 0:
        return f"{da}{list(sa)}: equal values, bytes differ (signed zeros or NaN payloads)"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)) if a.dtype.kind != "c" else np.abs(a - b)
    return f"{da}{list(sa)}: {n} of {a.size} elements differ, max abs difference {float(np.nanmax(diff)):.3g}"


def npz_members(path) -> dict:
    with zipfile.ZipFile(path) as z:
        return {i.filename: (format(i.CRC, "08x"), i.file_size) for i in z.infolist()}


def compare_npz(pa, pb, diffs: list) -> None:
    ma, mb = npz_members(pa), npz_members(pb)
    for name in sorted(set(ma) ^ set(mb)):
        diffs.append(("structure", f"[{name}]", "array only in " + ("A" if name in ma else "B")))
    with zipfile.ZipFile(pa) as za, zipfile.ZipFile(pb) as zb:
        for name in sorted(set(ma) & set(mb)):
            if ma[name] == mb[name]:
                continue
            ra, rb = za.read(name), zb.read(name)
            if ra == rb:
                continue
            ja, jb = member_json(ra), member_json(rb)
            if ja is not None and jb is not None:
                json_diff(ja, jb, f"[{name}]", diffs)
            else:
                diffs.append(("array", f"[{name}]", array_detail(ra, rb)))


def compare_text(pa, pb, diffs: list) -> None:
    try:
        la = Path(pa).read_text(encoding="utf-8").splitlines()
        lb = Path(pb).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        diffs.append(("value", "", "bytes differ"))
        return
    if len(la) != len(lb):
        diffs.append(("structure", "", f"{len(la)} vs {len(lb)} lines"))
        return
    n_path = n_real = 0
    worst = 0.0
    numeric = True
    for x, y in zip(la, lb):
        if x == y:
            continue
        if normalise_text(x) == normalise_text(y):
            n_path += 1
            continue
        n_real += 1
        tx, ty = x.split(), y.split()
        try:
            if len(tx) != len(ty):
                raise ValueError
            worst = max(worst, max(abs(float(p) - float(q)) for p, q in zip(tx, ty)))
        except ValueError:
            numeric = False
    if n_path:
        diffs.append(("path", f"{n_path} lines", None))
    if n_real:
        diffs.append(("value", f"{n_real} of {len(la)} lines",
                      f"max abs difference {worst:.3g}" if numeric else "text differs"))


# ------------------------------------------------------------------------------------------------ trees
def selected(rel: str, include: list) -> bool:
    return not include or any(fnmatch.fnmatch(rel, g) or rel.startswith(g.rstrip("/") + "/") for g in include)


def list_tree(root: Path, include: list) -> list:
    """The files under ``root`` (relative, sorted) that ``include`` selects; only the folders it names are walked."""
    starts = {""}
    if include:
        starts = set()
        for g in include:
            g = g.rstrip("/")
            m = re.search(r"[*?\[]", g)
            starts.add(g if m is None else g[:m.start()].rpartition("/")[0])
    out = set()
    for start in starts:
        top = root / start if start else root
        if top.is_file():
            out.add(start)
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                rel = (Path(dirpath) / f).relative_to(root).as_posix()
                if selected(rel, include):
                    out.add(rel)
    return sorted(out)


def file_content_diffs(pa: Path, pb: Path) -> list:
    """Every difference between two files of the same relative path whose bytes differ."""
    diffs: list = []
    suffix = pa.suffix.lower()
    try:
        if suffix == ".npz":
            compare_npz(pa, pb, diffs)
        elif suffix == ".json":
            json_diff(json.loads(pa.read_text()), json.loads(pb.read_text()), "", diffs)
            if pa.name in TIMING_FILES:  # nothing but timings, keyed by type name
                diffs[:] = [("volatile:timings", w, None) if k == "number" else (k, w, d) for k, w, d in diffs]
        elif suffix == ".npy":
            diffs.append(("array", "", array_detail(pa.read_bytes(), pb.read_bytes())))
        elif suffix in TEXT_SUFFIXES:
            compare_text(pa, pb, diffs)
        else:
            diffs.append(("value", "", "bytes differ"))
    except Exception as e:
        diffs.append(("value", "", f"could not be read as {suffix}: {type(e).__name__}: {e}"))
    return diffs


def compare_trees(A: Path, B: Path, include: list, index: list | None = None) -> dict:
    fa, fb = list_tree(A, include), list_tree(B, include)
    index = index or []  # the files a recorded checksum may be looked up among: by default every file of the trees
    sha_a = {rel: sha256_file(A / rel) for rel in sorted(set(list_tree(A, index)) | set(fa))}
    sha_b = {rel: sha256_file(B / rel) for rel in sorted(set(list_tree(B, index)) | set(fb))}
    by_sha_a, by_sha_b = {}, {}
    for rel, h in sha_a.items():
        by_sha_a.setdefault(h, set()).add(rel)
    for rel, h in sha_b.items():
        by_sha_b.setdefault(h, set()).add(rel)
    res = {}
    for rel in sorted(set(fa) | set(fb)):
        if rel not in sha_b:
            res[rel] = {"status": "missing"}
        elif rel not in sha_a:
            res[rel] = {"status": "extra"}
        elif sha_a[rel] == sha_b[rel]:
            res[rel] = {"status": "identical"}
        else:
            res[rel] = {"status": None, "diffs": file_content_diffs(A / rel, B / rel)}

    def pair_status(rel: str, seen: tuple = ()) -> str:
        """The result of a file both trees hold, settled on demand (a checksum may point at any file of the trees)."""
        if rel not in res:
            if sha_a.get(rel) == sha_b.get(rel):
                res[rel] = {"status": "identical", "outside_selection": True}
            else:
                res[rel] = {"status": None, "diffs": file_content_diffs(A / rel, B / rel), "outside_selection": True}
        r = res[rel]
        if r["status"] is not None:
            return r["status"]
        if rel in seen:
            return "same content"
        real, follows, ignored = [], [], {}
        for kind, where, detail in r["diffs"]:
            if kind == "path":
                ignored["path strings"] = ignored.get("path strings", 0) + 1
            elif kind.startswith("volatile:"):
                ignored[kind[9:]] = ignored.get(kind[9:], 0) + 1
            elif kind == "checksum":
                both = by_sha_a.get(detail[0], set()) & by_sha_b.get(detail[1], set())
                if not both:
                    real.append((where, "a checksum that is no file's of these trees differs"))
                    continue
                other = sorted(both)[0]
                st = pair_status(other, seen + (rel,)) if other != rel else "same content"
                if st in ("differs", "follows"):
                    follows.append((where, other))
                else:
                    ignored["checksums of same-content files"] = ignored.get("checksums of same-content files", 0) + 1
            else:
                real.append((where, detail if isinstance(detail, str) else f"numbers differ by {detail:.3g}"))
        r["ignored"], r["real"], r["follows"] = ignored, real, follows
        r["status"] = "differs" if real else "follows" if follows else "same content"
        return r["status"]

    for rel in list(res):
        if res[rel]["status"] is None:
            pair_status(rel)
    return {rel: r for rel, r in res.items() if not r.get("outside_selection")}


# ------------------------------------------------------------------------------------------------ digest
def file_digest(p: Path) -> dict:
    """What the digest keeps of one file: its sha256 and size, and a path-free digest of its content."""
    out = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    suffix = p.suffix.lower()
    try:
        if suffix == ".npz":
            members = {}
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    doc = member_json(z.read(i.filename)) if i.file_size < (1 << 22) else None
                    members[i.filename] = (["json", canonical_hash(doc)] if doc is not None
                                           else [format(i.CRC, "08x"), i.file_size])
            out["members"] = members
        elif suffix == ".json":
            out["content"] = "timings" if p.name in TIMING_FILES else canonical_hash(json.loads(p.read_text()))
        elif suffix in TEXT_SUFFIXES:
            out["content"] = sha256_bytes(normalise_text(p.read_text(encoding="utf-8")).encode())
    except Exception as e:
        out["unreadable"] = f"{type(e).__name__}: {e}"
    return out


def tree_digest(root: Path, include: list) -> dict:
    return {"format": DIGEST_FORMAT, "note": "relative paths, sizes and checksums only; see compare_libraries.py",
            "include": list(include), "files": {rel: file_digest(root / rel) for rel in list_tree(root, include)}}


def compare_with_digest(ref: dict, B: Path, include: list) -> dict:
    include = include or ref.get("include") or []  # by default the selection the digest was written with
    want = {rel: d for rel, d in ref["files"].items() if selected(rel, include)}
    have = set(list_tree(B, include))
    res = {}
    for rel in sorted(set(want) | have):
        if rel not in have:
            res[rel] = {"status": "missing"}
            continue
        if rel not in want:
            res[rel] = {"status": "extra"}
            continue
        d, mine = want[rel], file_digest(B / rel)
        if d["sha256"] == mine["sha256"]:
            res[rel] = {"status": "identical"}
        elif "members" in d and "members" in mine:
            bad = sorted(k for k in set(d["members"]) | set(mine["members"])
                         if d["members"].get(k) != mine["members"].get(k))
            res[rel] = ({"status": "differs", "real": [(f"[{k}]", "array or metadata differs from the digest")
                                                       for k in bad], "follows": [], "ignored": {}}
                        if bad else {"status": "same content", "ignored": {"metadata (paths, times)": 1}})
        elif "content" in d and d["content"] == mine.get("content"):
            res[rel] = {"status": "same content", "ignored": {"paths, times, checksums": 1}}
        else:
            res[rel] = {"status": "differs", "real": [("", "content differs from the digest")], "follows": [],
                        "ignored": {}}
    return res


# ------------------------------------------------------------------------------------------------ report
ORDER = ("identical", "same content", "follows", "differs", "missing", "extra")


def group_of(rel: str, depth: int) -> str:
    parts = rel.split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or "."


def report(res: dict, depth: int, verbose: bool, max_lines: int) -> None:
    groups: dict = {}
    for rel, r in res.items():
        g = groups.setdefault(group_of(rel, depth), dict.fromkeys(ORDER, 0))
        g[r["status"]] += 1
    width = max([len(g) for g in groups] + [5])
    print(f"{'folder':{width}s}  " + "  ".join(f"{s:>12s}" for s in ORDER))
    for g in sorted(groups):
        print(f"{g:{width}s}  " + "  ".join(f"{groups[g][s]:12d}" for s in ORDER))
    total = {s: sum(g[s] for g in groups.values()) for s in ORDER}
    print(f"{'TOTAL':{width}s}  " + "  ".join(f"{total[s]:12d}" for s in ORDER))
    print(f"{len(res)} files")
    ignored: dict = {}
    for r in res.values():
        if r["status"] in ("same content", "follows", "differs"):
            for k, n in (r.get("ignored") or {}).items():
                e = ignored.setdefault(k, [0, 0])
                e[0] += 1
                e[1] += n
    if ignored:
        print("\nDifferences not counted against a file (files, fields):")
        for k in sorted(ignored):
            print(f"  {k}: {ignored[k][0]} files, {ignored[k][1]} fields")
    for status in ("differs", "follows", "missing", "extra"):
        rels = [rel for rel, r in res.items() if r["status"] == status]
        if not rels:
            continue
        print(f"\n{status.upper()} ({len(rels)}):")
        for rel in rels:
            r = res[rel]
            if status == "follows":
                others = sorted({o for _w, o in r["follows"]})
                print(f"  {rel}: holds the checksum of {len(others)} differing file(s): {', '.join(others[:3])}"
                      f"{' ...' if len(others) > 3 else ''}")
            elif status == "differs":
                print(f"  {rel}: {len(r['real'])} difference(s)")
                for where, detail in r["real"][:None if verbose else max_lines]:
                    print(f"      {where or '(file)'}: {detail}")
                if not verbose and len(r["real"]) > max_lines:
                    print(f"      ... {len(r['real']) - max_lines} more (--verbose)")
            else:
                print(f"  {rel}")
    if verbose:
        rels = [rel for rel, r in res.items() if r["status"] == "same content"]
        print(f"\nSAME CONTENT ({len(rels)}):")
        for rel in rels:
            print(f"  {rel}: " + ", ".join(f"{k} x{n}" for k, n in sorted(res[rel].get("ignored", {}).items())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare two object-library trees (or a digest and a tree) by content.",
                                 epilog="See the top of this file for what each result means.")
    ap.add_argument("a", help="the reference: a library folder, or a digest file written with --digest")
    ap.add_argument("b", nargs="?", help="the tree to judge")
    ap.add_argument("--digest", action="store_true", help="print a path-free digest of tree A as JSON and stop")
    ap.add_argument("--include", action="append", default=[],
                    help="only these relative paths: a folder or a glob (may be repeated)")
    ap.add_argument("--index", action="append", default=[],
                    help="where a recorded checksum is looked up: folders or globs under the roots (may be repeated; "
                         "default: every file of the two trees, so name folders when a root holds much else)")
    ap.add_argument("--depth", type=int, default=3, help="folder depth of the summary table (default 3)")
    ap.add_argument("--verbose", action="store_true", help="list every difference and every same-content file")
    ap.add_argument("--max-lines", type=int, default=6, help="differences shown per file without --verbose")
    ap.add_argument("--json", default=None, help="also write every file's result to this JSON file")
    a = ap.parse_args(argv)
    A = Path(a.a)
    if a.digest:
        print(json.dumps(tree_digest(A, a.include), indent=1, sort_keys=True))
        return 0
    if not a.b:
        ap.error("two arguments are needed: the reference and the tree to judge")
    B = Path(a.b)
    if not B.is_dir():
        ap.error(f"{B} is not a folder")
    if A.is_file():
        ref = json.loads(A.read_text())
        if ref.get("format") != DIGEST_FORMAT:
            ap.error(f"{A} is not a digest written by this tool")
        print(f"A (reference digest): {A}\nB: {B}\n")
        res = compare_with_digest(ref, B, a.include)
    else:
        print(f"A (reference): {A}\nB: {B}\n")
        res = compare_trees(A, B, a.include, a.index)
    report(res, a.depth, a.verbose, a.max_lines)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return 0 if all(r["status"] in ("identical", "same content") for r in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
