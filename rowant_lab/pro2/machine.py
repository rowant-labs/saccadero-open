"""Runner side: where the benchmark's download, the two interpreters and the depth nets are on this machine, and
the step that fits a registered parameter text to this machine. Standard library only.

Environment variables (``configs/paths.example.env`` lists them with examples):

* ``ROWANT_DATA``: the data folder (``rowant_lab/settings.py``).
* ``ROWANT_BENCHMARK``: the unpacked LIBERO-Pro2.0 evaluation download, the folder that holds ``simulator-source``
  and ``service-source``. Default: ``<data>/benchmark/LIBERO-Pro-evaluation``.
* ``ROWANT_WORKER_PYTHON``, ``ROWANT_SERVER_PYTHON``: the interpreters of the two environments
  (``requirements/worker.txt``, ``requirements/server.txt``). Defaults: ``<data>/venvs/worker/bin/python`` and
  ``<data>/venvs/server/bin/python``.
* ``ROWANT_EXPORT_PYTHON``: the interpreter of the environment that builds the object libraries' simulator steps
  (``requirements/export.txt``: the worker's packages with MuJoCo 3.3.2, the version the libraries of the scored
  runs were made with; the episodes themselves ran MuJoCo 3.2.3). Default: ``<data>/venvs/export/bin/python``.
* ``ROWANT_NETS``: the folder with the two depth nets, ``colour-long.pt`` and ``colour2.pt``. Default:
  ``<data>/nets``.
* ``ROWANT_BUNDLE``: the folder that holds the benchmark's archive ``evaluation.zip`` as it was downloaded (needed
  only to compare its checksum with the registered one, and for the test split). Default: ``<data>/benchmark``.

The two registered parameter texts (``configs/v9.registered.json``, ``configs/v11.registered.json``) are published
byte for byte, so that their checksums equal the ones registered before each scored run. They name four places as
they were on the machines of those runs (``REGISTERED``). ``localize`` returns a copy with those four values
replaced by this machine's; nothing else is changed, and a text with any other value in one of those places is
refused.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from rowant_lab.settings import DATA

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(os.environ.get("ROWANT_BENCHMARK") or DATA / "benchmark" / "LIBERO-Pro-evaluation").expanduser()
WORKER_PYTHON = os.environ.get("ROWANT_WORKER_PYTHON") or str(DATA / "venvs" / "worker" / "bin" / "python")
SERVER_PYTHON = os.environ.get("ROWANT_SERVER_PYTHON") or str(DATA / "venvs" / "server" / "bin" / "python")
EXPORT_PYTHON = os.environ.get("ROWANT_EXPORT_PYTHON") or str(DATA / "venvs" / "export" / "bin" / "python")
NETS = Path(os.environ.get("ROWANT_NETS") or DATA / "nets").expanduser()
BUNDLE = Path(os.environ.get("ROWANT_BUNDLE") or DATA / "benchmark").expanduser()  # holds evaluation.zip as downloaded
LONG_LIB = DATA / "runs" / "pro2" / "long-lib"  # the object libraries the scored configurations read

# the four places a registered text names, as they were on the machines of the scored runs
REGISTERED = {
    "robot_dir": "/workspace/monty/rowant_lab/pro2/robot",
    "lfb_library": "/Volumes/RowantNVMe/rowant-cold/runs/pro2/long-lib",
    "net": "/workspace/out/colour-long/net/last.pt",
    "net2": "/workspace/out/colour2/net/last.pt",
}


def local() -> dict:
    """This machine's values for the four places."""
    return {"robot_dir": str(REPO_ROOT / "rowant_lab" / "pro2" / "robot"), "lfb_library": str(LONG_LIB),
            "net": str(NETS / "colour-long.pt"), "net2": str(NETS / "colour2.pt")}


def localize(params: dict, here: dict | None = None) -> dict:
    """A copy of a registered parameter set with its four places set to this machine's (``here``: ``local()``)."""
    here = here or local()
    p = copy.deepcopy(params)
    spots = [(p, "robot_dir", "robot_dir"), (p, "lfb_library", "lfb_library"),
             (p.get("server") or {}, "lfb_library", "lfb_library"),
             (p.get("colour") or {}, "net", "net"), (p.get("colour") or {}, "net2", "net2")]
    for holder, key, name in spots:
        if key not in holder:
            continue
        if holder[key] not in (REGISTERED[name], here[name]):
            raise ValueError(f"{key!r} holds {holder[key]!r}: neither the registered value nor this machine's")
        holder[key] = here[name]
    return p
