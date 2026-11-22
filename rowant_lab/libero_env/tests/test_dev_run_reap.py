"""A worker slot's leftover perception servers (``dev_run.stale_servers``, 2026-11-18): a worker the kernel kills for
memory cannot stop its own server, and 48 of them once filled a pod."""
import inspect

from rowant_lab.pro2 import dev_run as DR


def fake_proc(root, table):
    for name, args in table.items():
        d = root / name
        d.mkdir(parents=True)
        if args is not None:
            (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
    return root


def server(log_dir):
    return ["/venv/bin/python", "-P", "-m", "rowant_lab.libero_env.m2_perceive", "--socket", "p.sock", "--log-dir",
            str(log_dir), "--suites", "libero_goal"]


def test_only_this_slots_servers_are_named(tmp_path):
    out = tmp_path / "run"
    (out / "w03" / "server").mkdir(parents=True)
    (out / "w04" / "server").mkdir(parents=True)
    proc = fake_proc(tmp_path / "proc", {
        "101": server(out / "w03" / "server"),
        "102": server(out / "w04" / "server"),  # another slot's
        "103": ["/venv/bin/python", "-I", "-B", "dev_run.py", "worker", "--out", str(out), "--index", "3"],
        "104": None,  # ended while we looked: no cmdline
        "105": server(out / "w03" / "server"),
        "106": ["python", "-m", "rowant_lab.libero_env.m2_perceive", "--log-dir"],  # cut short
        "self": server(out / "w03" / "server"),  # not a process number
    })
    assert DR.stale_servers(out / "w03", proc) == [101, 105]
    assert DR.stale_servers(out / "w04", proc) == [102]
    assert DR.stale_servers(out / "w05", proc) == []


def test_no_proc_folder_means_no_servers(tmp_path):
    assert DR.stale_servers(tmp_path / "w00", tmp_path / "no-proc") == []


def test_the_run_ends_a_slots_leftovers_before_it_starts_the_slots_worker():
    src = inspect.getsource(DR.cmd_run)
    assert 0 < src.index("stale_servers(") < src.index("os.kill(pid, signal.SIGKILL)") < src.index("subprocess.call(")
