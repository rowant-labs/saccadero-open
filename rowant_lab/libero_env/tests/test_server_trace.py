"""The server's traceback kept on a failed request (2026-11-18, record only): ``dev_episode.TracedClient`` and the row's
``perception_trace`` (``dev_episode.server_trace``)."""

import pytest

from rowant_lab.libero_env import m2_bridge as BR
from rowant_lab.pro2 import dev_episode as DE

TEXT = ('Traceback (most recent call last):\n  File "/repo/rowant_lab/libero_env/m2_bridge.py", line 209, in serve\n'
        '    rep = handler(msg)\n  File "/repo/rowant_lab/libero_env/m2_perceive.py", line 1890, in handle\n'
        '    return self.full(msg)\n  File "/repo/rowant_lab/fastlayer/icp.py", line 96, in icp\n'
        '    vis = RF.visible(Xw, Nw, fr)\nIndexError: index 349929 is out of bounds for axis 0 with size 116964\n')


def test_a_failed_request_keeps_the_servers_traceback_and_the_row_names_the_place(monkeypatch):
    replies = [({"error": "IndexError: index 349929 is out of bounds", "traceback": TEXT}, 10), ({"ok": True}, 5)]
    monkeypatch.setattr(BR, "send", lambda sock, msg: 7)
    monkeypatch.setattr(BR, "recv", lambda sock: replies.pop(0))
    c = object.__new__(DE.TracedClient)
    c.sock, c.bytes_out, c.bytes_in = None, 0, 0
    with pytest.raises(BR.RemoteError) as e:
        c.call({"op": "look"})
    assert str(e.value).startswith("IndexError") and e.value.server_trace == TEXT
    assert DE.server_trace(e.value) == ["m2_bridge.py:209 serve", "m2_perceive.py:1890 handle", "icp.py:96 icp"]
    assert c.call({"op": "look"}) == {"ok": True} and (c.bytes_out, c.bytes_in) == (14, 15)  # (as the bridge's client)
    assert DE.server_trace(BR.RemoteError("no traceback sent")) is None and DE.server_trace(ValueError("x")) is None
